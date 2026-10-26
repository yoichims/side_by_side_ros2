// Differential-drive odometry simulator.
//
// Integrates cmd_vel from a configurable initial pose and publishes the result
// as odometry plus the odom -> base_link transform. Because integration starts
// at the initial pose, the odom frame origin coincides with the map origin, so
// map -> odom is identity and a static transform stands in for the localizer
// that the real vehicle would run.

#include <cmath>
#include <memory>
#include <string>

#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>

namespace
{
constexpr char DEFAULT_ODOM_FRAME_ID[] = "odom";
constexpr char DEFAULT_BASE_FRAME_ID[] = "base_link";
constexpr double DEFAULT_INITIAL_POSE_X = 0.0;
constexpr double DEFAULT_INITIAL_POSE_Y = 0.0;
constexpr double DEFAULT_INITIAL_POSE_THETA = 0.0;
constexpr double UPDATE_RATE_HZ = 40.0;

geometry_msgs::msg::Quaternion yawToQuaternion(double yaw)
{
  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, yaw);
  return geometry_msgs::msg::Quaternion(tf2::toMsg(q));
}
}  // namespace

class OdoSimulatorNode : public rclcpp::Node
{
public:
  OdoSimulatorNode();

private:
  void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
  void update();

  double dv_{0.0};   // commanded
  double dw_{0.0};
  double v_{0.0};    // actual, lagging the command
  double w_{0.0};
  double tau_v_{0.41};
  double tau_w_{0.49};
  double x_{0.0};
  double y_{0.0};
  double theta_{0.0};
  rclcpp::Time last_timestamp_;
  bool have_last_timestamp_{false};

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_;

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr vel_sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> odom_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;

  geometry_msgs::msg::TransformStamped odom_trans_;
  nav_msgs::msg::Odometry odom_;
};

OdoSimulatorNode::OdoSimulatorNode()
: rclcpp::Node("odo_simulator")
{
  const std::string odom_frame_id =
    this->declare_parameter<std::string>("odom_frame_id", DEFAULT_ODOM_FRAME_ID);
  const std::string base_frame_id =
    this->declare_parameter<std::string>("base_frame_id", DEFAULT_BASE_FRAME_ID);

  // First-order actuation lag, fitted from the recorded WHILL logs (stage 7:
  // T = 0.41 s linear, 0.49 s angular, both to within a few percent across
  // sessions). Integrating the commanded velocity directly, as this node used
  // to, makes the vehicle reach any commanded speed instantly -- while the
  // planner's rollouts are generated *through* this same lag. The planner then
  // predicts a ramp, the vehicle snaps, and every pair term it computed off
  // those predictions is wrong. Set either constant to 0 for the old
  // instantaneous behaviour.
  tau_v_ = this->declare_parameter<double>("tau_v", tau_v_);
  tau_w_ = this->declare_parameter<double>("tau_w", tau_w_);

  x_ = this->declare_parameter<double>("initialPoseX", DEFAULT_INITIAL_POSE_X);
  y_ = this->declare_parameter<double>("initialPoseY", DEFAULT_INITIAL_POSE_Y);
  theta_ = this->declare_parameter<double>("initialPoseTheta", DEFAULT_INITIAL_POSE_THETA);

  vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
    "~/cmd_vel", 1,
    std::bind(&OdoSimulatorNode::cmdVelCallback, this, std::placeholders::_1));

  // Odometry carries the pose and both velocities, and is what every consumer
  // actually reads: trajectory_tracker and sbs_agent both subscribe to it.
  //
  // The ROS 1 node also published vehicle_speed, vehicle_yawrate,
  // vehicle_acceleration, current_pose and filtered_ndt_current_pose. None of
  // them had a consumer after the port -- sbs_agent takes velocity from odom
  // and its vehicle_speed/vehicle_yawrate subscriptions are commented out, and
  // the pose it reads is published by unit_converter's tfToPoseStamped at the
  // namespace level. The last one was actively misleading: a private
  // <ns>/odo_simple_simulator/filtered_ndt_current_pose looks like localizer
  // output but is dead-reckoned ground truth that nothing subscribes to.
  pub_ = this->create_publisher<nav_msgs::msg::Odometry>("~/base_pose_ground_truth", 1);

  odom_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  odom_trans_.header.frame_id = odom_frame_id;
  odom_trans_.child_frame_id = base_frame_id;
  odom_.header.frame_id = odom_frame_id;
  odom_.child_frame_id = base_frame_id;

  timer_ = this->create_wall_timer(
    std::chrono::duration<double>(1.0 / UPDATE_RATE_HZ),
    std::bind(&OdoSimulatorNode::update, this));

  RCLCPP_INFO(this->get_logger(), "Differential drive odometry simulator started");
}

void OdoSimulatorNode::cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
  dv_ = msg->linear.x;
  dw_ = msg->angular.z;
}

void OdoSimulatorNode::update()
{
  const rclcpp::Time current_time = this->now();

  // The ROS 1 loop integrated with the delta measured on the previous
  // iteration; at a steady 40 Hz that is the same number, but measuring it up
  // front also avoids the first-iteration division by zero below.
  double delta_time = 0.0;
  if (have_last_timestamp_) {
    delta_time = (current_time - last_timestamp_).seconds();
  }
  last_timestamp_ = current_time;
  have_last_timestamp_ = true;

  // Actuation lag, then integrate what the wheels are actually doing.
  const double alpha_v =
    (tau_v_ > 0.0) ? std::exp(-delta_time / tau_v_) : 0.0;
  const double alpha_w =
    (tau_w_ > 0.0) ? std::exp(-delta_time / tau_w_) : 0.0;
  v_ = alpha_v * v_ + (1.0 - alpha_v) * dv_;
  w_ = alpha_w * w_ + (1.0 - alpha_w) * dw_;

  theta_ += w_ * delta_time;
  x_ += v_ * std::cos(theta_) * delta_time;
  y_ += v_ * std::sin(theta_) * delta_time;

  const geometry_msgs::msg::Quaternion odom_quat = yawToQuaternion(theta_);

  odom_trans_.header.stamp = current_time;
  odom_trans_.transform.translation.x = x_;
  odom_trans_.transform.translation.y = y_;
  odom_trans_.transform.translation.z = 0.0;
  odom_trans_.transform.rotation = odom_quat;
  odom_broadcaster_->sendTransform(odom_trans_);

  odom_.header.stamp = current_time;
  odom_.pose.pose.position.x = x_;
  odom_.pose.pose.position.y = y_;
  odom_.pose.pose.position.z = 0.0;
  odom_.pose.pose.orientation = odom_quat;
  odom_.twist.twist.linear.x = v_;
  odom_.twist.twist.linear.y = 0.0;
  odom_.twist.twist.angular.z = w_;
  pub_->publish(odom_);
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OdoSimulatorNode>());
  rclcpp::shutdown();
  return 0;
}
