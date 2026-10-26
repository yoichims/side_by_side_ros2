// Simulated 2D laser scanner (Hokuyo UTM geometry).
//
// Raycasts the occupancy grid served by the map server from the sensor's
// current pose and publishes the result as a LaserScan, so obstacle avoidance
// can be exercised without the physical LRF.

#include <cmath>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace
{
constexpr int NUM_READINGS_PER_SCAN = 1081;
constexpr double FREQUENCY = 40.0;
constexpr double MAX_RANGE_M = 60.0;
constexpr double MIN_RANGE_M = 0.023;
constexpr double MAX_ANGLE_RAD = 2.35619449;
constexpr double MIN_ANGLE_RAD = -2.35619449;
constexpr char DEFAULT_FRAME_ID[] = "laser";
constexpr char DEFAULT_MAP_TOPIC[] = "map";
constexpr char DEFAULT_DYNAMIC_OBSTACLE_TOPIC[] = "dynamic_obstacles";

constexpr int OCCUPIED = 100;
constexpr int UNKNOWN = -1;
}  // namespace

class UTMSimulatorNode : public rclcpp::Node
{
public:
  UTMSimulatorNode();

private:
  int value(int index) const;
  int value(int row, int column) const;
  void raycast(
    const geometry_msgs::msg::PointStamped & start,
    geometry_msgs::msg::PointStamped & end) const;
  static double distance(
    const geometry_msgs::msg::PointStamped & p1,
    const geometry_msgs::msg::PointStamped & p2);
  void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
  void dynamicObstaclesCallback(const sensor_msgs::msg::PointCloud::SharedPtr msg);
  void update();

  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr pub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr dynamic_obstacles_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  sensor_msgs::msg::LaserScan scan_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::string frame_id_;

  bool map_loaded_{false};
  mutable std::mutex mutex_;
  double resolution_{0.0};
  double origin_x_{0.0};
  double origin_y_{0.0};
  int width_{0};
  int height_{0};
  std::vector<int> static_data_;
  std::vector<int> data_;
  std::string global_frame_id_;
};

UTMSimulatorNode::UTMSimulatorNode()
: rclcpp::Node("utm_simulator")
{
  frame_id_ = this->declare_parameter<std::string>("frame_id", DEFAULT_FRAME_ID);
  const std::string map_topic =
    this->declare_parameter<std::string>("map_topic", DEFAULT_MAP_TOPIC);
  const std::string dynamic_obstacles_topic =
    this->declare_parameter<std::string>("dynamic_obstacles_topic", DEFAULT_DYNAMIC_OBSTACLE_TOPIC);

  pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>("~/scan", 20);

  // The map server latches its map, so a transient-local subscription is
  // required to receive the one message it published before this node started.
  auto map_qos = rclcpp::QoS(1).transient_local().reliable();
  map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
    map_topic, map_qos,
    std::bind(&UTMSimulatorNode::mapCallback, this, std::placeholders::_1));

  dynamic_obstacles_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud>(
    dynamic_obstacles_topic, 1,
    std::bind(&UTMSimulatorNode::dynamicObstaclesCallback, this, std::placeholders::_1));

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  scan_.header.frame_id = frame_id_;
  scan_.angle_min = MIN_ANGLE_RAD;
  scan_.angle_max = MAX_ANGLE_RAD;
  scan_.angle_increment = (MAX_ANGLE_RAD - MIN_ANGLE_RAD) / (NUM_READINGS_PER_SCAN - 1);
  scan_.time_increment = (1.0 / FREQUENCY) / NUM_READINGS_PER_SCAN;
  scan_.range_min = MIN_RANGE_M;
  scan_.range_max = MAX_RANGE_M;
  scan_.ranges.resize(NUM_READINGS_PER_SCAN);

  timer_ = this->create_wall_timer(
    std::chrono::duration<double>(1.0 / FREQUENCY),
    std::bind(&UTMSimulatorNode::update, this));
}

void UTMSimulatorNode::mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  resolution_ = msg->info.resolution;
  origin_x_ = msg->info.origin.position.x;
  origin_y_ = msg->info.origin.position.y;
  width_ = static_cast<int>(msg->info.width);
  height_ = static_cast<int>(msg->info.height);
  global_frame_id_ = msg->header.frame_id;

  static_data_.assign(msg->data.begin(), msg->data.end());
  data_ = static_data_;
  map_loaded_ = true;

  RCLCPP_INFO(
    this->get_logger(), "Map received: %dx%d at %.3f m/cell in frame '%s'",
    width_, height_, resolution_, global_frame_id_.c_str());
}

void UTMSimulatorNode::dynamicObstaclesCallback(const sensor_msgs::msg::PointCloud::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (!map_loaded_) {
    return;
  }

  data_ = static_data_;
  for (const auto & point : msg->points) {
    const int r = static_cast<int>((point.x - origin_x_) / resolution_);
    const int c = static_cast<int>((point.y - origin_y_) / resolution_);
    if (r < 0 || r >= width_ || c < 0 || c >= height_) {
      continue;
    }
    data_[r + c * width_] = OCCUPIED;
  }
}

int UTMSimulatorNode::value(int index) const
{
  if (index < 0 || index >= static_cast<int>(data_.size())) {
    return UNKNOWN;
  }
  return data_[index];
}

int UTMSimulatorNode::value(int i, int j) const
{
  if (j > height_ - 1 || j < 0 || i > width_ - 1 || i < 0) {
    return UNKNOWN;
  }
  return value(i + j * width_);
}

double UTMSimulatorNode::distance(
  const geometry_msgs::msg::PointStamped & p1,
  const geometry_msgs::msg::PointStamped & p2)
{
  const double dx = p1.point.x - p2.point.x;
  const double dy = p1.point.y - p2.point.y;
  return std::sqrt(dx * dx + dy * dy);
}

void UTMSimulatorNode::raycast(
  const geometry_msgs::msg::PointStamped & start,
  geometry_msgs::msg::PointStamped & end) const
{
  int r0 = static_cast<int>((start.point.x - origin_x_) / resolution_);
  int c0 = static_cast<int>((start.point.y - origin_y_) / resolution_);

  const int r1 = static_cast<int>((end.point.x - origin_x_) / resolution_);
  const int c1 = static_cast<int>((end.point.y - origin_y_) / resolution_);

  const int dx = std::abs(r1 - r0);
  const int dy = std::abs(c1 - c0);
  const int sx = r0 < r1 ? 1 : -1;
  const int sy = c0 < c1 ? 1 : -1;
  int err = dx - dy;

  while (true) {
    if (value(r0, c0) == OCCUPIED) {
      break;
    }
    if (r0 == r1 && c0 == c1) {
      break;
    }

    const int e2 = 2 * err;
    if (e2 > -dy) {
      err -= dy;
      r0 += sx;
    }
    if (r0 == r1 && c0 == c1) {
      break;
    }
    if (e2 < dx) {
      err += dx;
      c0 += sy;
    }
  }

  end.point.x = static_cast<double>(r0) * resolution_ + origin_x_;
  end.point.y = static_cast<double>(c0) * resolution_ + origin_y_;
}

void UTMSimulatorNode::update()
{
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!map_loaded_) {
      return;
    }
  }

  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = tf_buffer_->lookupTransform(global_frame_id_, frame_id_, tf2::TimePointZero);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "%s", ex.what());
    return;
  }

  const rclcpp::Time current_time(transform.header.stamp);
  scan_.header.stamp = current_time;

  geometry_msgs::msg::PointStamped sensor_in;
  sensor_in.header.stamp = current_time;
  sensor_in.header.frame_id = frame_id_;
  sensor_in.point.x = 0.0;
  sensor_in.point.y = 0.0;
  sensor_in.point.z = 0.0;

  geometry_msgs::msg::PointStamped sensor_out;
  tf2::doTransform(sensor_in, sensor_out, transform);

  const double l = scan_.range_max;

  std::lock_guard<std::mutex> lock(mutex_);
  #pragma omp parallel for
  for (int i = 0; i < NUM_READINGS_PER_SCAN; i++) {
    const double a = scan_.angle_min + i * scan_.angle_increment;
    geometry_msgs::msg::PointStamped stamped_in;
    stamped_in.header.stamp = current_time;
    stamped_in.header.frame_id = frame_id_;
    stamped_in.point.x = l * std::cos(a);
    stamped_in.point.y = l * std::sin(a);
    stamped_in.point.z = 0.0;

    geometry_msgs::msg::PointStamped stamped_out;
    tf2::doTransform(stamped_in, stamped_out, transform);

    raycast(sensor_out, stamped_out);
    scan_.ranges[i] = static_cast<float>(distance(stamped_out, sensor_out));
  }

  pub_->publish(scan_);
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<UTMSimulatorNode>());
  rclcpp::shutdown();
  return 0;
}
