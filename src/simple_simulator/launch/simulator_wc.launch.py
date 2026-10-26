"""Simulated PMV: differential-drive odometry plus a simulated 2D laser.

Launched once per vehicle, inside that vehicle's namespace. odo_simulator
integrates from the initial pose, so the odom frame origin coincides with the
map origin and map -> odom is a static identity transform. That is what the
ROS 1 fake_localization node provided; it has no ROS 2 counterpart and none is
needed here.

TF frame names drop the leading slash the ROS 1 launch files used
("/v1/odom" -> "v1/odom"): tf2 rejects frame ids that start with a separator.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    tf_prefix = LaunchConfiguration('tf_prefix')
    use_localization = LaunchConfiguration('use_localization')
    delta_x = LaunchConfiguration('delta_x')
    delta_y = LaunchConfiguration('delta_y')
    delta_yaw = LaunchConfiguration('delta_yaw')
    scan_frame = LaunchConfiguration('scan_frame')

    odom_frame = PythonExpression(["'", tf_prefix, "/odom'"])
    base_frame = PythonExpression(["'", tf_prefix, "/base_link'"])

    return LaunchDescription([
        DeclareLaunchArgument('tf_prefix', default_value=''),
        DeclareLaunchArgument('use_localization', default_value='true'),
        DeclareLaunchArgument('delta_x', default_value='0.0'),
        DeclareLaunchArgument('delta_y', default_value='0.0'),
        DeclareLaunchArgument('delta_yaw', default_value='0.0'),
        DeclareLaunchArgument('scan_frame', default_value=''),

        Node(
            package='simple_simulator',
            executable='odo_simulator',
            name='odo_simple_simulator',
            output='screen',
            parameters=[{
                'odom_frame_id': odom_frame,
                'base_frame_id': base_frame,
                'initialPoseX': delta_x,
                'initialPoseY': delta_y,
                'initialPoseTheta': delta_yaw,
            }],
            remappings=[
                ('~/base_pose_ground_truth', 'odom'),
                ('~/cmd_vel', 'cmd_vel'),
            ],
        ),

        # Stands in for fake_localization. odo_simulator already integrates from
        # the initial pose, so the ground-truth map pose is odom -> base_link and
        # this transform is identity.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom',
            condition=IfCondition(use_localization),
            arguments=['0', '0', '0', '0', '0', '0', 'map', odom_frame],
        ),

        Node(
            package='simple_simulator',
            executable='utm_simulator',
            name='simulator_hokuyo',
            output='screen',
            parameters=[{
                'frame_id': scan_frame,
                # One map server serves both vehicles, so this is the absolute
                # name; unqualified it would resolve to <ns>/map and never
                # arrive, leaving the simulated laser silent.
                'map_topic': '/map',
            }],
            remappings=[('~/scan', 'scan_orig0')],
        ),
    ])
