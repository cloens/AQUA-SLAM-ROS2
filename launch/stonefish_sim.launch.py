import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('aqua_slam')

    settings_arg = DeclareLaunchArgument(
        'settings',
        default_value=os.path.join(pkg_share, 'data', 'stonefish_sim.yaml'),
    )
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
    )

    slam_node = Node(
        package='aqua_slam',
        executable='aqua_slam_node',
        name='stereo_dvl',
        arguments=[LaunchConfiguration('settings')],
        output='screen',
    )

    # sim_dvl_converter runs in the testbed container (stonefish_ros2 installed there).
    # It publishes /bluerov2/dvl (nav_msgs/Odometry) which this node subscribes to.

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world2odom',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
            '--frame-id', 'odom',
            '--child-frame-id', 'orb_slam',
        ],
    )

    with open(os.path.join(pkg_share, 'urdf', 'bluerov.urdf'), 'r') as f:
        robot_desc = f.read()

    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_desc}],
    )

    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(pkg_share, 'launch', 'falcon_tightly.rviz')],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    return LaunchDescription([
        settings_arg,
        use_rviz_arg,
        static_tf,
        slam_node,
        robot_state_pub,
        rviz2,
    ])
