# MIGRATED Humble -> Jazzy / Gazebo Harmonic
#
# Changes from the Humble version:
#   * ignition.msgs.* -> gz.msgs.*                         (Harmonic type strings)
#   * /clock bridge uncommented                            (REQUIRED for use_sim_time)
#   * use_sim_time is now actually applied as a node parameter
#   * DeclareLaunchArgument moved out of the OpaqueFunction (it was a no-op there)
#   * nav_arg is now declared AND returned (it was dropped before)
#   * static_transform_publisher switched to named arguments

from launch import LaunchDescription, LaunchService
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
    nav = LaunchConfiguration('nav').perform(context)

    use_sim_time_bool = use_sim_time.lower() == 'true'

    remappings_default = [('/odom/tf', 'tf')]
    if nav == 'true':
        remappings_default += [('/controller/cmd_vel', '/cmd_vel')]

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Velocity command (ROS2 -> GZ)
            '/controller/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            # Odometry (GZ -> ROS2)
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            # TF (GZ -> ROS2)
            '/odom/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            # Clock (GZ -> ROS2) -- required when use_sim_time:=true.
            # gz_sim.launch.py does NOT bridge this for you.
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            # Joint states (GZ -> ROS2)
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            # Lidar (GZ -> ROS2)
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            # IMU (GZ -> ROS2)
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
            # Camera (GZ -> ROS2)
            '/depth_cam/depth_cam@sensor_msgs/msg/Image[gz.msgs.Image',
            '/depth_cam/rgb/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        parameters=[{'use_sim_time': use_sim_time_bool}],
        remappings=remappings_default,
        output='screen',
    )

    # NOTE: this static map->odom transform will fight AMCL if you ever run
    # nav:=true, because AMCL publishes map->odom itself. See migration report.
    map_static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time_bool}],
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.0',
            '--roll', '0.0', '--pitch', '0.0', '--yaw', '0.0',
            '--frame-id', 'map', '--child-frame-id', 'odom',
        ],
    )

    return [bridge, map_static_tf]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('nav', default_value='false'),
        OpaqueFunction(function=launch_setup),
    ])


if __name__ == '__main__':
    ld = generate_launch_description()
    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
