from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 获取包路径
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_rospider = get_package_share_directory('rospider_description')

    # 启动Gazebo空世界
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [pkg_gazebo_ros, '/launch/gzserver.launch.py']
        ),
        launch_arguments={'world': 'empty'}.items()
    )

    # 启动GUI客户端
    gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [pkg_gazebo_ros, '/launch/gzclient.launch.py']
        )
    )

    # TF静态变换 (base_link → base_footprint)
    tf_static = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "base_link", "base_footprint"]
    )

    # 生成机器人模型
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'rospider',
            '-file', f'{pkg_rospider}/urdf/rospider_description.urdf'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        gazebo_client,
        tf_static,
        spawn_robot
    ])