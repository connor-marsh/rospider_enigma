# MIGRATED Humble -> Jazzy / Gazebo Harmonic
# Same changes as worlds.launch.py. See migration report.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchService
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def launch_setup(context, *args, **kwargs):
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
    world_name = LaunchConfiguration('world_name').perform(context)
    nav = LaunchConfiguration('nav').perform(context)

    robot_gazebo_path = get_package_share_directory('robot_gazebo')
    world = os.path.join(robot_gazebo_path, 'worlds', 'robocup_home.sdf')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'),
                         'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r {world}'}.items(),
    )

    ros_gz_bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_gazebo_path, 'launch', 'ros_ign_bridge.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'nav': nav,
        }.items(),
    )

    spawn_model_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_gazebo_path, 'launch', 'spawn_model.launch.py')),
        launch_arguments={
            'world': world_name,
            'use_sim_time': use_sim_time,
        }.items(),
    )

    spawn_objects_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_gazebo_path, 'launch', 'spawn_objects.launch.py')),
        launch_arguments={
            'world': world_name,
            'use_sim_time': use_sim_time,
        }.items(),
    )

    return [gz_sim, spawn_objects_launch, spawn_model_launch, ros_gz_bridge_launch]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('world_name', default_value='robocup_home'),
        DeclareLaunchArgument('nav', default_value='false'),
        OpaqueFunction(function=launch_setup),
    ])


if __name__ == '__main__':
    ld = generate_launch_description()
    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
