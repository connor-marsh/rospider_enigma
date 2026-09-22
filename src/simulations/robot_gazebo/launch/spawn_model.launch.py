# MIGRATED Humble -> Jazzy / Gazebo Harmonic
#
# Changes from the Humble version:
#   * Controller spawners now pass --param-file. REQUIRED in Jazzy:
#     controllers set use_global_arguments=false, so parameters loaded only
#     into the controller_manager node no longer reach the controllers.
#     (ros2_control Humble->Jazzy migration notes, PR #1694)
#   * 'robot_cofig.yaml' typo fixed -> 'robot_config.yaml' (and the variable
#     is now actually used, for --param-file)
#   * use_sim_time is applied to the spawn entity node from the argument
#     instead of being hardcoded True
#   * DeclareLaunchArgument moved out of the OpaqueFunction (no-op there)
#   * sim_ign is still computed from moveit_unite, unchanged
#   * Unused imports removed

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchService
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
    moveit_unite = LaunchConfiguration('moveit_unite').perform(context)

    use_sim_time_bool = use_sim_time.lower() == 'true'
    sim_ign = 'false' if moveit_unite == 'true' else 'true'

    robot_gazebo_path = get_package_share_directory('robot_gazebo')
    xacro_file = os.path.join(robot_gazebo_path, 'urdf', 'robot.gazebo.xacro')
    controller_config_file = os.path.join(robot_gazebo_path, 'config', 'robot_config.yaml')

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': Command(['xacro ', xacro_file, ' sim_ign:=', sim_ign]),
            'use_sim_time': use_sim_time_bool,
        }],
    )

    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        output='screen',
        parameters=[{
            'source_list': ['/controller_manager/joint_states'],
            'rate': 20.0,
            'use_sim_time': use_sim_time_bool,
        }],
    )

    def spawner(controller_name):
        return Node(
            package='controller_manager',
            executable='spawner',
            arguments=[
                controller_name,
                '--param-file', controller_config_file,
            ],
            output='screen',
        )

    joint_state_broadcaster_spawner = spawner('joint_state_broadcaster')
    arm_controller_spawner = spawner('arm_controller')
    gripper_controller_spawner = spawner('gripper_controller')
    L_leg_controller_spawner = spawner('L_leg_controller')
    R_leg_controller_spawner = spawner('R_leg_controller')

    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'robot',
            '-allow_renaming', 'true',
            '-x', '0',
            '-y', '0',
        ],
        parameters=[{'use_sim_time': use_sim_time_bool}],
    )

    return [
        joint_state_publisher_node,
        robot_state_publisher_node,

        RegisterEventHandler(event_handler=OnProcessExit(
            target_action=gz_spawn_entity,
            on_exit=[joint_state_broadcaster_spawner])),
        RegisterEventHandler(event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[arm_controller_spawner])),
        RegisterEventHandler(event_handler=OnProcessExit(
            target_action=arm_controller_spawner,
            on_exit=[gripper_controller_spawner])),
        RegisterEventHandler(event_handler=OnProcessExit(
            target_action=gripper_controller_spawner,
            on_exit=[L_leg_controller_spawner])),
        RegisterEventHandler(event_handler=OnProcessExit(
            target_action=L_leg_controller_spawner,
            on_exit=[R_leg_controller_spawner])),

        gz_spawn_entity,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('world', default_value='robocup_home'),
        DeclareLaunchArgument('moveit_unite', default_value='false'),
        OpaqueFunction(function=launch_setup),
    ])


if __name__ == '__main__':
    ld = generate_launch_description()
    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()
