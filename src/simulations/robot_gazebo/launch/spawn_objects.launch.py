
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    # MIGRATED: declare the args room_worlds.launch.py forwards, and switch
    # 'create' from the Gazebo Classic '-entity' flag to '-name'.
    world_arg = DeclareLaunchArgument('world', default_value='robocup_home')
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='true')

    spawn_bed = Node(
            package='ros_gz_sim',executable='create',name='spawn_entity',arguments=['-file', [os.path.join(get_package_share_directory('robot_gazebo'), 'models', 'bed.model')] , 
            '-name', 'bed',
            '-x', '5.0', '-y', '-3.9','-Y', '3.1415926']
        )
    spawn_sofa = Node(
            package='ros_gz_sim',executable='create',name='spawn_entity',arguments=['-file', [os.path.join(get_package_share_directory('robot_gazebo'), 'models', 'sofa.model')] , 
            '-name', 'sofa',
            '-x', '-1.0', '-y', '-3.9','-Y', '1.57']
        )
    spawn_tea_table = Node(
            package='ros_gz_sim',executable='create',name='spawn_entity',arguments=['-file', [os.path.join(get_package_share_directory('robot_gazebo'), 'models', 'tea_table.model')] , 
            '-name', 'tea_table',
            '-x', '-2.1', '-y', '-2.2','-Y', '1.57']
        )
    spawn_bookshelft = Node(
            package='ros_gz_sim',executable='create',name='spawn_entity',arguments=['-file', [os.path.join(get_package_share_directory('robot_gazebo'), 'models', 'bookshelft.model')] , 
            '-name', 'bookshelft',
            '-x', '2.0', '-y', '-0.55','-Y', '-1.57']
        )

    spawn_kitchen_table = Node(
            package='ros_gz_sim',executable='create',name='spawn_entity',arguments=['-file', [os.path.join(get_package_share_directory('robot_gazebo'), 'models', 'table','kitchen_table.model')] , 
            '-name', 'kitchen_table',
            '-x', '-3.5', '-y', '3.7','-Y', '1.57']
        )
    spawn_red_bottle = Node(
            package='ros_gz_sim',executable='create',name='spawn_entity',arguments=['-file', [os.path.join(get_package_share_directory('robot_gazebo'), 'models', 'bottles', 'red_bottle.model')] , 
            '-name', 'red_bottle',
            '-x', '-3.3', '-y', '3.55','-z', '0.8']
        )
    spawn_green_bottle = Node(
            package='ros_gz_sim',executable='create',name='spawn_entity',arguments=['-file', [os.path.join(get_package_share_directory('robot_gazebo'), 'models', 'bottles', 'green_bottle.model')] , 
            '-name', 'green_bottle',
            '-x', '-3.6', '-y', '3.55','-z', '0.8']
        )
    
    spawn_cupboard_0 = Node(
            package='ros_gz_sim',executable='create',name='spawn_entity',arguments=['-file', [os.path.join(get_package_share_directory('robot_gazebo'), 'models','cupboard', 'cupboard_0.model')] , 
            '-name', 'cupboard_0',
            '-x', '-2.0', '-y', '0.7' ,'-Y', '1.57']
        )
    spawn_cupboard_1 = Node(
            package='ros_gz_sim',executable='create',name='spawn_entity',arguments=['-file', [os.path.join(get_package_share_directory('robot_gazebo'), 'models','cupboard', 'cupboard_1.model')] , 
            '-name', 'cupboard_1',
            '-x', '-1.3', '-y', '3.7','-Y', '-1.57']
        )
    
    spawn_dinning_table_0 = Node(
            package='ros_gz_sim',executable='create',name='dinning_table_0',arguments=['-file', [os.path.join(get_package_share_directory('robot_gazebo'), 'models', 'table', 'dinning_table_0.model')] , 
            '-name', 'dinning_table_0',
            '-x', '1.5', '-y', '1.5','-Y', '1.57']
        )
    spawn_dinning_table_1 = Node(
            package='ros_gz_sim',executable='create',name='dinning_table_1',arguments=['-file', [os.path.join(get_package_share_directory('robot_gazebo'), 'models', 'table', 'dinning_table_1.model')] , 
            '-name', 'dinning_table_1',
            '-x', '1.5', '-y', '2.0','-Y', '1.57']
        )
    spawn_dinning_table_2 = Node(
            package='ros_gz_sim',executable='create',name='dinning_table_2',arguments=['-file', [os.path.join(get_package_share_directory('robot_gazebo'), 'models', 'table', 'dinning_table_2.model')] , 
            '-name', 'dinning_table_2',
            '-x', '2.7', '-y', '1.5','-Y', '1.57']
        )
    spawn_dinning_table_3 = Node(
            package='ros_gz_sim',executable='create',name='dinning_table_3',arguments=['-file', [os.path.join(get_package_share_directory('robot_gazebo'), 'models', 'table', 'dinning_table_3.model')] , 
            '-name', 'dinning_table_3',
            '-x', '2.7', '-y', '2.0','-Y', '1.57']
        )
    
    spawn_chair_0 = Node(
            package='ros_gz_sim',executable='create',name='spawn_entity',arguments=['-file', [os.path.join(get_package_share_directory('robot_gazebo'), 'models','chair', 'chair_0.model')] , 
            '-name', 'chair_0',
            '-x', '1.5', '-y', '1.2','-Y', '1.57']
        )
    spawn_chair_1 = Node(
            package='ros_gz_sim',executable='create',name='spawn_entity',arguments=['-file', [os.path.join(get_package_share_directory('robot_gazebo'), 'models','chair', 'chair_1.model')] , 
            '-name', 'chair_1',
            '-x', '1.5', '-y', '2.3','-Y', '-1.57']
        )
    spawn_chair_2 = Node(
            package='ros_gz_sim',executable='create',name='spawn_entity',arguments=['-file', [os.path.join(get_package_share_directory('robot_gazebo'), 'models','chair', 'chair_2.model')] , 
            '-name', 'chair_2',
            '-x', '2.7', '-y', '1.2','-Y', '1.57']
        )
    spawn_chair_3 = Node(
            package='ros_gz_sim',executable='create',name='spawn_entity',arguments=['-file', [os.path.join(get_package_share_directory('robot_gazebo'), 'models','chair', 'chair_3.model')] , 
            '-name', 'chair_3',
            '-x', '2.7', '-y', '2.3','-Y', '-1.57']
        )

    ld = LaunchDescription()

    ld.add_action(world_arg)
    ld.add_action(use_sim_time_arg)

    ld.add_action(TimerAction(period=1.0, actions=[spawn_bed,spawn_sofa,spawn_tea_table,spawn_bookshelft]))

    
    ld.add_action(TimerAction(period=2.0, actions=[spawn_kitchen_table,spawn_cupboard_0,spawn_cupboard_1]))

    
    ld.add_action(TimerAction(period=3.0, actions=[spawn_dinning_table_0,spawn_dinning_table_1,spawn_dinning_table_2,spawn_dinning_table_3]))

    
    ld.add_action(TimerAction(period=4.0, actions=[spawn_chair_0,spawn_chair_1,spawn_chair_2,spawn_chair_3]))


    ld.add_action(TimerAction(period=5.0, actions=[spawn_red_bottle,spawn_green_bottle]))

    return ld