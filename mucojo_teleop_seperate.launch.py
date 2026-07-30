import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess

from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    haptic_device_node = Node(
        package='haptic_device',
        executable='haptic_device_node',
        name='haptic_device_node',
        output='screen'
    )

    model_path = os.path.expanduser('/home/ao/mujoco_menagerie/franka_emika_panda/my_scene_insert.xml') #my_scene_insert
    
    return LaunchDescription([
        haptic_device_node,
        ExecuteProcess(
            cmd=['ros2', 'run', 'mujoco_sim_node', 'mujoco_node', model_path],
            output='screen'
        ),
        ExecuteProcess(
            cmd=['ros2', 'run', 'controller_node', 'controller_node', model_path],
            output='screen'
        )
    ])

