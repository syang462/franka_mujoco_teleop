import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess

from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    model_path = os.path.expanduser('/home/ao/mujoco_menagerie/franka_emika_panda/my_scene_insert.xml') #my_scene_insert or my_scene
    
    return LaunchDescription([
        ExecuteProcess(
            cmd=['ros2', 'run', 'mujoco_sim_node', 'mujoco_node', model_path],
            output='screen'
        ),
        ExecuteProcess(
            cmd=['ros2', 'run', 'controller_node', 'controller_node',
                 '--ros-args', '-p', 'use_topic_input:=true'],
            output='screen'
        ),
        ExecuteProcess(
            cmd=['ros2', 'run', 'gui_controller', 'gui_controller_node'],
            output='screen'
        )
    ])

