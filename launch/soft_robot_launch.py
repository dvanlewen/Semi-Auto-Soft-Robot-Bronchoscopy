from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='soft_robot_bronchoscopy',
            executable='ros_aurora_pub',
            name='sensor_tracker'
        ),
        Node(
            package='soft_robot_bronchoscopy',
            executable='camera_tracker',
            name='sensor_cam'
        ),
        Node(
            package='soft_robot_bronchoscopy',
            executable='ros_nidaq_interface',
            name='nidaq'
        ),
        Node(
            package='soft_robot_bronchoscopy',
            executable='ros_ur5e_controller',
            name='arm_control'
        ),
        Node(
            package='soft_robot_bronchoscopy',
            executable='ros_steering',
            name='steer_control'
        ),
        Node(
            package='soft_robot_bronchoscopy',
            executable='ros_ur5e_sub',
            name='arm_sub'
        ),
        Node(
            package='soft_robot_bronchoscopy',
            executable='panda_sub',
            name='plotter'
        ),
        Node(
            package='joy',
            executable='joy_node',
            name='controller'
        ),
        Node(
            package='soft_robot_bronchoscopy',
            executable='joy_listener',
            name='controller_translate'
        )
        #Node(
        #    package='soft_robot_bronchoscopy',
        #    executable='ros_localization',
        #    name='logic'
        #)
    ])
