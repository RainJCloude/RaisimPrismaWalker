from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
import xacro
from launch.conditions import IfCondition, UnlessCondition


 
def generate_launch_description():
    declared_arguments = []
        
    arm_dynamixel_path = os.path.join(
        get_package_share_directory('arm_dynamixel'))

    declared_arguments.append(
        DeclareLaunchArgument(
            "rviz_config_file", #this will be the name of the argument  
            default_value=PathJoinSubstitution(
                [FindPackageShare("arm_dynamixel"), "config", "rviz", "standing.rviz"]
            ),
            description="RViz config file (absolute path) to use when launching rviz.",
        )
    )
 
   
    urdf_arm = os.path.join(arm_dynamixel_path, "urdf", "arm.urdf.xacro")
    rviz_config = os.path.join(arm_dynamixel_path, "config", "rviz", "standing.rviz")
    controller_config = os.path.join(arm_dynamixel_path, "config", "pos_controller.yaml")

    robot_description_arm = {"robot_description": Command(['xacro ', urdf_arm])}  
 
 
    joint_state_publisher_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
    )
    
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description_arm,
                    {"use_sim_time": True},
            ],
        remappings=[('/robot_description', '/robot_description')]
    )
 

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config],
    )



    control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[controller_config],
        output='both',
        remappings=[
            ("~/robot_description", "/robot_description"),
        ],
    ) 

    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    ) #among the controller of controller_manager you are saying to launchg the joint_state_broadcaster

    joint_trajectory_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "--controller-manager", "/controller_manager"],  
    ) 

    joint_pos_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["position_controller", "--controller-manager", "/controller_manager"],  
    ) 


    #Launch the ros2 controllers after the model spawns in Gazebo 
    delay_joint_traj_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster,
            on_exit=[joint_pos_controller],
        )
    )

    delay_joint_state_broadcaster = (
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=control_node,
                on_exit=[joint_state_broadcaster],
            )
        )
    )
    
    bridge_camera = Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        arguments=[
            '/VideoXacroCamera@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            '--ros-args', 
            '-r', '/VideoXacroCamera:=/videocamera',
        ],
        output='screen'
    )
    static_tf_camera = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="screen",
        arguments=["0", "0", "0", "0", "0", "0", "1", "camera_link", "arm/dyn4_r/camera"]
    )

    nodes_to_start = [
        joint_state_publisher_node,
        robot_state_publisher_node,  
        control_node,
        bridge_camera,
        delay_joint_traj_controller, 
        delay_joint_state_broadcaster,
        rviz_node,
        static_tf_camera,
    ]

    return LaunchDescription(declared_arguments + nodes_to_start) 
