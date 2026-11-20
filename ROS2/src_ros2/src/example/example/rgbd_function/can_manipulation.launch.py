"""Start the onboard Aurora camera and can manipulation services."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='deptrum-ros-driver-aurora930',
            executable='aurora930_node',
            output='screen',
            parameters=[{
                'rgb_enable': True, 'depth_enable': True,
                'ir_enable': False, 'rgbd_enable': False,
                'point_cloud_enable': False, 'align_mode': True,
                'resolution_mode_index': 2, 'ir_fps': 12,
            }],
            remappings=[
                ('rgb/image_raw', '/depth_cam/rgb0/image_raw'),
                ('rgb/camera_info', '/depth_cam/rgb0/camera_info'),
                ('depth/image_raw', '/depth_cam/depth0/image_raw'),
            ],
        ),
        Node(
            package='example',
            executable='can_manipulation',
            output='screen',
        ),
    ])
