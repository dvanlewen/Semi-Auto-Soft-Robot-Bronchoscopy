from setuptools import find_packages, setup

package_name = 'soft_robot_bronchoscopy'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('lib/' + package_name, [package_name + '/AuroraRegistration.py']),
        ('lib/' + package_name, [package_name + '/rtde_rotation.py']),
        ('lib/' + package_name, [package_name + '/closedloop_steering.py']),
        ('lib/' + package_name, [package_name + '/newPaths.mat']),
        ('lib/' + package_name, [package_name + '/Lung.egg']),
        ('lib/' + package_name, [package_name + '/catheter.egg']),
        ('lib/' + package_name, [package_name + '/process_images_tracking.py']),
        ('lib/' + package_name, [package_name + '/Robot.egg']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Daniel',
    maintainer_email='danielvl@bu.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "ros_aurora_pub = soft_robot_bronchoscopy.ros_aurora_pub:main",
            "ros_camera = soft_robot_bronchoscopy.ros_camera:main",
            "ros_ur5e_sub = soft_robot_bronchoscopy.ros_ur5e_sub:main",
            "ros_steering = soft_robot_bronchoscopy.ros_steering:main",
            "ros_localization = soft_robot_bronchoscopy.ros_localization:main",
            "ros_nidaq_interface = soft_robot_bronchoscopy.ros_nidaq_interface:main",
            "ros_ur5e_controller = soft_robot_bronchoscopy.ros_ur5e_controller:main",
            "ros_data_logger = soft_robot_bronchoscopy.ros_data_logger:main",
            "joy_listener = soft_robot_bronchoscopy.joy_listener:main",
            "panda_sub = soft_robot_bronchoscopy.ros_panda_subscriber:main",
            "camera_tracker = soft_robot_bronchoscopy.ros_camera_tracker:main",
            "panda_sub_test = soft_robot_bronchoscopy.panda_subscriber:main"
        ],
    },
)
