# Semi-Autonomous Soft Robot for Bronchoscopy Procedures

This repository contains ROS2 packages for controlling a semi-autonomous soft robot designed for bronchoscopy procedures. The robot can be controlled semi-autonomously or via teleoperation using the `joy` package for ROS2. The project includes custom ROS2 interface messages, services, and actions, and is designed to run on Windows 11 with ROS2 Iron.

## Features

- **Semi-Autonomous Control**: Implementing advanced algorithms to assist in bronchoscopy procedures.
- **Teleoperation**: Integration with the `joy` package for ROS2 allows manual control of the robot.
- **Custom ROS2 Interfaces**: Includes custom messages, services, and actions tailored for the robot's operation.

## Hardware

- **High-Resolution Microcamera**: Utilizes OpenCV to interface with a 160k resolution microcamera.
- **NI DAQ Devices**: Communication with NI DAQ devices for motor and pressure control.
- **UR5e Robot Arm**: Integration with a UR5e robot arm.
- **Electromagnetic Tracking**: Interfaces with an NDI Aurora Electromagnetic Tracker for localization.

## Dependencies

Ensure you have the following dependencies installed:

- [ROS2 Iron](https://docs.ros.org/en/iron/index.html)
- [nidaqmx](https://nidaqmx-python.readthedocs.io/en/latest/)
- [ur_rtde](https://sdurobotics.gitlab.io/ur_rtde/)
- [OpenCV](https://opencv.org/)
- [scikit-surgerynditracker](https://scikit-surgerynditracker.readthedocs.io/en/latest/)
- [joy package for ROS2](https://github.com/ros-drivers/joystick_drivers)
- [Panda3D](panda3d.org) (for 3D simulation)

## Installation

1. **ROS2 Installation**: Follow the instructions to install [ROS2 Iron](https://docs.ros.org/en/iron/Installation.html) on Windows 11.
2. **Clone the Repository into ROS2 Workspace**:
   ```powershell
   git clone https://github.com/dvanlewen/Soft-Robot-Bronchoscopy.git
3. **Install Python Dependencies**:
   ```powershell
   pip install nidaqmx ur_rtde opencv-python scikit-surgerynditracker
4. **Build the ROS2 Packages**:
   ```powershell
   colcon build

## Usage
**Running the Robot**: 
- To start the robot with teleoperative control, run:
   ```powershell
   ros2 launch soft_robot_launch.py
- To start the semi-autonomous node, run:
  ```powershell
  ros2 run soft_robot_bronchoscopy ros_localization
- For optional data collection, run:
  ```powershell
  ros2 run soft_robot_bronchoscopy ros_data_logger
Use <kbd>Ctrl</kbd>+<kbd>Left click</kbd> in the camera window to begin frame and data collection
- Toggle between the two control modes using the <kbd>start</kbd> button as referenced on an 8BitDo Pro 2 Controller
- Toggle control over perspective in the Panda3D simulation using the <kbd>option</kbd> button

## Teleoperation 8BitDo Pro 2 Controls
1. **Rotation**: <kbd>right joystick</kbd>
2. **Translation**: <kbd>D-pad</kbd>
3. **Insertion**: <kbd>triangle (top button)</kbd>
4. **Retraction**: <kbd>x (bottom button) </kbd>
5. **Steering**: <kbd>right trigger</kbd>

