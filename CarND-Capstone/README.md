# **Capstone Project** 

**Capstone Project**

The goals / steps of this project are the following:
* Build a self driving car running on Carla vehicle

### Student Contact
* Full name: Xuan Duc Cu
* Project Start Date: 06/24/2020
* Project End Date: 07/10/2020
* Email: cxduc92@gmail.com
* Student display name: Duc C.

### Project change log
* Ver 0.0 - 07/10/2020  : Initial submission

## Introduction

Carla the Udacity self-driving car has 4 major subsystems:

  - Sensor subsystem, which constists of the hardware components that gather data about the environment. This subsystem includes lidar, radar and even GPS sensors mouted on the car.
  - Perception subsystem, consists of software to process sensor data.
  - Planning subsystem, uses the output from perception for behavior planning and for both sorth and long term path plan.
  - Control subsystem, which ensures that the vehicle follows the path provided by the planning subsystem and sends control commands to the vehicle.

### The Perception Subsystem

The perception subsystem processes data from sensor into structured information that can eventually be used for path planning or control. This is where most of the vehicles analysis of the environment takes place. We can further divide the perception subsystem ifself into two underlying subsystems:
  - detection
  - localization
The localization subsystem is responsible for using sensor and map data to determine the vehicle’s precise location. 
The detection subsystem is resonsible for understanding the surrounding environment. This subsystem includes software components such as:
  - lane detection
  - traffic sign and traffic light detection classifier
  - object detection and tracking
  - free space detection

The perception subsystem passes the data from localization and detection to the planning subsystem. The planning subsystem determines what maneuver the vehicle should undertake next.

### The Planning Subsystem

Once data from the sensors has been processed by the perception subsystem, the vehicle can use that information to plan its path. There are several components of the planning system:
  - route planning, responsible for high-level decisions about the path of the vehicle between two points on a map
  - prediction, estimates what actions or maneuver other objects might take in the futur
  - behavior planning, determines what behavior the vehicle should exhibit at any point in time
  - trajectory generation,  based on the desired immediate behavior it will determin which trajectory is best for executing this behavior

Once the vehicle has a planned trajectory the next step is to execute that trajectory. This is the responsibility of the control subsystem.

### The Control Subsystem

The last subsystem in the vehicle is the control subsystem. This subsystem contains software components to ensure that the vehicle follows the path specified by the planning subsystem. The control subsystem may include components such as PID controllers, model predictive controllers or other controlles.

## Implementation Details

### Traffic Light Image Classification

The ROS traffic light detector is implemented in node `tl_detector` in classes `TLDetector` and `TLClassifier`. `TLDetector` is responsible for finding a nearest traffic light position and calls `TLClassifier.get_classification` with the current camera image. `TLClassifier` uses the SSD MobileNet model to classify the traffic light color (red, yellow, green, none). If at least 2 consecutive images were classified as red then `TLDetector` publishes the traffic light waypoint index in the `/traffic_waypoint` topic.

### Waypoint Updater

As the vehicle moves along a path, the waypoint updater is responsible for making changes to the planned path. Waypoint updater publishes the next 75 waypoints ahead of the car position, with the velocity that the car needs to have at that point. Each 1/20 seconds, it does:

  - Update of closest waypoint. It does a local search from current waypoint until it finds a local minimum in the distance. 
  - Update of velocity. If there is a red ligth ahead, it updates waypoint velocities so that the car stops `~stop_distance` (*node parameter, default: 5.0 m*) meters behind the red light waypoint. Waypoint velocities before the stop point are updated considering a smooth deceleration.

Traversed waypoints are dequeued and new points enqueued, preserving and reusing those in the middle. When a light-state changes, the entire queue is updated as already discribed. 

### Drive By Wire

The drive-by-wire node adjusts steering, throttle and brakes according to the velocity targets published by the waypoint follower (which is informed by the waypoint updater node). If the list of waypoints contains a series of descending velocity targets, the PID velocity controller (in the twist controller component of DBW) will attempt to match the target velocity.

#### Steering
Steering is handled by a predictive steering which is implemented in the `YawController`class. 

#### Throttle

Throttle is controlled by a linear PID by passing in the velocity cross-track-error (difference between the current velocity and the proposed velocity).

#### Brake

If a negative value is returned by the throttle PID, it means that the car needs to decelerate by braking. The braking torque is calculated by the formula `(vehicle_mass + fuel_capacity * GAS_DENSITY) * wheel_radius * deceleration`.

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

