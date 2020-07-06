# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program.

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

# Goal

Using `autonomous` mode within the simulator gives us access to the throttle, steering and cross-track error.  The primary goal of the project is to steer the car around the track using the cross-track error and a PID controller.  The PID controller is a common, simple, effective and widely accepted technique for actuating autonomous vehicles. The steering will be determined by the lateral distance from the center of the track using the PID controller.  Since we do not know the dynamics of the car, we will need to tune the controller. Upon doing so the car should be able to stay on the road adequately. The idea, of course, is that a car should be able to be controlled
based on a path, and in this case the path is simply the center of the road.  It is critical to be able to control the car, as in real-life situations with a computed path the car needs to behave as expected. Secondarily, speed regulation is an objective of the project. This is simply applying a second PID controller to control the speed in the context of a non-zero control value.

# Overview

This projects implements a PID controller in C++for keeping the car on the lake race track by appropriately adjusting the steering angle to maneuver the vehicle around the track! The simulator provides you with the cross track error (CTE) and the velocity (mph) in order to compute the appropriate steering angle.
One more thing. Try to drive SAFELY as fast as possible! Get as near to 100mph by controlling the throttle. 

# PID Controller

A PID controller continuously calculates the error between a set point (SP) and desired variable (DV) to apply a correction based on proportional, derivative and integral errors. The correction is applied after multiplying the respective errors with some factors called gains. To achieve optimal system performance, however, these gains usually need to be tuned. Since the inception of PID control in 1911 by Elmer Sperry, the tuning process has remained elusive although some techniques for tuning have emerged over the last couple of decades. These include the popular Ziegler–Nichols method, Coordinate Ascent algorithm (TWIDDLE) or manual tuning. It can be shown that manual tuning of a PID controller without any system knowledge is a lost cause since a random process in three dimensions has negligible chances of converging to starting position. 

The tuning strategy adopted for this project was the one by Ziegler–Nichols. Although, it is difficult to get exact ultimate gain and period of oscillation required for Ziegler–Nichols tuning for the test track, a reasonable estimate was made to get starting gains. These gains were further tuned manually as per the tuning table given below:


| Response | Rise Time    | Overshoot | Settling Time | Steady-State Error | 
| -------- | ------------ | --------- | ------------- | ------------------ | 
| Kp       | Decrease     | Increase  | Minor Change  | Decrease           |
| Ki       | Decrease     | Increase  | Increase      | Eliminate          |
| Kd       | Minor Change | Decrease  | Decrease      | No Influence       |

The starting and final gains are:

| Method   |   Kp   |   Ki   |   Kd   |
| -------- | ------ | ------ | ------ |
| Starting | 0.2	| 0.002	 | 13.33  |
| Final    | 0.2	| 0.002	 |	7	  |

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1(mac, linux), 3.81(Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.

Fellow students have put together a guide to Windows set-up for the project [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/files/Kidnapped_Vehicle_Windows_Setup.pdf) if the environment you have set up for the Sensor Fusion projects does not work for this project. There's also an experimental patch for windows in this [PR](https://github.com/udacity/CarND-PID-Control-Project/pull/3).

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`. 

Tips for setting up your environment can be found [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/23d376c7-0195-4276-bdf0-e02f1f3c665d)

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

[Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!

More information is only accessible by people who are already enrolled in Term 2
of CarND. If you are enrolled, see [the project page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/f1820894-8322-4bb3-81aa-b26b3c6dcbaf/lessons/e8235395-22dd-4b87-88e0-d108c5e5bbf4/concepts/6a4d8d42-6a04-4aa6-b284-1697c0fd6562)
for instructions and the project rubric.

