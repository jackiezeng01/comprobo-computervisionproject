# Estimating Motion from Computer Vision

*ENGR3590: A Computational Introduction to Robotics, Olin College of Engineering, FA2022*

*Computer Vision Project*

*Simrun Mutha, Jackie Zeng and Melody Chiu*

write-up reqs:
* What was the goal of your project? Since everyone is doing a different project, you will have to spend some time setting this context.
* How did you solve the problem (i.e., what methods / algorithms did you use and how do they work)? As above, since not everyone will be familiar with the algorithms you have chosen, you will need to spend some time explaining what you did and how everything works.
* Describe a design decision you had to make when working on your project and what you ultimately did (and why)? These design decisions could be particular choices for how you implemented some part of an algorithm or perhaps a decision regarding which of two external packages to use in your project.
* What if any challenges did you face along the way?
* What would you do to improve your project if you had more time?
* Did you learn any interesting lessons for future robotic programming projects? These could relate to working on robotics projects in teams, working on more open-ended (and longer term) problems, or any other relevant topic.

## Introduction

blah blah

* [Implementation](#implementation)
    * [Image Data](#image-data)
    * [Keypoint Matching](#keypoint-matching)
    * [Triangulation](#triangulation)
* [Challenges](#challenges)
* [Lessons Learned](#lessons-learned)
* [Next Steps](#next-steps)

## Implementation
### Image Data

talk about
* hardware (Neato, raspi cam version 2.?)
* camera calibration
* camera intrinsics

### Keypoint Matching

[[source]](keypoint_matching.py)

talk about
* SURF/optical flow (Design decision)

### Triangulation

[[source]](triangulation.py)

* Constructing mat A from P1, P2,(camera position) and u1, u2 (3d point projected onto each camera's 2d view)
* Using eigen-math to solve for A

## Challenges


## Lessons Learned


## Next Steps

