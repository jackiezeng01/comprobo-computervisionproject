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

The goal of this project to identify the speed at which a Neato is moving based on camera input. We did this by extracting geometric structures from images taken through a camera's motion and using this information to estimate the Neato's position at the time each image was taken, relative to where it started. Knowing the time and calculating the Neato's approximate position for each camera input allowed us to determine how fast the Neato was moving as it was taking the images.

* [Implementation](#implementation)
    * [Image Data](#image-data)
    * [Keypoint Matching](#keypoint-matching)
    * [Triangulation](#triangulation)
    * [Matrix Calculations](#matrix-calculations)
* [Challenges](#challenges)
* [Lessons Learned](#lessons-learned)
* [Next Steps](#next-steps)

## Implementation
### Image Data
[simrun]

To collect the images used to estimate motion, we used a [Raspberry Pi camera v?](https://www.raspberrypi.com/documentation/accessories/camera.html) connected to the [Neato](https://neatorobotics.com/) robot vacuum.

* camera calibration

### Keypoint Matching
[simrun]

The first step to estimating motion from image data is matching features between images. Firstly, feature points are identified in each image, with descriptors, then each feature in the feature set of the first image is compared to the feature set of the second image to get the best match.

[insert photo to illustrate this]

[talk about the **design decision** to use SURF/optical flow]

### Matrix calculations
[jackie]
* Calculating the fundamental/essential matrix from two sets of corresponding keypoints
* Calculating P2 from F

### Triangulation
[melody]

From the matrix calculations in the step above, we now have all the information we need to reconstruct the scene and verify the accuracy of our math. We can use the canonical first camera (P1) and the second camera (P2), which we calculated from the fundamental matrix, to triangulate our two sets of 2D keypoints into one set of 3D points that should reflect the environment in which the Neato was recording images. 
* Constructing mat A from P1, P2,(camera position) and u1, u2 (3d point projected onto each camera's 2d view)
* Using eigen-math to solve for A
* Results in 4 possible P2s, check which one is correct by using each one to triangulate our points -> which one results in the most accurate 3D reconstruction.

## Challenges
[Jackie]

* Lack of documentation for TurtleBot2
* Understanding the geometry math

## Lessons Learned
[Melody]

* Stepping through the triangulation math with a single point was super useful in wrapping our heads around the math
* Getting images early
* I feel like we split up the work pretty well?

## Next Steps
[Simrun]

If we had more time,
* Use TurtleBot2 because it has better camera input, easy-to-get camera intrinsics
* Further explore the math behind SURF/optical flow
* Calculate Neato's speed from camera input in real time
