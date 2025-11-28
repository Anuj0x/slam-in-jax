# Simultaneous Localization and Mapping (SLAM)
<a target="_blank" href="https://colab.research.google.com/github/hhosseinian/Simultaneous-Localization-and-Mapping-SLAM">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Overview
This repository contains the code and documentation for my project on Simultaneous Localization and Mapping (SLAM). SLAM is a fundamental technique in robotics that allows a robot to map its environment and determine its own location within that environment in real-time.

## Project Description
In this project, I have implemented SLAM for a 2-dimensional world using Python and various libraries. The project includes the following key components:
- Sensor and motion data simulation for a virtual robot.
- Real-time tracking of the robot's location.
- Landmark detection and mapping, including buildings, trees, rocks, and more.
- Integration of the Kalman Filter for enhanced accuracy and robustness.

## Technologies Used
- Python
- PyTorch
- CUDA (optional)
- Kalman Filter

## Project Structure

The project directory is organized as follows:

1. **Robot Moving and Sensing.ipynb**: This notebook involves localizing a robot in a 2D grid world, forming the basis for simultaneous localization and mapping (SLAM). The robot gathers sensor data and movement information to reconstruct a map of its environment. Due to inherent uncertainties in robot motion and sensors, the project addresses how to handle these inaccuracies.

    Key components include defining a robot class with functionalities for movement and sensing landmarks within a specified range, incorporating noise factors. The notebook guides us through initializing the robot, simulating movements, creating landmarks, and implementing a sense function to measure distances to visible landmarks.

2. **Omega and Xi, Constraints.ipynb**: This notebook implements Graph SLAM using a matrix (omega) and a vector (xi) to represent robot poses and landmarks. Each observation updates these structures, creating numerical relationships between poses and landmarks.

    The project demonstrates solving linear systems of equations to determine pose values using linear algebra, specifically the product of the inverse of omega and xi. It also covers creating motion constraints and how these constraints fill the omega and xi matrices, affecting relationships between poses and landmarks.

    The final goal is to extend this to a 2D case, incorporating both x and y positional values for a comprehensive SLAM solution.

3. **Landmark Detection and Tracking.ipynb**: In this notebook, you will implement Simultaneous Localization and Mapping (SLAM) for a robot navigating a 2D grid world. The goal of SLAM is to both localize the robot and map its environment using real-time sensor data. You will define a function `slam` that calculates the robot's trajectory and the positions of landmarks in the environment, updating the constraints matrix and vector based on motion and measurement noise. The implementation involves initializing constraints, updating them iteratively with sensor measurements and motion data, and finally computing the best estimate of the robot's path and landmark locations using matrix operations. The project includes visualizations to validate the SLAM algorithm's performance against the true positions of the robot and landmarks.
4. **helpers.py**: This function displays the environment of a robot within a square grid of a specified size. It optionally includes a list of landmark positions to visualize within the grid.
5. **robot_class.py**: This Python script defines a robot class that simulates a robot operating in a 2D x-y space. The robot initially points in a random direction and moves in a straight line until it approaches a wall, at which point it stops. It senses the x- and y-distances to landmarks, simplifying the implementation of SLAM (Simultaneous Localization and Mapping) by avoiding complex range and bearing calculations.

## Getting Started
1. Clone the repository to your local machine.
2. Navigate to the `src/` directory.
3. Run the main SLAM script to start the simulation.

## Usage (Under Revision)
Provide instructions on how to run the project and any dependencies or libraries needed.
-  How they can implement SLAM in their project (Under Revision)
-  Describe the core functions and a few examples of modifications (Under Revision)
-  How they can enhance it to a 3D SLAM. (Under Revision)

## Results

### Expected Outcomes

Upon running the SLAM simulation, you should observe:

- **Map Visualization:**  
  A visual representation of the robot's environment, including landmarks detected and mapped by the robot.

- **Robot Trajectory:**  
  A plot showing the estimated path of the robot as it moves through the environment, compared against its true path if available.

- **Accuracy Metrics:**  
  Metrics evaluating the accuracy of the SLAM algorithm, such as the error between estimated and actual landmark positions, will be displayed.

## Troubleshooting
Please keep me posted if you faced any problem while running the code. I will keep the troubleshooting posted in WiKi pages. 

## Acknowledgments
This project is part of the Udacity Computer Vision Nanodegree program.
Feel free to explore the code and documentation for more details about the project. If you have any questions or feedback, please don't hesitate to reach out.
