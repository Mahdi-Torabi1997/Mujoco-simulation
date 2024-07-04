! (https://github.com/Mahdi-Torabi1997/Mujoco-simulation/blob/main/mujoco.png?raw=true)

## Bipedal Robot Simulation
This project showcases a bipedal robot simulation using MuJoCo. The robot walks using a simple controller and finite state machines (FSMs) to manage its gait.
## Biped Design
## Joints and Actuators
Hip Joint (Pin Joint): Allows the upper leg to swing.
Knee Joint (Slide Joint): Allows the foot to retract to avoid ground contact.
Actuators:
Position Servo: Controls the hip and knee joint positions.
Velocity Servo: Controls the joint velocities.
## Simulation Overview
## Initialization
The simulation loads the biped model from biped.xml and sets initial positions using init_controller.
## State Estimation
State estimation converts quaternions to Euler angles and calculates foot positions for the walking cycle.
## Controller
The controller manages the walking gait:
Hip: Alternates leg swings.
Knee: Manages leg extension and retraction.

