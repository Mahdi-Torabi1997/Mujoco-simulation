import mujoco as mj
from mujoco.glfw import glfw
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
import numpy as np
import os

# Path to the MuJoCo XML model file
xml_path = 'biped.xml'
simulation_duration = 30  # Simulation duration in seconds
current_step = 0

# FSM states for legs and knees
FSM_LEG1_SWING = 0
FSM_LEG2_SWING = 1
FSM_KNEE1_STANCE = 0
FSM_KNEE1_RETRACT = 1
FSM_KNEE2_STANCE = 0
FSM_KNEE2_RETRACT = 1

# Initialize FSM state variables
fsm_hip = FSM_LEG2_SWING
fsm_knee1 = FSM_KNEE1_STANCE
fsm_knee2 = FSM_KNEE2_STANCE

# Mouse interaction variables
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# ---------------------------------------------------------
# Custom functions for controlling the robot and simulation
# ---------------------------------------------------------

def controller(model, data):
    """
    Implements the finite state machine for controlling the robot's legs and knees
    based on the state of the robot. The legs and knees go through different phases:
    stance, swing, and retract.
    """
    global fsm_hip, fsm_knee1, fsm_knee2

    # State Estimation
    quat_leg1 = data.xquat[1, :]
    quat_leg2 = data.xquat[3, :]
    euler_leg1 = quat2euler(quat_leg1)
    euler_leg2 = quat2euler(quat_leg2)
    pos_foot1 = data.xpos[2, :]
    pos_foot2 = data.xpos[4, :]

    abs_leg1 = -euler_leg1[1]
    abs_leg2 = -euler_leg2[1]

    # Finite State Machine logic for leg swing control
    if fsm_hip == FSM_LEG2_SWING and pos_foot2[2] < 0.05 and abs_leg1 < 0.0:
        fsm_hip = FSM_LEG1_SWING
    elif fsm_hip == FSM_LEG1_SWING and pos_foot1[2] < 0.05 and abs_leg2 < 0.0:
        fsm_hip = FSM_LEG2_SWING

    # FSM logic for knee retraction control
    if fsm_knee1 == FSM_KNEE1_STANCE and pos_foot2[2] < 0.05 and abs_leg1 < 0.0:
        fsm_knee1 = FSM_KNEE1_RETRACT
    elif fsm_knee1 == FSM_KNEE1_RETRACT and abs_leg1 > 0.1:
        fsm_knee1 = FSM_KNEE1_STANCE

    if fsm_knee2 == FSM_KNEE2_STANCE and pos_foot1[2] < 0.05 and abs_leg2 < 0.0:
        fsm_knee2 = FSM_KNEE2_RETRACT
    elif fsm_knee2 == FSM_KNEE2_RETRACT and abs_leg2 > 0.1:
        fsm_knee2 = FSM_KNEE2_STANCE

    # Control logic: apply torques based on FSM states
    data.ctrl[0] = -0.5 if fsm_hip == FSM_LEG1_SWING else 0.5
    data.ctrl[2] = 0.0 if fsm_knee1 == FSM_KNEE1_STANCE else -0.25
    data.ctrl[4] = 0.0 if fsm_knee2 == FSM_KNEE2_STANCE else -0.25

def init_controller(model, data):
    """
    Initializes the controller by setting initial joint positions and control inputs.
    """
    data.qpos[4] = 0.5
    data.ctrl[0] = data.qpos[4]

def quat2euler(quat):
    """
    Converts quaternion to Euler angles (roll, pitch, yaw).
    """
    _quat = np.concatenate([quat[1:], quat[:1]])  # MuJoCo uses [w, x, y, z]
    r = R.from_quat(_quat)
    return r.as_euler('xyz', degrees=False)

# ---------------------------------------------------------
# Utility functions for mouse and keyboard interaction
# ---------------------------------------------------------

def keyboard(window, key, scancode, action, mods):
    """
    Resets the simulation if the BACKSPACE key is pressed.
    """
    if action == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, action, mods):
    """
    Tracks mouse button states (left, middle, right).
    """
    global button_left, button_middle, button_right
    button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    """
    Moves the camera based on mouse movement.
    """
    global lastx, lasty
    dx, dy = xpos - lastx, ypos - lasty
    lastx, lasty = xpos, ypos

    if not (button_left or button_middle or button_right):
        return

    width, height = glfw.get_window_size(window)
    mod_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS

    action = mj.mjtMouse.mjMOUSE_ZOOM if button_middle else (
        mj.mjtMouse.mjMOUSE_MOVE_H if button_right and mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V)

    mj.mjv_moveCamera(model, action, dx/height, dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    """
    Zooms the camera in and out using scroll.
    """
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)

# ---------------------------------------------------------
# Simulation Initialization and Execution
# ---------------------------------------------------------

# Set the path for the MuJoCo XML model
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname, xml_path)
xml_path = abspath

# Load MuJoCo model and create simulation context
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
cam = mj.MjvCamera()
opt = mj.MjvOption()

# Set up window using GLFW
glfw.init()
window = glfw.create_window(1200, 900, "Biped Robot Simulation", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# Initialize visualization tools
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# Set mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Configure the camera settings
cam.azimuth = 120.89
cam.elevation = -15.81
cam.distance = 8.0
cam.lookat = np.array([0.0, 0.0, 2.0])

# Set custom gravity for ramp simulation
model.opt.gravity[0] = 9.81 * np.sin(0.1)
model.opt.gravity[2] = -9.81 * np.cos(0.1)

# Initialize controller
init_controller(model, data)

# Main simulation loop
while not glfw.window_should_close(window):
    simstart = data.time
    while data.time - simstart < 1.0/60.0:  # Simulate at 60 FPS
        mj.mj_step(model, data)  # Perform simulation step
        controller(model, data)  # Apply control

    if data.time >= simulation_duration:
        break

    # Render the simulation
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)  # Swap OpenGL buffers
    glfw.poll_events()  # Process pending GUI events

# Clean up
glfw.terminate()
