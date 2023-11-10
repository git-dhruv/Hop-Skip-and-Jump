import numpy as np
from pydrake.multibody.tree import Frame
from dataclasses import dataclass

# Global variables for gait timing
LEFT_STANCE = 0
RIGHT_STANCE = 1
STANCE_DURATION = 0.2

def get_fsm(t: float) -> int:
    if np.fmod(t, 2 * STANCE_DURATION) > STANCE_DURATION:
        return RIGHT_STANCE
    else:
        return LEFT_STANCE

def time_since_switch(t: float) -> float:
    return np.fmod(t, STANCE_DURATION)

def time_until_switch(t: float) -> float:
    return STANCE_DURATION - time_since_switch(t)


@dataclass
class PointOnFrame:
    '''
        Wrapper class which holds a BodyFrame and a vector, representing a point 
        expressed in the BodyFrame
    '''
    frame: Frame
    pt: np.ndarray

@dataclass
class OscGains:
    kp_com: np.ndarray
    kd_com: np.ndarray
    w_com: np.ndarray
    kp_swing_foot: np.ndarray
    kd_swing_foot: np.ndarray
    w_swing_foot: np.ndarray
    kp_base: np.ndarray
    kd_base: np.ndarray
    w_base: np.ndarray
    w_vdot: float
