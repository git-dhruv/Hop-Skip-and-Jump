import numpy as np
from pydrake.all import JacobianWrtVariable
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

def calculateDoubleContactJacobians(plant, context):
    Jleft = plant.CalcJacobianTranslationalVelocity(context, JacobianWrtVariable.kV, 
                                                    plant.GetFrameByName("left_lower_leg"), np.array([0,0, -0.5]), plant.world_frame(), plant.world_frame())
    Jright = plant.CalcJacobianTranslationalVelocity(context, JacobianWrtVariable.kV, 
                                                     plant.GetFrameByName("right_lower_leg"), np.array([0,0, -0.5]), plant.world_frame(), plant.world_frame())
    JdotVleft = plant.CalcBiasTranslationalAcceleration(context, JacobianWrtVariable.kV,
                                                        plant.GetFrameByName("left_lower_leg"), np.array([0,0,-0.5]),plant.world_frame(), plant.world_frame())
    JdotVright = plant.CalcBiasTranslationalAcceleration(context, JacobianWrtVariable.kV,
                                                         plant.GetFrameByName("right_lower_leg"), np.array([0,0,-0.5]),plant.world_frame(), plant.world_frame())    
    J_c = np.row_stack((Jleft, Jright))
    J_c_dot_v = np.row_stack((JdotVleft, JdotVright))
    return J_c, J_c_dot_v




def fetchStates(context, plant):    
    ## COM Calculations ##
    J = plant.CalcJacobianCenterOfMassTranslationalVelocity(context,JacobianWrtVariable.kV,plant.world_frame(),plant.world_frame())
    com_pos = plant.CalcCenterOfMassPositionInWorld(context).ravel()        
    com_vel = (J @ plant.GetVelocities(context)).ravel()

    ## Torso Angle States ##
    joint_pos_idx = plant.GetJointByName("planar_roty").position_start()
    joint_vel_idx = plant.GetJointByName("planar_roty").velocity_start()
    torso_angle = plant.GetPositions(context)[joint_pos_idx:joint_pos_idx+1].ravel()

    torso_angle = np.arctan2(np.sin(torso_angle), np.cos(torso_angle))
    torso_ang_vel = plant.GetVelocities(context)[joint_vel_idx:joint_vel_idx+1].ravel()

    ## Foot Positions ##
    left = plant.CalcPointsPositions(context, plant.GetBodyByName("left_lower_leg").body_frame(),
                                    np.array([0, 0, -0.5]), plant.world_frame()).ravel()
    right = plant.CalcPointsPositions(context, plant.GetBodyByName("right_lower_leg").body_frame(),
                                    np.array([0, 0, -0.5]), plant.world_frame()).ravel()
    
    leftVel = (calculateDoubleContactJacobians(plant, context)[0] @ plant.GetVelocities(context)).ravel()[:3]
    rightVel = (calculateDoubleContactJacobians(plant, context)[0] @ plant.GetVelocities(context)).ravel()[3:]


    statePacket = {'com_pos': com_pos, 'com_vel': com_vel, 'torso_ang': torso_angle, 'torso_ang_vel': torso_ang_vel,
                   'left_leg': left, 'right_leg': right , 'leftVel': leftVel, 'rightVel': rightVel}
    return statePacket

