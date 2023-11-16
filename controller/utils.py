import numpy as np
from pydrake.all import JacobianWrtVariable

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


    statePacket = {'com_pos': com_pos, 'com_vel': com_vel, 'torso_ang': torso_angle, 'torso_ang_vel': torso_ang_vel,
                   'left_leg': left, 'right_leg': right }
    return statePacket
