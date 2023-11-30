import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

from pydrake.multibody.tree import Frame
from pydrake.trajectories import Trajectory
from pydrake.systems.framework import Context
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.all import JacobianWrtVariable




@dataclass
class PointOnFrame:
    '''
        Wrapper class which holds a BodyFrame and a vector, representing a point 
        expressed in the BodyFrame
    '''
    frame: Frame
    pt: np.ndarray


class OperationalSpaceTrackingObjective(ABC):
    '''
        Abstract class representing a general operational space tracking objective (Do Not Modify).
        Specific task spaces should implement
        - GetY
        - GetYdot
        - GetJ
        - GetJdotV
        With the assumption that the context will already be set to the correct state
    '''
    def __init__(self,
                 plant: MultibodyPlant,
                 plant_context: Context,
                 finite_states_to_track: List[int],
                 kp: np.ndarray,
                 kd: np.ndarray):

        self.kp = kp
        self.kd = kd
        self.fsm_states_to_track = finite_states_to_track
        self.plant = plant
        self.context = plant_context

        self.J = None
        self.JdotV = None
        self.yddot_cmd = None
        self.fsm = None

    def Update(self, t: float, y_des_traj: Trajectory, fsm: int):
        self.fsm = fsm
        y = self.CalcY()
        ydot = self.CalcYdot()
        
        self.J = self.CalcJ()
        self.JdotV = self.CalcJdotV()

        yd = y_des_traj.value(t).ravel()
        yd_ddot = 0
        if yd.shape[0] > 1:
            yd = self.fetchStates(yd)
            acc = y_des_traj.derivative(2).value(t).ravel()
            yd_ddot = np.array([acc[0], 0, acc[1]]) #Scheme
            # yd[0] = y[0]; yd[2] = 0.8
            # print(y - yd)

        self.yddot_cmd = yd_ddot - self.kp @ (y - yd) - self.kd @ (ydot)
        self.yddot_cmd = np.clip(self.yddot_cmd, -30, 30)
        

    def GetJ(self):
        return self.J

    def GetJdotV(self):
        return self.JdotV

    def GetYddotCmd(self):
        return self.yddot_cmd
    
    def fetchStates(self, state):
        
        plant = deepcopy(self.plant)    
        context = plant.CreateDefaultContext()
        
        #Get the internal robot to go to current state
        plant.SetPositionsAndVelocities(context, state)
        com_pos = plant.CalcCenterOfMassPositionInWorld(context).ravel()        
        return com_pos


    @abstractmethod
    def CalcJ(self) -> np.ndarray:
        pass

    @abstractmethod
    def CalcJdotV(self) -> np.ndarray:
        pass

    @abstractmethod
    def CalcY(self) -> np.ndarray:
        pass

    @abstractmethod
    def CalcYdot(self) -> np.ndarray:
        pass


class PointPositionTrackingObjective(OperationalSpaceTrackingObjective):
    '''
        Track the position of a point as measured in the world frame
    '''
    def __init__(self,
                 plant: MultibodyPlant,
                 plant_context: Context,
                 finite_states_to_track: List[int],
                 kp: np.ndarray,
                 kd: np.ndarray,
                 pts_to_track: Dict[int,PointOnFrame]):

        super().__init__(plant, plant_context, finite_states_to_track, kp, kd)
        self.pts_to_track = pts_to_track

    def CalcY(self) -> np.ndarray:
        pt_to_track = self.pts_to_track[self.fsm]
        #Position of points in pt_to_track to frame plant
        return self.plant.CalcPointsPositions(self.context, pt_to_track.frame,
                                              pt_to_track.pt, self.plant.world_frame()).ravel()

    def CalcJ(self) -> np.ndarray:
        pt_to_track = self.pts_to_track[self.fsm]
        return self.plant.CalcJacobianTranslationalVelocity(
            self.context, JacobianWrtVariable.kV, pt_to_track.frame,
            pt_to_track.pt, self.plant.world_frame(), self.plant.world_frame()
        )

    def CalcYdot(self) -> np.ndarray:
        return (self.CalcJ() @ self.plant.GetVelocities(self.context)).ravel()

    def CalcJdotV(self) -> np.ndarray:
        pt_to_track = self.pts_to_track[self.fsm]
        return self.plant.CalcBiasTranslationalAcceleration(
            self.context, JacobianWrtVariable.kV, pt_to_track.frame,
            pt_to_track.pt, self.plant.world_frame(), self.plant.world_frame()
        ).ravel()


class CenterOfMassPositionTrackingObjective(OperationalSpaceTrackingObjective):
    '''
        Track the center of mass of a robot
    '''
    def __init__(self, plant: MultibodyPlant,
                 plant_context: Context,
                 finite_states_to_track: List[int],
                 kp: np.ndarray,
                 kd: np.ndarray):
        super().__init__(plant, plant_context, finite_states_to_track, kp, kd)

    def CalcY(self) -> np.ndarray:
        return self.plant.CalcCenterOfMassPositionInWorld(self.context).ravel()

    def CalcYdot(self) -> np.ndarray:
        return (self.CalcJ() @ self.plant.GetVelocities(self.context)).ravel()

    def CalcJ(self) -> np.ndarray:
        return self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
            self.context, JacobianWrtVariable.kV, self.plant.world_frame(), self.plant.world_frame()
        )

    def CalcJdotV(self) -> np.ndarray:
        return self.plant.CalcBiasCenterOfMassTranslationalAcceleration(
            self.context, JacobianWrtVariable.kV, self.plant.world_frame(), self.plant.world_frame()
        ).ravel()


class JointAngleTrackingObjective(OperationalSpaceTrackingObjective):
    '''
        Tracking objective for representing a desired joint angle
    '''
    def __init__(self, plant: MultibodyPlant,
                 plant_context: Context,
                 finite_states_to_track: List[int],
                 kp: np.ndarray,
                 kd: np.ndarray,
                 joint_name: str):
        super().__init__(plant, plant_context, finite_states_to_track, kp, kd)
        
        self.joint_pos_idx = self.plant.GetJointByName(joint_name).position_start()
        self.joint_vel_idx = self.plant.GetJointByName(joint_name).velocity_start()

    def CalcY(self) -> np.ndarray:
        return self.plant.GetPositions(self.context)[self.joint_pos_idx:self.joint_pos_idx+1].ravel()

    def CalcYdot(self) -> np.ndarray:
        return self.plant.GetVelocities(self.context)[self.joint_vel_idx:self.joint_vel_idx+1].ravel()

    def CalcJ(self) -> np.ndarray:
        J = np.array([0,0,1,0,0,0,0]).reshape(1,-1)
        return J

    def CalcJdotV(self) -> np.ndarray:
        return 0

