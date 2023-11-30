"""
@brief: Given a Center of Mass or torso input, we calculate a desired acceleration based on gains
"""
import numpy as np
from utils import fetchStates
from pydrake.multibody.all import JacobianWrtVariable
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.math import RigidTransform

PREFLIGHT = 0
FLIGHT = 1
LAND = 2

def optyaw(des:float,curr:float) -> float:
    """
        Black magic
        Calculates the minimum angle of rotation for yaw between the desired and current yaw angles. 
        The angle is calculated by performing a dot product of two vectors and determining the sign. 
        The vectors are calculated by applying a rotation transformation to a base vector using the `rotz` method. 
        The resulting vectors are then normalized. 
        
        Parameters:
        desired_yaw (float): The desired yaw angle in radians.
        current_yaw (float): The current yaw angle in radians.
        
        Returns:
        float: The minimum angle of rotation for yaw in radians. If the angle is very small (less than 0.001), the function returns zero.


        Warning: This code was written for my MEAM620 Yaw controller. It is adapted for this project and is not refined in terms of code quality.  
    """
    def rotz(ang):
        ang = float(ang)
        c = np.cos(ang)
        s = np.sin(ang)
        return np.array([[c ,-s],[s, c]])
    def wrapper(y) :
        return np.arctan2(np.sin(y), np.cos(y))

    basevec = np.array([1,0]).reshape(-1,1)
    currvec = rotz(curr)@basevec
    desvec = rotz(des)@basevec

    currvec = currvec/np.linalg.norm(currvec)
    desvec = desvec/np.linalg.norm(desvec)

    shortestAngle = wrapper(np.arccos(currvec.T.dot(desvec)))
    if np.abs(shortestAngle)<0.001:
        return np.array([0])
    cr = np.cross(currvec.T,desvec.T)
    sign = np.sign(cr)

    return (sign*shortestAngle).flatten()

class valueFetcher:
    def __init__(self, trajtype):
        self.type = trajtype #1 means polytraj
    
    def getVal(self,traj, t):
        if self.type == 1:
            return traj.value(t).ravel() #, traj.derivative(2).value(t).ravel()
        return traj.value().ravel()

class pid:
    def __init__(self, Kp, Kd, saturations=1e3):
        self.Kp = Kp; self.Kd = Kd
        self.saturations = saturations

    def calcOut(self, y, ydes, ydot, yddotdes, angular = 0):
        #Don't use ydot if value too high
        if np.linalg.norm(ydot)> self.saturations:
            ydot = self.saturations*ydot/np.linalg.norm(ydot)

        e = ydes - y
        if angular:
            e = optyaw(ydes, y)
            out = self.Kp@e - self.Kd@ydot
        else:
            out = yddotdes + self.Kp@e - self.Kd@(ydot)

        return np.clip(out, -self.saturations, self.saturations)

class tracking_objective:
    def __init__(self, plant, context, COMParams=None, TorsoParams=None, FootParams=None, polyTraj = 0):
        if COMParams is not None:
            self.COMTracker = fetchCOMParams(COMParams, plant, context, polyTraj)
        
        if TorsoParams is not None:
            self.TorsoTracker = fetchTorsoParams(TorsoParams, plant, context, polyTraj)

        if FootParams is not None:
            self.FootTracker = fetchFootParams(FootParams, plant, context, polyTraj)
        

        

    def Update(self, t, y_des, objective, footNum=0, finiteState=0):
        if 'COM' in objective:
            return self.COMTracker.getAcc(y_des, t, finiteState), self.COMTracker.CalcJ(), self.COMTracker.CalcJdotV() 
        if 'torso' in objective:
            return self.TorsoTracker.getAcc(y_des, t) , self.TorsoTracker.CalcJ(), self.TorsoTracker.CalcJdotV()
        if 'foot' in objective:
            return self.FootTracker.getAcc(y_des, t, footNum), self.FootTracker.CalcJ(), self.FootTracker.CalcJdotV()         
        raise Exception("What the fuck have you provided in objective?")

class fetchCOMParams:
    def __init__(self, params, plant, context, polyTraj):
        self.pid = pid(params['Kp'], params['Kd'], params['saturations'])
        self.plant = plant; self.context = context

        self.getVal = valueFetcher(polyTraj)

        self.desiredPos = np.zeros((params['Kp'].shape[0],)); self.desiredVel = np.zeros((params['Kp'].shape[0],))

    def getAcc(self, ydes, t, finiteState):
        if finiteState == LAND:
            yd = self.getVal.getVal(ydes, t)
        if finiteState == PREFLIGHT:
            # We dont care about the polytraj in Preflight because it will always be preflight
            yd = self.getVal.getVal(ydes, t)
            yd = self.convertStateToCOM(yd)
            ## Yes these are state accelerations but we approximate base frame as COM
            acc = ydes.derivative(2).value(t).ravel()
            yd_ddotdes = np.array([acc[0], 0, acc[1]]) 

            

        
        ## @WARN this is not added to the PID yet and is merely an accessor ##
        # if yddot_des is None:
        #     yddot_des = yd*0

        y = self.CalcY()
        ydot = self.CalcYdot()    


        self.desiredPos = yd; self.desiredVel = yd*0
        return self.pid.calcOut(y, yd, ydot, yd_ddotdes)

    def CalcY(self) -> np.ndarray:
        return self.plant.CalcCenterOfMassPositionInWorld(self.context).ravel()

    def CalcYdot(self) -> np.ndarray:
        return (self.CalcJ() @ self.plant.GetVelocities(self.context)).ravel()

    def CalcJ(self) -> np.ndarray:
        return self.plant.CalcJacobianCenterOfMassTranslationalVelocity(self.context, JacobianWrtVariable.kV, self.plant.world_frame(), self.plant.world_frame())

    def CalcJdotV(self) -> np.ndarray:
        return self.plant.CalcBiasCenterOfMassTranslationalAcceleration(self.context, JacobianWrtVariable.kV, self.plant.world_frame(), self.plant.world_frame()).ravel()
    
    def convertStateToCOM(self, state):
        ## Replace this with deep copy of plant ##
        from copy import deepcopy
        plant = deepcopy(self.plant)
        # plant = MultibodyPlant(0.0)
        # parser = Parser(plant)
        # parser.AddModels("../models/planar_walker.urdf")
        # plant.WeldFrames(
        #     plant.world_frame(),
        #     plant.GetBodyByName("base").body_frame(),
        #     RigidTransform.Identity()
        # )
        # plant.Finalize()
        context = plant.CreateDefaultContext()        
        #Get the internal robot to go to current state
        plant.SetPositionsAndVelocities(context, state)
        com_pos = plant.CalcCenterOfMassPositionInWorld(context).ravel()        
        return com_pos




class fetchTorsoParams:
    def __init__(self, params, plant, context, polytraj):
        self.pid = pid(params['Kp'], params['Kd'], params['saturations'])
        self.plant = plant; self.context = context
        self.joint_pos_idx = self.plant.GetJointByName("planar_roty").position_start()
        self.joint_vel_idx = self.plant.GetJointByName("planar_roty").velocity_start()
        self.getVal = valueFetcher(polytraj)

        self.desiredPos = np.zeros((params['Kp'].shape[0],)); self.desiredVel = np.zeros((params['Kp'].shape[0],))
 
    def getAcc(self, ydes, t):
        y = self.CalcY()
        ydot = self.CalcYdot()        
        yd = self.getVal.getVal(ydes, t)

        self.desiredPos = yd; self.desiredVel = yd*0
        return self.pid.calcOut(y, yd, ydot, y*0 ,angular=1)

    def CalcY(self) -> np.ndarray:
        return self.plant.GetPositions(self.context)[self.joint_pos_idx:self.joint_pos_idx+1].ravel()

    def CalcYdot(self) -> np.ndarray:
        return self.plant.GetVelocities(self.context)[self.joint_vel_idx:self.joint_vel_idx+1].ravel()

    def CalcJ(self) -> np.ndarray:
        J = np.array([0,0,1,0,0,0,0]).reshape(1,-1)
        return J

    def CalcJdotV(self) -> np.ndarray:
        return 0

class fetchFootParams:
    def __init__(self, params, plant, context, polyTraj):
        self.pid = pid(params['Kp'], params['Kd'], params['saturations'])
        self.plant = plant; self.context = context

        self.getVal = valueFetcher(polyTraj)

        self.desiredPos = np.zeros((params['Kp'].shape[0],)); self.desiredVel = np.zeros((params['Kp'].shape[0],))

    def getAcc(self, ydes, t, fsm):
        self.fsm = fsm
        y = self.CalcY()
        ydot = self.CalcYdot()    
        yd = self.getVal.getVal(ydes, t)    
        com = fetchStates(self.context, self.plant )['com_pos']
        if fsm:
            target = np.array([com[0]+0.3, 0 , np.clip(com[-1]-0.6,0, np.inf)])
        else:
            target = np.array([com[0]-0.3, 0 , np.clip(com[-1]-0.6,0, np.inf)])

        self.desiredPos = target; self.desiredVel = target*0
        return self.pid.calcOut(y, target, ydot, y*0)


    def CalcY(self) -> np.ndarray:
        if self.fsm:
            return self.plant.CalcPointsPositions(self.context, self.plant.GetBodyByName("left_lower_leg").body_frame(),
                                        np.array([0, 0, -0.5]), self.plant.world_frame()).ravel()
        return self.plant.CalcPointsPositions(self.context, self.plant.GetBodyByName("right_lower_leg").body_frame(),
                                        np.array([0, 0, -0.5]), self.plant.world_frame()).ravel()


    def CalcJ(self) -> np.ndarray:
        if self.fsm:
            return self.plant.CalcJacobianTranslationalVelocity(
                self.context, JacobianWrtVariable.kV, self.plant.GetBodyByName("left_lower_leg").body_frame(),
                np.array([0,0,-0.5]), self.plant.world_frame(), self.plant.world_frame())
        else:
            return self.plant.CalcJacobianTranslationalVelocity(
                self.context, JacobianWrtVariable.kV, self.plant.GetBodyByName("right_lower_leg").body_frame(),
                np.array([0,0,-0.5]), self.plant.world_frame(), self.plant.world_frame())


    def CalcYdot(self) -> np.ndarray:
        return (self.CalcJ() @ self.plant.GetVelocities(self.context)).ravel()

    def CalcJdotV(self) -> np.ndarray:
        if self.fsm:
            return self.plant.CalcBiasTranslationalAcceleration(
                self.context, JacobianWrtVariable.kV, self.plant.GetBodyByName("left_lower_leg").body_frame(),
                np.array([0,0,-0.5]), self.plant.world_frame(), self.plant.world_frame()).ravel()
        else:
            return self.plant.CalcBiasTranslationalAcceleration(
                self.context, JacobianWrtVariable.kV, self.plant.GetBodyByName("right_lower_leg").body_frame(),
                np.array([0,0,-0.5]), self.plant.world_frame(), self.plant.world_frame()).ravel()