"""
@brief: Given a Center of Mass or torso input, we calculate a desired acceleration based on gains
"""
import numpy as np



from pydrake.multibody.all import JacobianWrtVariable

class valueFetcher:
    def __init__(self, trajtype):
        self.type = trajtype #1 means polytraj
    
    def getVal(self,traj, t):
        if self.type == 1:
            return traj.value(t).ravel()
        return traj.value().ravel()

class pid:
    def __init__(self, Kp, Kd, saturations=1e3):
        self.Kp = Kp; self.Kd = Kd
        self.saturations = saturations

    def calcOut(self, y, ydes, ydot):
        #Don't use ydot if value too high
        if np.linalg.norm(ydot)> self.saturations:
            print("Velocity Term too high!", ydot)
            ydot = self.saturations*ydot/np.linalg.norm(ydot)

        out = self.Kp@(ydes-y) - self.Kd@ydot

        return np.clip(out, -self.saturations, self.saturations)

class tracking_objective:
    def __init__(self, plant, context, COMParams=None, TorsoParams=None, FootParams=None, polyTraj = 0):
        if COMParams is not None:
            self.COMTracker = fetchCOMParams(COMParams, plant, context, polyTraj)
        
        if TorsoParams is not None:
            self.TorsoTracker = fetchTorsoParams(TorsoParams, plant, context, polyTraj)

        if FootParams is not None:
            self.FootTracker = fetchFootParams(FootParams, plant, context, polyTraj)
        

        

    def Update(self, t, y_des, objective):
        if 'COM' in objective:
            return self.COMTracker.getAcc(y_des, t), self.COMTracker.CalcJ(), self.COMTracker.CalcJdotV() 
        if 'torso' in objective:
            return self.TorsoTracker.getAcc(y_des, t) , self.TorsoTracker.CalcJ(), self.TorsoTracker.CalcJdotV()
        if 'foot' in objective:
            raise NotImplementedError
            return self.FootTracker.getAcc(y_des, t)         
        raise Exception("What the fuck have you provided in objective?")

class fetchCOMParams:
    def __init__(self, params, plant, context, polyTraj):
        self.pid = pid(params['Kp'], params['Kd'], params['saturations'])
        self.plant = plant; self.context = context

        self.getVal = valueFetcher(polyTraj)

    def getAcc(self, ydes, t):
        y = self.CalcY()
        ydot = self.CalcYdot()    
        yd = self.getVal.getVal(ydes, t)    
        return self.pid.calcOut(y, yd, ydot)

    def CalcY(self) -> np.ndarray:
        return self.plant.CalcCenterOfMassPositionInWorld(self.context).ravel()

    def CalcYdot(self) -> np.ndarray:
        return (self.CalcJ() @ self.plant.GetVelocities(self.context)).ravel()

    def CalcJ(self) -> np.ndarray:
        return self.plant.CalcJacobianCenterOfMassTranslationalVelocity(self.context, JacobianWrtVariable.kV, self.plant.world_frame(), self.plant.world_frame())

    def CalcJdotV(self) -> np.ndarray:
        return self.plant.CalcBiasCenterOfMassTranslationalAcceleration(self.context, JacobianWrtVariable.kV, self.plant.world_frame(), self.plant.world_frame()).ravel()



class fetchTorsoParams:
    def __init__(self, params, plant, context, polytraj):
        self.pid = pid(params['Kp'], params['Kd'], params['saturations'])
        self.plant = plant; self.context = context
        self.joint_pos_idx = self.plant.GetJointByName("planar_roty").position_start()
        self.joint_vel_idx = self.plant.GetJointByName("planar_roty").velocity_start()
        self.getVal = valueFetcher(polytraj)
 
    def getAcc(self, ydes, t):
        y = self.CalcY()
        ydot = self.CalcYdot()        
        yd = self.getVal.getVal(ydes, t)
        return self.pid.calcOut(y, yd, ydot)

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
    def __init__(self, params, plant, context):
        self.pid = pid(params['Kp'], params['Kd'], params['saturations'])
        self.plant = plant; self.context = context

    def getAcc(self, ydes, t):
        pass