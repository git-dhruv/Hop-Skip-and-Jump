"""
@author: Dhruv Parikh, Anirudh Kailaje
@file: osc.py
@brief: Based on FSM, tracks Foot traj, torso angle and COM
"""

"""
Lower level instantaneous QP
Tracks Desired COM with both feet on ground
"""

import numpy as np
from osc_objective import tracking_objective

from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import Context, LeafSystem, BasicVector
from pydrake.trajectories import Trajectory, PiecewisePolynomial
from pydrake.solvers import MathematicalProgram, OsqpSolver
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from utils import calculateDoubleContactJacobians, fetchStates

class OSC(LeafSystem):
    def __init__(self, urdf, polyTraj=0):
        LeafSystem.__init__(self)

        ### Internal Plant Model for simulation ###
        self.plant = MultibodyPlant(0.0)
        self.parser = Parser(self.plant)
        self.parser.AddModels(urdf)
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetBodyByName("base").body_frame(),
            RigidTransform.Identity()
        )
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()


        ## ___________Parameters for tracking___________ ##
        self.whatToTrack = ['COM', 'torso', 'foot']
        COMParams = {'Kp': np.diag([60, 0, 60]), 'Kd': np.diag([100, 0, 100])/60 , 'saturations': 20} #Max Lim: 1 G
        TorsoParams = {'Kp': np.diag([5]), 'Kd': np.diag([2]) , 'saturations': 15*np.pi/180} #Max Lim: 5 deg/s
        footParams = {'Kp': 700*np.eye(3,3), 'Kd': 30*np.eye(3,3) , 'saturations': 50000} #Max Lim: 10 m/s2
        ## Cost Weights ##
        self.WCOM = np.eye(3,3)
        self.WTorso = np.diag([0.01]) #Maybe a consistant way to set the weights - 5*np.pi/180*(self.WCOM.max()/10)
        self.wFoot = np.array([[2,0,0],[0,2,0],[0,0,2]])
        self.Costs = {'COM': self.WCOM, 'torso' : self.WTorso, 'foot': self.wFoot} 
        ##_______________________________________________##

        ## ______________Solver Parameters______________ ##
        self.max_iter = 30000
        ##_______________________________________________##


        # !WARNING : IT IS ASSUMED THAT OSC AND OSC_TRACKING SHARE THE SAME PLANT!
        self.tracking_objective_air = tracking_objective(self.plant, self.plant_context, None, None, footParams, polyTraj=polyTraj)
        self.tracking_objective_land = tracking_objective(self.plant, self.plant_context, COMParams, TorsoParams, None, polyTraj=polyTraj)


        ## ______________Declaring Ports______________ ##        
        self.robot_state_input_port_index = self.DeclareVectorInputPort("x", self.plant.num_positions() + self.plant.num_velocities()).get_index()
        if polyTraj:
            intype = [PiecewisePolynomial(),PiecewisePolynomial(), PiecewisePolynomial()]
        else:
            intype = [BasicVector(3),BasicVector(1), BasicVector(6)]
        self.traj_input_ports = {
            self.whatToTrack[0]: self.DeclareAbstractInputPort("com_traj", AbstractValue.Make(intype[0])).get_index(),
            self.whatToTrack[1]: self.DeclareAbstractInputPort("base_joint_traj", AbstractValue.Make(intype[1])).get_index(),
            self.whatToTrack[2]: self.DeclareAbstractInputPort("foot_traj", AbstractValue.Make(intype[2])).get_index()}
        
        self.torque_output_port = self.DeclareVectorOutputPort("u", self.plant.num_actuators(), self.CalcTorques)
        self.u = np.zeros((self.plant.num_actuators()))

        self.logging_port = self.DeclareVectorOutputPort("logs", BasicVector(24), self.logCB)
        ##_______________________________________________##

        self.idx = None; self.inAir = 0; 


    def fetchTrackParams(self):
        return {'COM_pos_d': self.tracking_objective_land.COMTracker.desiredPos, 'COM_vel_d':self.tracking_objective_land.COMTracker.desiredVel,
               'Torso_pos_d': self.tracking_objective_land.TorsoTracker.desiredPos, 'Torso_vel_d':self.tracking_objective_land.TorsoTracker.desiredVel,
               'Foot_pos_d': np.array([0]), 'Foot_vel_d': np.array([0])}
                
    def logParse(self, x):
        #Parses the log data and returns in human readable form - test in ipynb
        i = self.log_idx
        COM_POS = x[:i[1],:]
        COM_VEL = x[i[1]:i[2],:]
        T_POS = x[i[2]:i[3],:]
        T_VEL = x[i[3]:i[4],:]
        left = x[i[4]:i[5],:]
        right = x[i[5]:i[6],:]
        COM_POS_DESIRED = x[i[6]:i[7],:]
        COM_VEL_DESIRED = x[i[7]:i[8],:]
        Torso_POS_DESIRED = x[i[8]:i[9],:]
        Torso_VEL_DESIRED = x[i[9]:i[10],:]
        Ft_POS_DESIRED = x[i[10]:i[11],:]
        Ft_POS_DESIRED = x[i[11]:i[12],:]
        return COM_POS, COM_VEL, T_POS, T_VEL, left, right, COM_POS_DESIRED, COM_VEL_DESIRED, Torso_POS_DESIRED, Torso_VEL_DESIRED, Ft_POS_DESIRED, Ft_POS_DESIRED
        
    def logCB(self, context, output):
        # COM, Torso Angle, Foot pos and velocities
        statePacket = fetchStates(self.plant_context, self.plant)

        # Desired COM, Torso, Desired Foot angles
        trackPacket = self.fetchTrackParams()
        

        # Desired acceleration for the QP - not calculating now and assuming sats take care of it
        
        #Don't ask questions move on with your life
        idx = [0]; outVector = None
        for _, value in statePacket.items():
            if len(idx)==1:
                outVector = value.flatten()
                idx.append(outVector.shape[0])
            else:
                outVector = np.concatenate((outVector, value.flatten()))
                idx.append(outVector.shape[0])
        for _, value in trackPacket.items():
            outVector = np.concatenate((outVector, value.flatten()))
            idx.append(outVector.shape[0])
        
        self.log_idx = idx
        self.acc = 0
        output.SetFromVector(outVector)        

        

    def solveQP(self, context):
        ## Get the context from the diagram ##
        x = self.EvalVectorInput(context, self.robot_state_input_port_index).get_value()
        t = context.get_time()

        ## Update the internal context ##
        self.plant.SetPositionsAndVelocities(self.plant_context, x)

        stancefoot = fetchStates(self.plant_context, self.plant)
        if stancefoot['left_leg'][-1]>=1e-2 and stancefoot['right_leg'][-1]>=1e-2:
            self.inAir = 1
            self.previnAir = t
            
        else:
            self.inAir = 0




        ## Create the Mathematical Program ##
        qp = MathematicalProgram()
        u = qp.NewContinuousVariables(self.plant.num_actuators(), "u")
        vdot = qp.NewContinuousVariables(self.plant.num_velocities(), "vdot")
        lambda_c = qp.NewContinuousVariables(6, "lambda_c")

        ## Quadratic Costs ##
        for track in self.whatToTrack:
            #Get desired trajectory and costs
            traj = self.EvalAbstractInput(context, self.traj_input_ports[track]).get_value()
            cost = self.Costs[track]

            if self.inAir and 'foot' in track:
                for fsm in [0, 1]:
                    # Get what to track and system states
                    yddot_cmd_i, J_i, JdotV_i = self.tracking_objective_air.Update(t, traj, track, fsm)
                    yii = JdotV_i + J_i@vdot
                    qp.AddQuadraticCost( (yddot_cmd_i - yii).T@cost@(yddot_cmd_i - yii) )

            elif ('foot' not in track) and (self.inAir==0):
                #Get what to track and system states
                yddot_cmd_i, J_i, JdotV_i = self.tracking_objective_land.Update(t, traj, track)    
                yii = JdotV_i + J_i@vdot
                qp.AddQuadraticCost( (yddot_cmd_i - yii).T@cost@(yddot_cmd_i - yii) )

        # qp.AddQuadraticCost( 1e-7*u.T@u )
        # qp.AddQuadraticCost(0.01*(self.u-u).T@(self.u-u) )
        # Calculate terms in the manipulator equation
        M = self.plant.CalcMassMatrix(self.plant_context)
        Cv = self.plant.CalcBiasTerm(self.plant_context)    
        # Drake considers gravity to be an external force (on the right side of the dynamics equations), 
        # so we negate it to match the homework PDF and slides
        G = -self.plant.CalcGravityGeneralizedForces(self.plant_context)
        B = self.plant.MakeActuationMatrix()

        #Calculate Contact Jacobians
        if self.inAir == 0:
            J_c, J_c_dot_v = calculateDoubleContactJacobians(self.plant, self.plant_context)
            #Dynamics
            qp.AddLinearEqualityConstraint(M@vdot + Cv + G - B@u - J_c.T@lambda_c, np.zeros((7,)))
            #Contact
            qp.AddLinearEqualityConstraint(J_c_dot_v + (J_c@vdot).reshape(-1,1), np.zeros((6,1)))
            # Friction Cone Constraint
            mu = 1
            A_fric = np.array([[1, 0, -mu, 0, 0, 0], # Friction constraint for left foot, positive x-direction
                            [-1, 0, -mu, 0, 0, 0],   # Friction constraint for left foot, negative x-direction
                            [0, 0, 0, 1, 0, -mu],    # Friction constraint for right foot, positive x-direction
                            [0, 0, 0, -1, 0, -mu]])  # Friction constraint for right foot, negative x-direction

            # The constraint is applied to the 6x1 lambda_c vector
            qp.AddLinearConstraint(A_fric @ lambda_c.reshape(6, 1), -np.inf * np.ones((4, 1)), np.zeros((4, 1)))
            qp.AddLinearEqualityConstraint(lambda_c[1] == 0)
            qp.AddLinearEqualityConstraint(lambda_c[4] == 0)
        else:
            qp.AddLinearEqualityConstraint(M@vdot + Cv + G - B@u, np.zeros((7,)))
        for i in range(len(u)):
            qp.AddLinearConstraint(u[i], np.array([-1400]), np.array([1400]))

        # Solve the QP
        solver = OsqpSolver()
        qp.SetSolverOption(solver.id(), "max_iter", self.max_iter)
        qp.SetSolverOption(solver.id(), "eps_abs", 1e-5)
        qp.SetInitialGuess(u, self.u)

        result = solver.Solve(qp)

        # If we exceed iteration limits use the previous solution
        if not result.is_success():            
            print("Solver not working, pal!!!  ", t)
            usol = self.u
        else:
            usol = result.GetSolution(u)
            self.acc = result.GetSolution(vdot)
            self.u = usol        
        if  np.linalg.norm(usol)>1400:
            usol = 1400*usol/np.linalg.norm(usol)
        import time
        # time.sleep(0.05)
        return usol

    ## Ignore ##
    def CalcTorques(self, context: Context, output: BasicVector) -> None:
        usol = self.solveQP(context)
        output.SetFromVector(usol)
    def get_traj_input_port(self, traj_name):
        return self.get_input_port(self.traj_input_ports[traj_name])    
    def get_state_input_port(self):
        return self.get_input_port(self.robot_state_input_port_index)