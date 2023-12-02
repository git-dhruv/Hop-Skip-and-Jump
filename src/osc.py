"""
@author: Dhruv Parikh, Anirudh Kailaje
@file: osc.py
@brief: Based on FSM, tracks Foot traj, torso angle and COM
"""

"""
Lower level instantaneous QP
Tracks Desired COM with both feet on ground
"""

import numpy as np, logging
from osc_objective import tracking_objective

from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import Context, LeafSystem, BasicVector
from pydrake.trajectories import Trajectory, PiecewisePolynomial
from pydrake.solvers import MathematicalProgram, OsqpSolver
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from utils import calculateDoubleContactJacobians, fetchStates

### FSM MODES ###
PREFLIGHT = 0
FLIGHT = 1
LAND = 2

module_logger = logging.getLogger("OSC")
class OSC(LeafSystem):
    def __init__(self, urdf, utraj = None):
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

        self.dircolUTraj = utraj
        


        ## ___________Parameters for tracking___________ ##
        self.whatToTrack = [['COM', 'torso'],['foot'],['COM', 'torso']]
        
        COMParams = {'Kp': np.diag([60, 0, 60]), 'Kd': 0.0*np.diag([100, 0, 100])/25 , 'saturations': 50} #Max Lim: 1 G
        COMParams_land = {'Kp': np.diag([600, 0, 600])/2, 'Kd': np.diag([100, 0, 100]) , 'saturations': 50} #Max Lim: 1 G
        TorsoParams = {'Kp': np.diag([0]), 'Kd': np.diag([2]) , 'saturations': 50} #Max Lim: 5 deg/s2
        TorsoParams_land = {'Kp': np.diag([5.85]), 'Kd': np.diag([2.85*1.75]) , 'saturations': 50} #Max Lim: 5 deg/s2
        footParams = {'Kp': 1500*np.eye(3,3), 'Kd': 30*np.eye(3,3) , 'saturations': 5e5} #Max Lim: 10 m/s2
        
        ## Cost Weights ##
        self.WCOM = np.eye(3,3)
        self.WTorso = np.diag([40]) 
        self.wFoot = np.array([[2,0,0],[0,2,0],[0,0,2]])
        self.Costs = {'COM': self.WCOM, 'torso' : self.WTorso, 'foot': self.wFoot} 
        ##_______________________________________________##

        ## ______________Solver Parameters______________ ##
        self.max_iter = 10000
        ##_______________________________________________##


        # !WARNING : IT IS ASSUMED THAT OSC AND OSC_TRACKING SHARE THE SAME PLANT!
        self.tracking_objective_preflight = tracking_objective(self.plant, self.plant_context, COMParams, TorsoParams, None, polyTraj=1)
        self.tracking_objective_air = tracking_objective(self.plant, self.plant_context, None, TorsoParams, footParams, polyTraj=0)
        self.tracking_objective_land = tracking_objective(self.plant, self.plant_context, COMParams_land, TorsoParams_land, None, polyTraj=0)


        ## ______________Declaring Ports______________ ##        
        self.robot_state_input_port_index = self.DeclareVectorInputPort("x", self.plant.num_positions() + self.plant.num_velocities()).get_index()
        self.dircolInput = self.DeclareAbstractInputPort("preflight",AbstractValue.Make(PiecewisePolynomial())).get_index()
        self.flightInput = self.DeclareAbstractInputPort("flight", AbstractValue.Make(BasicVector(6))).get_index()
        self.landInput = self.DeclareAbstractInputPort("landing", AbstractValue.Make(BasicVector(3))).get_index()
        self.phaseInput = self.DeclareAbstractInputPort("phase", AbstractValue.Make(BasicVector(1))).get_index()        
        self.torque_output_port = self.DeclareVectorOutputPort("u", self.plant.num_actuators(), self.CalcTorques)
        self.logging_port = self.DeclareVectorOutputPort("logs", BasicVector(46), self.logCB)

        self.traj_input_ports = [self.dircolInput, self.flightInput, self.landInput]

        ##_______________________________________________##

        self.idx = None; self.inAir = 0;
        self.fsm = PREFLIGHT 

        self.usol = np.zeros((self.plant.num_actuators()))
        self.contactForces = np.zeros((6,))
        self.solutionCost = 0
        module_logger.debug("Created OSC")
        module_logger.debug(f"OSC COM Params: {COMParams}")
        module_logger.debug(f"OSC COMland Params: {COMParams}")
        module_logger.debug(f"OSC Torso Params: {COMParams}")
        module_logger.debug(f"OSC Torsoland Params: {COMParams}")
        module_logger.debug(f"OSC Foot Params: {COMParams}")

    def fetchTrackParams(self):
        ##@TODO: Take FSM as a Class parameter and record the particular objective. Also record the FSM 
        ## Based on the FSM tracking objectives are Decided
        INACTIVE = 0
        if self.fsm == PREFLIGHT:
            COM_pos_d = self.tracking_objective_preflight.COMTracker.desiredPos
            COM_vel_d = self.tracking_objective_preflight.COMTracker.desiredVel
            Torso_pos_d = self.tracking_objective_preflight.TorsoTracker.desiredPos
            Torso_vel_d = self.tracking_objective_preflight.TorsoTracker.desiredVel
            LFoot_pos_d = self.tracking_objective_air.FootTracker.desiredPos[:3]*INACTIVE
            RFoot_pos_d = self.tracking_objective_air.FootTracker.desiredPos[3:]*INACTIVE
        elif self.fsm == FLIGHT:
            COM_pos_d = self.tracking_objective_preflight.COMTracker.desiredPos*INACTIVE
            COM_vel_d = self.tracking_objective_preflight.COMTracker.desiredVel*INACTIVE
            Torso_pos_d = self.tracking_objective_preflight.TorsoTracker.desiredPos*INACTIVE
            Torso_vel_d = self.tracking_objective_preflight.TorsoTracker.desiredVel*INACTIVE
            LFoot_pos_d = self.tracking_objective_air.FootTracker.desiredPos[:3]
            RFoot_pos_d = self.tracking_objective_air.FootTracker.desiredPos[3:]
        elif self.fsm == LAND:
            COM_pos_d = self.tracking_objective_land.COMTracker.desiredPos
            COM_vel_d = self.tracking_objective_land.COMTracker.desiredVel
            Torso_pos_d = self.tracking_objective_land.TorsoTracker.desiredPos
            Torso_vel_d = self.tracking_objective_land.TorsoTracker.desiredVel
            LFoot_pos_d = self.tracking_objective_air.FootTracker.desiredPos[:3]*INACTIVE
            RFoot_pos_d = self.tracking_objective_air.FootTracker.desiredPos[3:]*INACTIVE
        else:
            raise Exception(f"Incorrect FSM Mode Supplied, FSM = {self.fsm}!")
        
        return {'COM_pos_d': COM_pos_d, 'COM_vel_d':COM_vel_d,
               'Torso_pos_d': Torso_pos_d, 'Torso_vel_d':Torso_vel_d,
               'LFoot_pos_d': LFoot_pos_d, 'RFoot_vel_d': RFoot_pos_d,
               'FSM': np.diag([self.fsm])}
                
    def logParse(self, x):
        #Parses the log data and returns in human readable form - test in ipynb
        i = self.log_idx

        ## State Vectors ##
        COM_POS = x[:i[1],:]
        COM_VEL = x[i[1]:i[2],:]
        T_POS = x[i[2]:i[3],:]
        T_VEL = x[i[3]:i[4],:]
        LEFT_FOOT = x[i[4]:i[5],:]
        RIGHT_FOOT = x[i[5]:i[6],:]
        LEFT_FOOT_VEL = x[i[6]:i[7],:]
        RIGHT_FOOT_VEL = x[i[7]:i[8],:]

        ## Targets ##
        COM_POS_DESIRED = x[i[8]:i[9],:]
        COM_VEL_DESIRED = x[i[9]:i[10],:]
        Torso_POS_DESIRED = x[i[10]:i[11],:]
        Torso_VEL_DESIRED = x[i[11]:i[12],:]
        LFt_POS_DESIRED = x[i[12]:i[13],:]
        RFt_POS_DESIRED = x[i[13]:i[14],:]
        FSM = x[i[14]:i[15],:]

        Torques = x[i[15]:i[16],:]
        Cost = x[i[16]:i[17],:]
        ContactForces = x[i[17]:,:]
        LeftContactForces = ContactForces[:3]
        RightContactForces = ContactForces[3:]

        return COM_POS, COM_VEL, T_POS, T_VEL, LEFT_FOOT, RIGHT_FOOT, COM_POS_DESIRED, COM_VEL_DESIRED, Torso_POS_DESIRED, Torso_VEL_DESIRED, LFt_POS_DESIRED, RFt_POS_DESIRED, FSM, Torques, Cost, LeftContactForces, RightContactForces
        
    def logCB(self, context, output):
        # COM, Torso Angle, Foot pos and velocities
        statePacket = fetchStates(self.plant_context, self.plant)

        # Desired COM, Torso, Desired Foot angles
        trackPacket = self.fetchTrackParams()
                
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

        ## Torque Outputs ##        
        outVector = np.concatenate((outVector, self.usol.flatten()))
        idx.append(outVector.shape[0])
        ## Costs ##
        outVector = np.concatenate((outVector, np.diag([self.solutionCost]).flatten()))
        idx.append(outVector.shape[0])
        ## Contact Forces ##
        outVector = np.concatenate((outVector, self.contactForces.flatten()))
        idx.append(outVector.shape[0])

        self.log_idx = idx
        output.SetFromVector(outVector)        

        

    def solveQP(self, context):
        ## Get the context from the diagram ##
        x = self.EvalVectorInput(context, self.robot_state_input_port_index).get_value()
        t = context.get_time()

        ## Update the internal context ##
        self.plant.SetPositionsAndVelocities(self.plant_context, x)

        self.fsm = int(self.EvalAbstractInput(context, self.phaseInput).get_value().get_value()) - 1


        ## Create the Mathematical Program ##
        qp = MathematicalProgram()
        u = qp.NewContinuousVariables(self.plant.num_actuators(), "u")
        vdot = qp.NewContinuousVariables(self.plant.num_velocities(), "vdot")
        lambda_c = qp.NewContinuousVariables(6, "lambda_c")

        ## Quadratic Costs ##
        whatToTrack = self.whatToTrack[self.fsm]
        for track in whatToTrack:
            #Get desired trajectory and costs
            if 'torso' in track:
                if self.fsm == PREFLIGHT:
                    traj = PiecewisePolynomial(np.zeros(1,))
                else:
                    traj = BasicVector(np.zeros(1,))
            else:
                traj = self.EvalAbstractInput(context, self.traj_input_ports[self.fsm]).get_value()
            cost = self.Costs[track]
            
            ## In Air Phase has foot trajectory to track
            if self.fsm==FLIGHT:
                qp.AddQuadraticCost( 1e-3*u.T@u )
                if 'foot' in track:
                    for footNum in [0, 1]:
                        # Get what to track and system states
                        yddot_cmd_i, J_i, JdotV_i = self.tracking_objective_air.Update(t, traj, track, footNum)
                        yii = JdotV_i + J_i@vdot
                        qp.AddQuadraticCost( (yddot_cmd_i - yii).T@cost@(yddot_cmd_i - yii) )
                else:
                    yddot_cmd_i, J_i, JdotV_i = self.tracking_objective_air.Update(t, traj, track, footNum)
                    yii = JdotV_i + J_i@vdot
                    qp.AddQuadraticCost( (yddot_cmd_i - yii).T@cost@(yddot_cmd_i - yii) )
                        
            ## Preflight and Land phase has other things to track
            if self.fsm==PREFLIGHT:
                yddot_cmd_i, J_i, JdotV_i = self.tracking_objective_preflight.Update(t, traj, track, finiteState=self.fsm)    
                yii = JdotV_i + J_i@vdot
                qp.AddQuadraticCost( (yddot_cmd_i - yii).T@cost@(yddot_cmd_i - yii) )                
            if self.fsm==LAND:
                yddot_cmd_i, J_i, JdotV_i = self.tracking_objective_land.Update(t, traj, track, finiteState=self.fsm)    
                yii = JdotV_i + J_i@vdot
                qp.AddQuadraticCost( (yddot_cmd_i - yii).T@cost@(yddot_cmd_i - yii) )

        # qp.AddQuadraticCost(1e-3*(self.usol-u).T@(self.usol-u) ) #(np.random.random(self.plant.CalcMassMatrix(self.plant_context).shape) - 0.5)
        # Calculate terms in the manipulator equation
        M = self.plant.CalcMassMatrix(self.plant_context)
        Cv = self.plant.CalcBiasTerm(self.plant_context)    
        # Drake considers gravity to be an external force (on the right side of the dynamics equations), 
        # so we negate it to match the homework PDF and slides
        G = -self.plant.CalcGravityGeneralizedForces(self.plant_context)
        B = self.plant.MakeActuationMatrix()

        #Calculate Contact Jacobians
        if self.fsm != FLIGHT:
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
            qp.AddLinearConstraint(u[i], np.array([-1200]), np.array([1200]))

        # Solve the QP
        solver = OsqpSolver()
        qp.SetSolverOption(solver.id(), "max_iter", self.max_iter)
        qp.SetSolverOption(solver.id(), "eps_abs", 1e-5)        
        if self.fsm == PREFLIGHT:
            uinit = self.dircolUTraj.vector_values([t]).flatten()
        else:
            uinit = self.usol
        qp.SetInitialGuess(u, uinit)

        result = solver.Solve(qp)

        self.solutionCost = result.get_optimal_cost()
        self.contactForces = result.GetSolution(lambda_c)

        # If we exceed iteration limits use the previous solution
        if not result.is_success():            
            module_logger.debug(f"Solver not working, pal! at t:{t}")            
            usol = self.usol
        else:
            usol = result.GetSolution(u)                    
        if  np.linalg.norm(usol)>1000:
            usol = 1000*usol/np.linalg.norm(usol)
        return usol

    ## Ignore ##
    def CalcTorques(self, context: Context, output: BasicVector) -> None:
        usol = self.solveQP(context)
        self.usol = usol
        output.SetFromVector(usol)
    def get_traj_input_port(self, traj_name):
        return self.get_input_port(self.traj_input_ports[traj_name])    
    def get_state_input_port(self):
        return self.get_input_port(self.robot_state_input_port_index)


    def get_preflightinput_port_index(self):
        return self.get_input_port(self.dircolInput)

    def get_flightinput_port_index(self):
        return self.get_input_port(self.flightInput)
    def get_landinginput_port_index(self):
        return self.get_input_port(self.landInput)

    def get_phase_port_index(self):
        return self.get_input_port(self.phaseInput)
