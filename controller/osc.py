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
from utils import calculateDoubleContactJacobians

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
        self.whatToTrack = ['COM', 'torso']
        COMParams = {'Kp': np.diag([100, 0, 100]), 'Kd': np.diag([100, 0, 100])/10 , 'saturations': 10} #Max Lim: 1 G
        TorsoParams = {'Kp': np.diag([5]), 'Kd': np.diag([1]) , 'saturations': 15*np.pi/180} #Max Lim: 5 deg/s
        ## Cost Weights ##
        self.WCOM = np.eye(3,3)
        self.WTorso = np.diag([5*np.pi/180*(self.WCOM.max()/10)]) #Maybe a consistant way to set the weights
        self.Costs = {'COM': self.WCOM, 'torso' : self.WTorso} 
        ##_______________________________________________##

        ## ______________Solver Parameters______________ ##
        self.max_iter = 3000
        ##_______________________________________________##


        # !WARNING : IT IS ASSUMED THAT OSC AND OSC_TRACKING SHARE THE SAME PLANT!
        self.tracking_objective = tracking_objective(self.plant, self.plant_context, COMParams, TorsoParams, None, polyTraj=polyTraj)


        ## ______________Declaring Ports______________ ##        
        self.robot_state_input_port_index = self.DeclareVectorInputPort("x", self.plant.num_positions() + self.plant.num_velocities()).get_index()
        if polyTraj:
            intype = [PiecewisePolynomial(),PiecewisePolynomial()]
        else:
            intype = [BasicVector(3),BasicVector(1)]
        self.traj_input_ports = {
            self.whatToTrack[0]: self.DeclareAbstractInputPort("com_traj", AbstractValue.Make(intype[0])).get_index(),
            self.whatToTrack[1]: self.DeclareAbstractInputPort("base_joint_traj", AbstractValue.Make(intype[1])).get_index()}
        
        self.torque_output_port = self.DeclareVectorOutputPort("u", self.plant.num_actuators(), self.CalcTorques)
        self.u = np.zeros((self.plant.num_actuators()))
        ##_______________________________________________##
        

    def solveQP(self, context):
        
        ## Get the context from the diagram ##
        x = self.EvalVectorInput(context, self.robot_state_input_port_index).get_value()
        t = context.get_time()

        ## Update the internal context ##
        self.plant.SetPositionsAndVelocities(self.plant_context, x)

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
            
            #Get what to track and system states
            yddot_cmd_i, J_i, JdotV_i = self.tracking_objective.Update(t, traj, track)

            yii = JdotV_i + J_i@vdot
            qp.AddQuadraticCost( (yddot_cmd_i - yii).T@cost@(yddot_cmd_i - yii) )

        
        # Calculate terms in the manipulator equation
        M = self.plant.CalcMassMatrix(self.plant_context)
        Cv = self.plant.CalcBiasTerm(self.plant_context)    
        # Drake considers gravity to be an external force (on the right side of the dynamics equations), 
        # so we negate it to match the homework PDF and slides
        G = -self.plant.CalcGravityGeneralizedForces(self.plant_context)
        B = self.plant.MakeActuationMatrix()

        #Calculate Contact Jacobians
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

        # Solve the QP
        solver = OsqpSolver()
        qp.SetSolverOption(solver.id(), "max_iter", self.max_iter)

        result = solver.Solve(qp)

        # If we exceed iteration limits use the previous solution
        if not result.is_success():
            print("Solver not working, pal!!!")
            usol = self.u
        else:
            usol = result.GetSolution(u)
            self.u = usol        

        return usol

    ## Ignore ##
    def CalcTorques(self, context: Context, output: BasicVector) -> None:
        usol = self.solveQP(context)
        output.SetFromVector(usol)
    def get_traj_input_port(self, traj_name):
        return self.get_input_port(self.traj_input_ports[traj_name])    
    def get_state_input_port(self):
        return self.get_input_port(self.robot_state_input_port_index)



            
      





