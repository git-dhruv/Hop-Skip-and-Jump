"""
Lower level instantaneous QP
Tracks Desired COM with both feet on ground
"""
import numpy as np
from typing import List, Tuple

from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import Context, LeafSystem, BasicVector
from pydrake.trajectories import Trajectory, PiecewisePolynomial
from pydrake.solvers import MathematicalProgram, OsqpSolver
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform


from pydrake.multibody.all import JacobianWrtVariable
from osc_tracking_objective import *

LEFT_STANCE = 0
RIGHT_STANCE = 1


class OperationalSpaceWalkingController(LeafSystem):
    def __init__(self):
        """
            Constructor for the operational space controller (Do Not Modify).
            We load a drake MultibodyPlant representing the planar walker
            to use for kinematics and dynamics calculations.

            We then define tracking objectives, and finally,
            we declare input ports and output ports
        """
        LeafSystem.__init__(self)

        ### Internal Plant Model for simulation ###
        self.plant = MultibodyPlant(0.0)
        self.parser = Parser(self.plant)
        self.parser.AddModels("/home/dhruv/final/initialsquattest/planar_walker.urdf")
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetBodyByName("base").body_frame(),
            RigidTransform.Identity()
        )
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        ''' Assign contact frames '''
        self.contact_points = {
            LEFT_STANCE: PointOnFrame(
                self.plant.GetBodyByName("left_lower_leg").body_frame(),
                np.array([0, 0, -0.5])
            ),
            RIGHT_STANCE: PointOnFrame(
                self.plant.GetBodyByName("right_lower_leg").body_frame(),
                np.array([0, 0, -0.5])
            )
        }


        ''' Initialize tracking objectives '''
        self.tracking_objectives = {
            "com_traj": CenterOfMassPositionTrackingObjective(
                self.plant, self.plant_context, [LEFT_STANCE, RIGHT_STANCE],
                np.diag([500, 0, 500]), np.diag([100, 0, 100])/10
            ),
            "base_joint_traj": JointAngleTrackingObjective(
                self.plant, self.plant_context, [LEFT_STANCE, RIGHT_STANCE],
                10*np.eye(1), np.eye(1), "planar_roty"
            )
        }

        ### @ask ta about this non convexity ###
        Wcom = np.eye(3,3)
        Wcom[1,1] = 0
        Wcom[2,2] = 4
        Wcom[0,0] = 4

        self.tracking_costs = {
            "com_traj": Wcom,
            "base_joint_traj": np.eye(1)
        }

        self.trajs = self.tracking_objectives.keys()

        ''' Declare Input Ports '''
        # State input port
        self.robot_state_input_port_index = self.DeclareVectorInputPort(
            "x", self.plant.num_positions() + self.plant.num_velocities()
        ).get_index()

        # Trajectory Input Ports
        # trj = BasicVector()
        self.traj_input_ports = {
            "com_traj": self.DeclareAbstractInputPort("com_traj", AbstractValue.Make(BasicVector(3))).get_index(),
            "base_joint_traj": self.DeclareAbstractInputPort("base_joint_traj", AbstractValue.Make(BasicVector(1))).get_index()}

        # Define the output ports
        self.torque_output_port = self.DeclareVectorOutputPort("u", self.plant.num_actuators(), self.CalcTorques)

        ## Log callback ##
        self.logging_port = self.DeclareVectorOutputPort("metrics", BasicVector(6), self.logCB)

        self.u = np.zeros((self.plant.num_actuators()))

    def logCB(self, context, output):
        ### For now we are tracking the COM Z, Zdot and Zddot ###
        #Get the current state from the input port
        state = self.EvalVectorInput(context, self.robot_state_input_port_index).value()
        #Simulate our robot to the current state
        self.plant.SetPositionsAndVelocities(self.plant_context, state)
        #Extract the Center of Mass of our robot
        com_pos = self.plant.CalcCenterOfMassPositionInWorld(self.plant_context).ravel()

        J = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(self.plant_context,
                                                                     JacobianWrtVariable.kV,
                                                                     self.plant.world_frame(),
                                                                     self.plant.world_frame())
        Jdv = self.plant.CalcBiasCenterOfMassTranslationalAcceleration(self.plant_context,
                                                                     JacobianWrtVariable.kV,
                                                                     self.plant.world_frame(),
                                                                     self.plant.world_frame())
        
        com_vel = (J @ self.plant.GetVelocities(self.plant_context)).ravel()
        # com_acc = (J @ self.plant.GetVelocities(self.context)).ravel() How tf to calculate this

        
        output.SetFromVector(np.concatenate((com_pos, com_vel)))

    def get_traj_input_port(self, traj_name):
        return self.get_input_port(self.traj_input_ports[traj_name])
    
    def get_state_input_port(self):
        return self.get_input_port(self.robot_state_input_port_index)

    def CalculateContactJacobian(self, fsm: int) -> Tuple[np.ndarray, np.ndarray]:
        """
            For a given finite state, LEFT_STANCE or RIGHT_STANCE, calculate the
            Jacobian terms for the contact constraint, J and Jdot * v.

            As an example, see CalcJ and CalcJdotV in PointPositionTrackingObjective

            use self.contact_points to get the PointOnFrame for the current stance foot
        """
        J = np.zeros((3, self.plant.num_velocities()))
        JdotV = np.zeros((3,))

        # TODO - STUDENT CODE HERE:
        pt_to_track = self.contact_points[fsm]
        J = self.plant.CalcJacobianTranslationalVelocity(
            self.plant_context, JacobianWrtVariable.kV, pt_to_track.frame,
            pt_to_track.pt, self.plant.world_frame(), self.plant.world_frame()
        )

        JdotV = self.plant.CalcBiasTranslationalAcceleration(
            self.plant_context, JacobianWrtVariable.kV, pt_to_track.frame,
            pt_to_track.pt, self.plant.world_frame(), self.plant.world_frame()
        ).ravel()

        return J, JdotV

    def SetupAndSolveQP(self,  context: Context) -> Tuple[np.ndarray, MathematicalProgram]:

        # First get the state, time, and fsm state
        x = self.EvalVectorInput(context, self.robot_state_input_port_index).get_value()
        t = context.get_time()
        fsm = 1

        # Update the plant context with the current position and velocity
        self.plant.SetPositionsAndVelocities(self.plant_context, x)

        # Update tracking objectives
        for traj_name in self.trajs:
            traj = self.EvalAbstractInput(context, self.traj_input_ports[traj_name]).get_value()
            self.tracking_objectives[traj_name].Update(t, traj, fsm)

        '''Set up and solve the QP '''
        prog = MathematicalProgram()

        # Make decision variables
        u = prog.NewContinuousVariables(self.plant.num_actuators(), "u")
        vdot = prog.NewContinuousVariables(self.plant.num_velocities(), "vdot")
        lambda_c = prog.NewContinuousVariables(6, "lambda_c")

        # Add Quadratic Cost on Desired Acceleration
        for traj_name in self.trajs:
            obj = self.tracking_objectives[traj_name]
            yddot_cmd_i = obj.yddot_cmd
            J_i = obj.J
            JdotV_i = obj.JdotV
            W_i = self.tracking_costs[traj_name]

            # TODO: Add Cost per tracking objective
            yii = JdotV_i + J_i@vdot
            prog.AddQuadraticCost((yddot_cmd_i - yii).T@W_i@(yddot_cmd_i - yii))

            

        # TODO: Add Quadratic Cost on vdot using self.gains.w_vdot
        # prog.AddQuadraticCost(1e-1*vdot.T@vdot)


        
        #I concatenate both contact Jacobians and then lamba c is modified to make 6,1
        J_c, J_c_dot_v = self.CalculateContactJacobian(0)
        J_c_2, J_c_dot_v_2 = self.CalculateContactJacobian(1)
        J_c = np.row_stack((J_c, J_c_2))
        J_c_dot_v = np.row_stack((J_c_dot_v.reshape(-1,1), J_c_dot_v_2.reshape(-1,1)))
        
        # Calculate terms in the manipulator equation
        M = self.plant.CalcMassMatrix(self.plant_context)
        Cv = self.plant.CalcBiasTerm(self.plant_context)
    
        # Drake considers gravity to be an external force (on the right side of the dynamics equations), 
        # so we negate it to match the homework PDF and slides
        G = -self.plant.CalcGravityGeneralizedForces(self.plant_context)
        B = self.plant.MakeActuationMatrix()


        #Dynamics
        prog.AddLinearEqualityConstraint(M@vdot + Cv + G - B@u - J_c.T@lambda_c, np.zeros((7,)))

        #Contact
        prog.AddLinearEqualityConstraint(J_c_dot_v + (J_c@vdot).reshape(-1,1), np.zeros((6,1)))

        # Friction Cone Constraint
        mu = 1
        A_fric = np.array([[1, 0, -mu, 0, 0, 0], # Friction constraint for left foot, positive x-direction
                        [-1, 0, -mu, 0, 0, 0],   # Friction constraint for left foot, negative x-direction
                        [0, 0, 0, 1, 0, -mu],    # Friction constraint for right foot, positive x-direction
                        [0, 0, 0, -1, 0, -mu]])  # Friction constraint for right foot, negative x-direction

        # The constraint is applied to the 6x1 lambda_c vector
        prog.AddLinearConstraint(A_fric @ lambda_c.reshape(6, 1), -np.inf * np.ones((4, 1)), np.zeros((4, 1)))
        prog.AddLinearEqualityConstraint(lambda_c[1] == 0)
        prog.AddLinearEqualityConstraint(lambda_c[4] == 0)


        # Solve the QP
        solver = OsqpSolver()
        prog.SetSolverOption(solver.id(), "max_iter", 2000)

        result = solver.Solve(prog)

        # If we exceed iteration limits use the previous solution
        if not result.is_success():
            print("Solver not working, pal!!!")
            usol = self.u
        else:
            usol = result.GetSolution(u)
            self.u = usol

        return usol, prog

    def CalcTorques(self, context: Context, output: BasicVector) -> None:
        usol, _ = self.SetupAndSolveQP(context)
        output.SetFromVector(usol)
