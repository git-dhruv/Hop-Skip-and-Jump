import numpy as np
from typing import List, Tuple

from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import Context, LeafSystem, BasicVector
from pydrake.trajectories import Trajectory, PiecewisePolynomial
from pydrake.solvers import MathematicalProgram, OsqpSolver
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform

from osc_tracking_objective import *

from fsm_utils import LEFT_STANCE, RIGHT_STANCE


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
                np.diag([100, 100, 100]), np.diag([100, 100, 100])/10
            ),
            "base_joint_traj": JointAngleTrackingObjective(
                self.plant, self.plant_context, [LEFT_STANCE, RIGHT_STANCE],
                np.eye(1), np.eye(1), "planar_roty"
            )
        }


        Wcom = np.eye(3,3)

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
        self.torque_output_port = self.DeclareVectorOutputPort(
            "u", self.plant.num_actuators(), self.CalcTorques
        )

        self.u = np.zeros((self.plant.num_actuators()))

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
            # print(traj_name)
            traj = self.EvalAbstractInput(context, self.traj_input_ports[traj_name]).get_value()
            self.tracking_objectives[traj_name].Update(t, traj, fsm)

        '''Set up and solve the QP '''
        prog = MathematicalProgram()

        # Make decision variables
        u = prog.NewContinuousVariables(self.plant.num_actuators(), "u")
        vdot = prog.NewContinuousVariables(self.plant.num_velocities(), "vdot")
        lambda_c = prog.NewContinuousVariables(3, "lambda_c")

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
        prog.AddQuadraticCost(1e-5*vdot.T@vdot)


        # Calculate terms in the manipulator equation
        for fsm in [0,1]:
            J_c, J_c_dot_v = self.CalculateContactJacobian(fsm)
            M = self.plant.CalcMassMatrix(self.plant_context)
            Cv = self.plant.CalcBiasTerm(self.plant_context)
        
            # Drake considers gravity to be an external force (on the right side of the dynamics equations), 
            # so we negate it to match the homework PDF and slides
            G = -self.plant.CalcGravityGeneralizedForces(self.plant_context)
            B = self.plant.MakeActuationMatrix()

            # TODO: Add the dynamics constraint
            #Forcing reshape because Gradescope is angry
            prog.AddLinearEqualityConstraint(M@vdot + Cv + G - B@u - J_c.T@lambda_c, np.zeros((7,)))

            # TODO: Add Contact Constraint
            prog.AddLinearEqualityConstraint(J_c_dot_v + J_c@vdot, np.zeros((3,1)))

        # TODO: Add Friction Cone Constraint assuming mu = 1
        mu = 1
        A_fric = np.array([[1,0,-mu],
                        [-1,0,-mu]])
        prog.AddLinearConstraint(A_fric@lambda_c.reshape(3,1), -1e10*np.ones((2,1)), np.zeros((2,1))) #Seems fine
        prog.AddLinearEqualityConstraint(lambda_c[1] == 0)


        # Solve the QP
        solver = OsqpSolver()
        prog.SetSolverOption(solver.id(), "max_iter", 2000)

        result = solver.Solve(prog)

        # If we exceed iteration limits use the previous solution
        if not result.is_success():
            usol = self.u
        else:
            usol = result.GetSolution(u)
            self.u = usol

        return usol, prog

    def CalcTorques(self, context: Context, output: BasicVector) -> None:
        usol, _ = self.SetupAndSolveQP(context)
        output.SetFromVector(usol)

