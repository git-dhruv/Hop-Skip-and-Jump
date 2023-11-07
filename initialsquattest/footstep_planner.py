import numpy as np
from typing import List, Tuple
from scipy.linalg import expm

from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import Context, LeafSystem, BasicVector, DiscreteValues, EventStatus
from pydrake.trajectories import Trajectory, PiecewisePolynomial
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform

import importlib
import fsm_utils
importlib.reload(fsm_utils)

from fsm_utils import LEFT_STANCE, RIGHT_STANCE, get_fsm,\
                      time_until_switch, time_since_switch 
from osc_tracking_objective import PointOnFrame


'''
    Footstep planner for a planar walking robot - do not modify
    Generates footstep and CoM trajectories to be tracked by the OSC
'''

def make_exponential_spline(K: np.ndarray, 
                            A: np.ndarray, 
                            alpha: np.ndarray, 
                            pp_part: PiecewisePolynomial, 
                            n: int = 10) -> PiecewisePolynomial: 
    """
        Helper function to approximate the solution to a linear system as 
        a cubic spline, since ExponentialPlusPiecewisePolynomial doesn't have python bindings
        (https://drake.mit.edu/doxygen_cxx/classdrake_1_1trajectories_1_1_exponential_plus_piecewise_polynomial.html)
    """
    
    time_vect = np.linspace(pp_part.start_time(), pp_part.end_time(), n).tolist()
    knots = [ 
        np.expand_dims(
            K @ expm((t - time_vect[0]) * A) @ alpha + pp_part.value(t).ravel(), 
            axis=1
        )\
        for t in time_vect
    ]
    return PiecewisePolynomial.CubicShapePreserving(time_vect, knots)

class LipTrajPlanner(LeafSystem):
    def __init__(self):
        """
            Constructor for a Linear Inverted Pendulum footstep and CoM planner.
            Implements the 1-step look ahead controller from https://arxiv.org/abs/2008.10763
        """
        LeafSystem.__init__(self)

        ''' Load the MultibodyPlant '''
        self.plant = MultibodyPlant(0.0)
        self.parser = Parser(self.plant)
        self.parser.AddModels("planar_walker.urdf")
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetBodyByName("base").body_frame(),
            RigidTransform.Identity()
        )
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        ''' Assign contact frames '''
        # Note that during right stance, the left leg is the swing foot, and vice versa
        self.swing_foot_points = {
            RIGHT_STANCE: PointOnFrame(
                self.plant.GetBodyByName("left_lower_leg").body_frame(),
                np.array([0, 0, -0.5])
            ),
            LEFT_STANCE: PointOnFrame(
                self.plant.GetBodyByName("right_lower_leg").body_frame(),
                np.array([0, 0, -0.5])
            )
        }
        self.stance_foot_points = {
            LEFT_STANCE: self.swing_foot_points[RIGHT_STANCE],
            RIGHT_STANCE: self.swing_foot_points[LEFT_STANCE]
        }
        ''' input ports '''
        self.robot_state_input_port_index = self.DeclareVectorInputPort(
            "x", 
            self.plant.num_positions() + self.plant.num_velocities()
        ).get_index()
        
        self.walking_speed_input_port_index = self.DeclareVectorInputPort("vdes", 1).get_index()
        
        ''' output ports'''
        self.com_traj_output_port_index = self.DeclareAbstractOutputPort(
            "com_traj", 
            lambda: AbstractValue.Make(PiecewisePolynomial()),
            self.CalcComTraj
        ).get_index()
        self.swing_foot_traj_output_port_index = self.DeclareAbstractOutputPort(
            "swing_foot_traj", 
            lambda: AbstractValue.Make(PiecewisePolynomial()),
            self.CalcSwingFootTraj
        ).get_index()
        
        ''' discrete states '''
        self.prev_fsm_state_idx = self.DeclareDiscreteState(1)
        self.foot_position_at_liftoff_idx = self.DeclareDiscreteState(3)

        ''' control parameters '''
        self.H = 0.9
        self.m = self.plant.CalcTotalMass(self.plant_context)

    def get_state_input_port(self):
        return self.get_input_port(self.robot_state_input_port_index)

    def get_walking_speed_input_port(self):
        return self.get_input_port(self.walking_speed_input_port_index)

    def get_com_traj_output_port(self):
        return self.get_output_port(self.com_traj_output_port_index)

    def get_swing_foot_traj_output_port(self):
        return self.get_output_port(self.swing_foot_traj_output_port_index)

    def DiscreteStateUpdate(self, context: Context, discrete_state: DiscreteValues) -> EventStatus:
        t = context.get_time()
        fsm = get_fsm(t)

        prev_fsm = int(discrete_state.get_value(self.prev_fsm_state_idx))

        # When the fsm switches, record the foot location to use in constructing the spline
        if fsm != prev_fsm:
            robot_state = self.EvalVectorInput(context, self.robot_state_input_port_index).value()
            self.plant.SetPositionsAndVelocities(self.plant_context, robot_state)
            swing_foot_pos = self.plant.CalcPointsPositions(
                self.plant_context,
                self.swing_foot_points[fsm].frame,
                self.swing_foot_points[fsm].pt,
                self.plant.world_frame()
            )
            discrete_state.get_mutable_vector(self.foot_position_at_liftoff_idx).set_value(swing_foot_pos)

        # update the previous fsm value to the current fsm value
        discrete_state.get_mutable_vector(self.prev_fsm_state_idx).set_value(np.array([fsm]))
        return EventStatus().Succeeded()

    def CalcAlipState(self, fsm: int, robot_state: np.ndarray) -> Trajectory:
        self.plant.SetPositionsAndVelocities(self.plant_context, robot_state)
        stance_foot = self.stance_foot_points[fsm]
        stance_foot_pos = self.plant.CalcPointsPositions(
            self.plant_context, 
            stance_foot.frame, 
            stance_foot.pt, 
            self.plant.world_frame()
        ).ravel()
        com_pos = self.plant.CalcCenterOfMassPositionInWorld(self.plant_context).ravel()
        L = self.plant.CalcSpatialMomentumInWorldAboutPoint(self.plant_context, stance_foot_pos).rotational().ravel()
        return np.array([com_pos[0] - stance_foot_pos[0], L[1]])

    def ConstructAlipComTraj(self, t_start, t_end, alip_state, stance_foot_location):
        Y = np.zeros((3, 2))
        Y[0] = stance_foot_location[0] * np.ones((2,))
        Y[2] = self.H * np.ones((2,))

        if t_end - t_start <= 1e-5:
            t_start = t_end - 1e-4

        pp_part = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            np.array([t_start, t_end]),
            Y,
            np.zeros((3,)),
            np.zeros((3,))
        )
        A = np.fliplr(np.diag([1.0 / (self.m * self.H), self.m * 9.81]))
        K = np.zeros((3,2))
        K[:2] = np.eye(2)
        return make_exponential_spline(K, A, alip_state, pp_part)

    def CalcSwingFootTraj(self, context: Context, output: Trajectory) -> None:
        state = self.EvalVectorInput(context, self.robot_state_input_port_index).value()
        t = context.get_time()
        fsm = get_fsm(t)
        start_time = t - time_since_switch(t)
        end_time = t + time_until_switch(t)
        
        alip_state = self.CalcAlipState(fsm, state)
        A = np.fliplr(np.diag([1.0 / (self.m * self.H), self.m * 9.81]))
        alip_pred = expm(time_until_switch(t) * A) @ alip_state

        stance_foot = self.stance_foot_points[fsm]
        stance_foot_pos = self.plant.CalcPointsPositions(
            self.plant_context, stance_foot.frame, stance_foot.pt, self.plant.world_frame()).ravel()
        com_pred = self.ConstructAlipComTraj(t, end_time, alip_state, stance_foot_pos).value(end_time).ravel()

        vdes = self.EvalVectorInput(context, self.walking_speed_input_port_index).get_value()
        Ly_des = vdes[0] * self.H * self.m
        omega = np.sqrt(9.81/self.H)
        T = time_until_switch(t)
        Ly_0 = alip_pred[-1]

        # swing foot update from https://arxiv.org/abs/2008.10763
        p_x_foot_to_com = (Ly_des - np.cosh(omega * T) * Ly_0) / (self.m * self.H * omega * np.sinh(omega * T))

        swing_pos_at_liftoff = context.get_discrete_state(self.foot_position_at_liftoff_idx).get_value()

        Y0 = swing_pos_at_liftoff
        Y2 = np.zeros((3,))
        Y2[0] =  com_pred[0] - p_x_foot_to_com 
        Y1 = 0.5 * (Y0 + Y2)
        Y1[-1] = 0.01
        output.set_value(
            PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                [start_time, 0.5*(start_time + end_time),  end_time],
                [np.expand_dims(y, axis=1) for y in [Y0, Y1, Y2]],
                np.zeros((3,)),
                np.array([0, 0, -0.5]) # negative foot z velocity at the end of swing to help establish contact
            )
        )

    def CalcComTraj(self, context: Context, output) -> None:
        state = self.EvalVectorInput(context, self.robot_state_input_port_index).value()
        t = context.get_time()
        fsm = get_fsm(t)
        end_time = t + time_until_switch(t)
        
        alip_state = self.CalcAlipState(fsm, state)
        stance_foot = self.stance_foot_points[fsm]
        stance_foot_pos = self.plant.CalcPointsPositions(
            self.plant_context, stance_foot.frame, stance_foot.pt, self.plant.world_frame()).ravel()

        com_traj = self.ConstructAlipComTraj(t, end_time, alip_state, stance_foot_pos)
        output.set_value(com_traj)

