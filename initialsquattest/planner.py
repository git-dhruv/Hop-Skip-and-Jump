
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import Context, LeafSystem, BasicVector, DiscreteValues, EventStatus
from pydrake.trajectories import Trajectory, PiecewisePolynomial
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.multibody.all import JacobianWrtVariable

import numpy as np
from scipy.linalg import expm

class COMPlanner(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)

        #Make internal dynamics model to get the COM and stuff#
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

        #Input Ports are States of drake and Desired reference
        self.robot_state_input_port_index = self.DeclareVectorInputPort("x", self.plant.num_positions() + self.plant.num_velocities()).get_index()
        #Desired COM (Z) Trajectory
        self.walking_speed_input_port_index = self.DeclareVectorInputPort("zdes", 1).get_index()

        #Output Port
        self.com_traj_output_port_index = self.DeclareAbstractOutputPort("com_traj", lambda: AbstractValue.Make(BasicVector(3)),self.calcDesiredZddot).get_index()

    def calcCOM(self, context):
        # if state is None:
        #     state = self.EvalVectorInput(self.plant_context, self.robot_state_input_port_index).value()
        state = self.EvalVectorInput(context, self.robot_state_input_port_index).value()
        self.plant.SetPositionsAndVelocities(self.plant_context, state)
        com_pos = self.plant.CalcCenterOfMassPositionInWorld(self.plant_context).ravel()
        return com_pos
    
    def calcDesiredZddot(self, context: Context, output):
        z_des = self.EvalVectorInput(context, self.walking_speed_input_port_index).value()
        z_dd_des = -(z_des - self.calcCOM(context)[-1])
        output.set_value(BasicVector([0,0,0.3]))

    def CalcYdot(self, state) -> np.ndarray:
        if state is None:
            state = self.EvalVectorInput(self.plant_context, self.robot_state_input_port_index).value()
        self.plant.SetPositionsAndVelocities(self.plant_context, state)
        return (self.CalcJ() @ self.plant.GetVelocities(self.context)).ravel()

    def CalcJ(self) -> np.ndarray:
        return self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
            self.context, JacobianWrtVariable.kV, self.plant.world_frame(), self.plant.world_frame())
    def get_com_traj_output_port(self):
        return self.get_output_port(self.com_traj_output_port_index)
    def get_com_zdes_input_port(self):
        return self.get_input_port(self.walking_speed_input_port_index)
    def get_com_state_input_port(self):
        return self.get_input_port(self.robot_state_input_port_index)