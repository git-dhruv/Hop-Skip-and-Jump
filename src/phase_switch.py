from utils import *
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import Context, LeafSystem, BasicVector, DiscreteValues, EventStatus
from pydrake.trajectories import Trajectory, PiecewisePolynomial
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.multibody.all import JacobianWrtVariable


class PhaseSwitch(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self.plant = MultibodyPlant(0.0)
        self.parser = Parser(self.plant)
        self.parser.AddModels("src/models/planar_walker.urdf")
        self.plane.WeldFrames(self.plant.world_frame(), self.plant.GetBodyByName("base").body_frame(), RigidTransform.Identity())
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()


        traj_dim  = self.plant.num_positions()+self.plant.num_velocities()
        self.preflight_trajectory_port_index = self.DeclareVectorInputPort("preflight_Trajectory", traj_dim).get_index()
        self.aerial_trajectory_port_index = self.DeclareVectorInputPort("Aerial_Trajectory", traj_dim).get_index()
        self.landing_trajectory_port_index = self.DeclareVectorInputPort("Landing_Trajectory", traj_dim).get_index()

        self.osc_trajectory_port_index = self.DeclareVectorOutputPort("OSC_Trajectory", lambda: AbstractValue.Make(BasicVector(traj_dim), self.DeterminePhase)).get_index()

    def DeterminePhase(self, context, robotstate):
        #Do something
        pass

    def get_preflight_port_index(self): return self.preflight_trajectory_port_index
    def get_aerial_trajectory_port_index(self): return self.aerial_trajectory_port_index
    def get_landing_trajectory_port_index(self): return self.landing_trajectory_port_index

    