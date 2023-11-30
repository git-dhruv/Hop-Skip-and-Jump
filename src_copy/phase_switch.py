from utils import *
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import Context, LeafSystem, BasicVector, DiscreteValues, EventStatus
from pydrake.trajectories import Trajectory, PiecewisePolynomial
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.multibody.all import JacobianWrtVariable


class PhaseSwitch(LeafSystem):
    def __init__(self, height, jump_time, x_traj, z_des):
        LeafSystem.__init__(self)
        self.plant = MultibodyPlant(0.0)
        self.parser = Parser(self.plant)
        self.parser.AddModels("models/planar_walker.urdf")
        self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetBodyByName("base").body_frame(), RigidTransform.Identity())
        self.plant.Finalize()
        self.req_height = height
        self.jump_time = jump_time
        self.x_traj = x_traj
        self.z_des = z_des
        self.plant_context = self.plant.CreateDefaultContext()

        self.preflightoutput_trajectory_port_index = self.DeclareAbstractOutputPort("OSCPreflight_Trajectory", lambda: AbstractValue.Make(PiecewisePolynomial()), self.SetPreflightOutput).get_index()
        self.flightoutput_trajectory_port_index = self.DeclareAbstractOutputPort("OSCLFight_Trajectory", lambda: AbstractValue.Make(BasicVector(6)), self.SetFlightOutput).get_index()
        self.landingoutput_trajectory_port_index = self.DeclareAbstractOutputPort("OSCLanding_Trajectory", lambda: AbstractValue.Make(BasicVector(3)), self.SetLandingOutput).get_index()
        self.phaseoutput_port_index = self.DeclareAbstractOutputPort("Phase", lambda: AbstractValue.Make(BasicVector(1)), self.PhaseOutput).get_index()

    def DeterminePhase(self):
        required_vel = (2*9.18*self.req_height)**0.5
        statePacket = fetchStates(self.plant_context, self.plant)
        t  = self.plant_context.get_time()
        phase = 0
        if phase == 0:
            phase = 1
        if (t>1.1*self.jump_time or statePacket['com_vel'] > 0.97*required_vel) and statePacket['left_leg'] > 1e-2 and statePacket['right_leg'] > 1e-2:
            phase = 2
        if statePacket['com_vel'] < 0 and statePacket['left_leg'] < 1e-2 and statePacket["right_leg"] < 1e-2 and t>self.jump_time:
            phase = 3
        return phase
    
    def SetPreflightOutput(self, context, output):
        phase = self.DeterminePhase()
        if phase == 1:
            output.set_value(self.x_traj)
    
    def SetFlightOutput(self, context, output):
        phase = self.DeterminePhase()
        np.array([com[0]+0.3, 0 , np.clip(com[-1]-0.6,0, np.inf)])
        if phase == 2:
            statePacket = fetchStates(self.plant_context, self.plant)
            com = statePacket['com_pos']
            leftleg = statePacket['left_leg']
            rightleg = statePacket['right_leg']

            if leftleg[0] > rightleg[0]:
                output.set_value(BasicVector([com[0]+0.3, 0 , np.clip(com[-1]-0.6,0, np.inf), com[0]-0.3, 0 , np.clip(com[-1]-0.6,0, np.inf)]))
            else:
                output.set_value(BasicVector([com[0]-0.3, 0 , np.clip(com[-1]-0.6,0, np.inf), com[0]+0.3, 0 , np.clip(com[-1]-0.6,0, np.inf)]))

    def SetLandingOutput(self, context, output):
        phase = self.DeterminePhase()
        if phase == 3:
            statePacket = fetchStates(self.plant_context, self.plant)
            com = statePacket['com_pos']
            output.set_value(BasicVector([com[0],0,float(self.z_des)]))

    def PhaseOutput(self, output):
        phase = self.DeterminePhase()
        output.set_value(BasicVector([phase]))

    def get_preflight_port_index(self): return self.get_input_port(self.preflightoutput_trajectory_port_index)
    def get_aerial_trajectory_port_index(self): return self.get_input_port(self.flightoutput_trajectory_port_index)
    def get_landing_trajectory_port_index(self): return self.get_input_port(self.landingoutput_trajectory_port_index)
    def get_phase_switch_output_port_index(self): return self.get_output_port(self.phaseoutput_port_index)
    