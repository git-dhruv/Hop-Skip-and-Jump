from utils import *
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import Context, LeafSystem, BasicVector, DiscreteValues, EventStatus
from pydrake.trajectories import Trajectory, PiecewisePolynomial
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.multibody.all import JacobianWrtVariable
from copy import deepcopy


class PhaseSwitch(LeafSystem):
    def __init__(self, height, jump_time, x_traj, z_des, model):
        LeafSystem.__init__(self)
        self.plant = MultibodyPlant(0.0)
        self.parser = Parser(self.plant)
        self.parser.AddModels(model)
        self.plant.WeldFrames(self.plant.world_frame(), self.plant.GetBodyByName("base").body_frame(), RigidTransform.Identity())
        self.plant.Finalize()
        self.req_height = height
        self.jump_time = jump_time
        self.x_traj = x_traj
        self.z_des = z_des
        self.plant_context = self.plant.CreateDefaultContext()
        self.phase = None

        self.preflightoutput_trajectory_port_index = self.DeclareAbstractOutputPort("OSCPreflight_Trajectory", lambda: AbstractValue.Make(PiecewisePolynomial()), self.SetPreflightOutput).get_index()
        self.flightoutput_trajectory_port_index = self.DeclareAbstractOutputPort("OSCLFight_Trajectory", lambda: AbstractValue.Make(BasicVector(6)), self.SetFlightOutput).get_index()
        self.landingoutput_trajectory_port_index = self.DeclareAbstractOutputPort("OSCLanding_Trajectory", lambda: AbstractValue.Make(BasicVector(3)), self.SetLandingOutput).get_index()
        self.phaseoutput_port_index = self.DeclareAbstractOutputPort("Phase", lambda: AbstractValue.Make(BasicVector(1)), self.PhaseOutput).get_index()
        self.robot_state_input_port_index = self.DeclareVectorInputPort("x", self.plant.num_positions() + self.plant.num_velocities()).get_index()

    def DeterminePhase(self, context):
        required_vel = (2*9.18*self.req_height)**0.5
        stateVector = self.EvalVectorInput(context, self.robot_state_input_port_index).get_value()
        # stateVector = self.generateNoiseForStates(stateVector)
        self.plant.SetPositionsAndVelocities(self.plant_context, stateVector)
        statePacket = fetchStates(self.plant_context, self.plant)
        t  = context.get_time()
        phase = 0
        if t<=0.9*self.jump_time: # and self.phase is None:
            phase = 1
            self.phase = phase

        elif (t>0.9*self.jump_time or statePacket['com_vel'][-1] > 0.9*required_vel) and statePacket['left_leg'][-1] > 0 and statePacket['right_leg'][-1] > 0:
            phase = 2
            self.phase = phase
        if  (statePacket['left_leg'][-1] < 1e-2 or statePacket["right_leg"][-1] < 1e-2) and t>1.1*self.jump_time and statePacket['com_vel'][-1] < 0:
            phase = 3
            self.phase = phase
        return self.phase
    
    def SetPreflightOutput(self, x, output):
        phase = self.DeterminePhase(x)
        if phase == 1:
            output.set_value(self.x_traj)
    
    def SetFlightOutput(self, x, output):
        phase = self.DeterminePhase(x)
        if phase == 2:
            statePacket = fetchStates(self.plant_context, self.plant)
            com = statePacket['com_pos']
            leftleg = statePacket['left_leg']
            rightleg = statePacket['right_leg']

            XOffset = 0.2
            ZOffset = 0.97*0.85
            if leftleg[0] > rightleg[0]:
                output.set_value(BasicVector([com[0]+XOffset  , 0 , np.clip(com[-1]-ZOffset,0, np.inf), com[0]-XOffset, 0 , np.clip(com[-1]-ZOffset,0, np.inf)]))
            else:
                output.set_value(BasicVector([com[0]-XOffset, 0 , np.clip(com[-1]-ZOffset,0, np.inf), com[0]+XOffset, 0 , np.clip(com[-1]-ZOffset,0, np.inf)]))

    def SetLandingOutput(self, x, output):
        phase = self.DeterminePhase(x)
        if phase == 3:
            statePacket = fetchStates(self.plant_context, self.plant)
            com = statePacket['com_pos']
            swingFootCenterPos = (statePacket['left_leg'][0] + statePacket['right_leg'][0])/2
            output.set_value(BasicVector([swingFootCenterPos,0,float(self.z_des)]))

    def PhaseOutput(self, x, output):
        phase = self.DeterminePhase(x)
        output.set_value(BasicVector([phase]))

    def generateNoiseForStates(self, x):
        RAD2DEG = np.pi/180
        std = np.array([  (.01)/3, (0.01)/3, RAD2DEG*5/3, RAD2DEG*5/3, RAD2DEG*5/3, RAD2DEG*5/3, RAD2DEG*5/3,
                  (.1)/3, (0.1)/3, RAD2DEG*50/3, RAD2DEG*50/3, RAD2DEG*50/3, RAD2DEG*50/3  ,RAD2DEG*50/3])/10
        state = np.random.normal(x, std)
        joint_limit_lower = np.array([-1e-1, 0, -1e-1, -3.14, 0, -3.14, 0])
        joint_limit_upper = np.array([1e-1, 1.2, 1e-1, 3.14, 3.14, 3.14, 3.14])  
        for i in range(3,7):
            state[i] = np.clip(state[i], joint_limit_lower[i], joint_limit_upper[i])
        return state


    def get_preflight_port_index(self): return self.get_output_port(self.preflightoutput_trajectory_port_index)
    def get_aerial_trajectory_port_index(self): return self.get_output_port(self.flightoutput_trajectory_port_index)
    def get_landing_trajectory_port_index(self): return self.get_output_port(self.landingoutput_trajectory_port_index)
    def get_phase_switch_output_port_index(self): return self.get_output_port(self.phaseoutput_port_index)
    def get_state_input_port(self): return self.get_input_port(self.robot_state_input_port_index)
    