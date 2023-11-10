"""
@author: Dhruv Parikh, Anirudh Kailaje
@file: planner.py
@brief: Center of mass planner. This is a minimal example of a higher level controller class.

@TODO: Use Polynomials to find traj derivatives
"""

from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import Context, LeafSystem, BasicVector, DiscreteValues, EventStatus
from pydrake.trajectories import Trajectory, PiecewisePolynomial
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.multibody.all import JacobianWrtVariable


class COMPlanner(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)

        #Make internal dynamics model to get the COM and stuff#
        self.plant = MultibodyPlant(0.0)
        self.parser = Parser(self.plant)
        self.parser.AddModels("models/planar_walker.urdf")
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
        # Calculates current center of mass

        #Get the current state from the input port
        state = self.EvalVectorInput(context, self.robot_state_input_port_index).value()
        #Simulate our robot to the current state
        self.plant.SetPositionsAndVelocities(self.plant_context, state)
        #Extract the Center of Mass of our robot
        com_pos = self.plant.CalcCenterOfMassPositionInWorld(self.plant_context).ravel()
        return com_pos
    
    def calcDesiredZddot(self, context: Context, output):
        #Get the current desired state
        z_des = self.EvalVectorInput(context, self.walking_speed_input_port_index).value()
        #Calculate z_dd_des which I wont use anywhere
        z_dd_des = -(z_des - self.calcCOM(context)[-1])

        ## As of now, we are just routing z_des 1 to 1
        output.set_value(BasicVector([0.0,0,float(z_des)]))
        # print(np.sin(context.get_time()*10))

    ## These 2 functions are not tested and are not working as well ##
    def CalcYdot(self, state):
        if state is None:
            state = self.EvalVectorInput(self.plant_context, self.robot_state_input_port_index).value()
        self.plant.SetPositionsAndVelocities(self.plant_context, state)
        return (self.CalcJ() @ self.plant.GetVelocities(self.context)).ravel()

    def CalcJ(self):
        return self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
            self.context, JacobianWrtVariable.kV, self.plant.world_frame(), self.plant.world_frame())
    
    ### These just act as a function that return the ports ###
    def get_com_traj_output_port(self):
        return self.get_output_port(self.com_traj_output_port_index)
    def get_com_zdes_input_port(self):
        return self.get_input_port(self.walking_speed_input_port_index)
    def get_com_state_input_port(self):
        return self.get_input_port(self.robot_state_input_port_index)