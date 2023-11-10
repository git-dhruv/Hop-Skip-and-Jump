#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Dhruv Parikh, Anirudh Kailaje
@date: 10th Nov 2023
@file: preflight.py
@brief: Preflight Direct Collocation Planner. I dont know how will this turn out to be!
"""

#####  Pydrake Files  #####
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import Context, LeafSystem, BasicVector, DiscreteValues, EventStatus
from pydrake.trajectories import Trajectory, PiecewisePolynomial
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.multibody.all import JacobianWrtVariable


import numpy as np
import sys
import logging



# Module constants
# CONSTANT_1 = "value"

# Module "global" variables
# global_var = None

class dircol(LeafSystem):
    """
    Direct Collocation for flight phase

    @output ports: 1 ;size = 4; Data = [com_x, com_y, com_z, torso angle]
    @input ports: 3;
        x: State Input
        com_des: Center of Mass (size = 1) - read as desired height
        time_des: Final time of stance (size = 1) - in seconds
    """
    
    def __init__(self):
        LeafSystem.__init__(self)

        # Make internal dynamics model to get the COM and stuff #
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

        # Input Ports #
        self.robot_state_input_port_index = self.DeclareVectorInputPort("x", self.plant.num_positions() + self.plant.num_velocities()).get_index()
        self.com_des_input_port_index = self.DeclareVectorInputPort("com_des", BasicVector(1)).get_index()
        self.time_des_input_port_index = self.DeclareVectorInputPort("time_des", BasicVector(1)).get_index()


        # Output Ports #
        self.com_trajectory_output_port_index = self.DeclareAbstractOutputPort("comtraj", lambda: AbstractValue.Make(BasicVector(self.plant.num_positions()+self.plant.num_velocities())),self.comtrajCB).get_index()

    def comtrajCB(self):
        # I just made the ugliest name to give to a function
        pass

    
    ## Port Accessors ##
    def get_state_input_port(self):
        return self.get_input_port(self.robot_state_input_port_index)
    def get_com_input_port(self):
        return self.get_input_port(self.com_des_input_port_index)
    def get_time_input_port(self):
        return self.get_input_port(self.time_des_input_port_index)
    def get_com_output_port(self):
        return self.get_output_port(self.com_trajectory_output_port_index)