"""
Main Simulation File, teh Multibody plant is defined is here. Planners are invoked and simultion is conduct.
"""
import pydot
import numpy as np
from IPython.display import SVG, display

from pydrake.all import Simulator, DiagramBuilder, AddMultibodyPlantSceneGraph,\
                        Parser, RigidTransform, MeshcatVisualizer, MeshcatVisualizerParams, \
                        ConstantVectorSource, ConstantValueSource, PiecewisePolynomial,\
                        AbstractValue, HalfSpace, CoulombFriction
from preflight import dircol
from flight import flight
from landing import landing
from phase_switch import PhaseSwitch

from osc_modified import OperationalSpaceWalkingController

from pydrake.all import StartMeshcat, BasicVector, LogVectorOutput
import matplotlib.pyplot as plt

#Start the meshcat server
meshcat = StartMeshcat(); builder = DiagramBuilder()

#### Designing our world ####
# Add a planar walker to the simulation
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.00005)
#Half space means a plane -> Ground Plane in particular
X_WG = HalfSpace.MakePose(np.array([0,0, 1]), np.zeros(3,))
plant.RegisterCollisionGeometry(plant.world_body(), X_WG, HalfSpace(), 
    "collision", CoulombFriction(1.0, 1.0))

#Make the plant
parser = Parser(plant)
parser.AddModels("models/planar_walker.urdf")
plant.WeldFrames(plant.world_frame(),
    plant.GetBodyByName("base").body_frame(),
    RigidTransform.Identity())
plant.Finalize()


#### Designing the controller ####
zdes = 0.8 #desired Z height in meters
z_height_desired = builder.AddSystem(ConstantVectorSource(np.array([zdes])))
jump_time = builder.AddSystem(ConstantVectorSource(np.array([2.5])))
foot_clearance = builder.AddSystem(ConstantVectorSource(np.array([0.1])))
base_traj_src = builder.AddSystem(ConstantVectorSource(np.array([1])))
post_flight_traj_src = builder.AddSystem(ConstantVectorSource(np.zeros(3,))) ##np.zeros(3,)

preflight_planner = builder.AddSystem(dircol())
flight_planner = builder.AddSystem(flight())
landing_planner = builder.AddSystem(landing())
phase_switch = builder.AddSystem(PhaseSwitch())
osc = builder.AddSystem(OperationalSpaceWalkingController())

## Logger ##
logger = LogVectorOutput(osc.GetOutputPort("metrics"),builder)


#### Wiring ####

#COM wiring
builder.Connect(z_height_desired.get_output_port(), preflight_planner.get_com_input_port())
builder.Connect(jump_time.get_output_port(), preflight_planner.get_time_input_port())
builder.Connect(plant.get_state_output_port(), preflight_planner.get_state_input_port())
builder.Connect(foot_clearance.get_output_port(), flight_planner.get_ftclearance_input_port())
builder.Connect(plant.get_state_output_port(), flight_planner.get_state_input_port())
builder.Connect(plant.get_state_output_port(), landing_planner.get_state_input_port())
#Phase Switch wiring
builder.Connect(preflight_planner.get_output_port(), phase_switch.get_preflight_port_index())
builder.Connect(flight_planner.get_output_port(), phase_switch.get_aerial_trajectory_port_index())
builder.Connect(landing_planner.get_output_port(), phase_switch.get_landing_trajectory_port_index())
# OSC wiring
builder.Connect(base_traj_src.get_output_port(), landing_planner.get_torso_input_port())
builder.Connect(post_flight_traj_src.get_output_port(), landing_planner.get_com_input_port())

builder.Connect(phase_switch.get_phase_switch_output_port_index(), osc.get_traj_input_port("com_traj"))
builder.Connect(plant.get_state_output_port(), osc.get_state_input_port()) 
builder.Connect(osc.torque_output_port, plant.get_actuation_input_port())

# Add the visualizer
vis_params = MeshcatVisualizerParams(publish_period=0.0005)
MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params=vis_params)

#simulate
diagram = builder.Build()
graph = (pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=2))[0].create_svg())

with open('graph.svg', 'wb') as f:
    f.write(graph)


################

sim_time = 1
simulator = Simulator(diagram)
simulator.Initialize()
simulator.set_target_realtime_rate(0.1)

# Initial State
plant_context = diagram.GetMutableSubsystemContext(
    plant, simulator.get_mutable_context())
q = np.zeros((plant.num_positions(),))
q[0] = 0
q[1] = 0.8*1.2
theta = -np.arccos(q[1])
q[3] = theta/2
q[4] = -2 * theta
q[5] = theta
q[6] = -2 * theta/2
plant.SetPositions(plant_context, q)

# Simulate the robot
simulator.AdvanceTo(sim_time)

## Logs and Plots ##
log = logger.FindLog(simulator.get_mutable_context()) #xyz vxvyvz
t = log.sample_times()
x = log.data()[2]   
xdot = log.data()[-1]

plt.figure()
plt.plot(t, x)
plt.figure()
plt.plot(t, xdot)
plt.show()