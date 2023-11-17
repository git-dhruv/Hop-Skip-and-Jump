"""
@author: Dhruv Parikh, Anirudh Kailaje
@date: 7th Nov 2023
@file: bipedSquat.py
@brief: Biped Does a Squat!
Mathematically: We track the Center of Mass trajectory on z direction with constraints of both foot on ground
"""
import pydot
import numpy as np

from pydrake.all import Simulator, DiagramBuilder, AddMultibodyPlantSceneGraph,\
                        Parser, RigidTransform, MeshcatVisualizer, MeshcatVisualizerParams, \
                        ConstantVectorSource, ConstantValueSource, PiecewisePolynomial,\
                        AbstractValue, HalfSpace, CoulombFriction
import planner
from osc import OSC

from pydrake.all import StartMeshcat, BasicVector, LogVectorOutput
import matplotlib.pyplot as plt, datetime

time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

#Start the meshcat server
meshcat = StartMeshcat()
builder = DiagramBuilder()

#### Designing our world ####
# Add a planar walker to the simulation
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0005)
#Half space means a plane -> Ground Plane in particular
X_WG = HalfSpace.MakePose(np.array([0,0, 1]), np.zeros(3,))
plant.RegisterCollisionGeometry(
    plant.world_body(), 
    X_WG, HalfSpace(), 
    "collision", CoulombFriction(1.0, 1.0))

#Make the plant
urdf = r"/home/anirudhkailaje/Documents/01_UPenn/02_MEAM5170/03_FinalProject/src/planar_walker.urdf"
parser = Parser(plant)
parser.AddModels(urdf)
plant.WeldFrames(
    plant.world_frame(),
    plant.GetBodyByName("base").body_frame(),
    RigidTransform.Identity()
)
plant.Finalize()


#### Designing the controller ####
"""
Here,
zdes: desired squat Z height in meters
z_height_desired: A constant source which outputs the desired Z height, given to the com_planner
com_planner: A controller which tracks the desired Z height for the center of mass
base_traj_src: A constant source which outputs a constant vector (No movement in the Y direction for the robot), given to the osc
osc: The operational space controller which tracks the desired Z height and the base_traj_src
"""
zdes = 0.8 #desired Z height in meters
z_height_desired = builder.AddSystem(ConstantVectorSource(np.array([zdes])))
com_planner = builder.AddSystem(planner.COMPlanner())
base_traj_src = builder.AddSystem(ConstantValueSource(AbstractValue.Make(BasicVector(np.zeros(1,)))))
osc = builder.AddSystem(OSC(urdf))

#### Wiring ####
#COM wiring
builder.Connect(z_height_desired.get_output_port(), com_planner.get_com_zdes_input_port())
builder.Connect(plant.get_state_output_port(), com_planner.get_com_state_input_port())
# OSC wiring
builder.Connect(com_planner.get_com_traj_output_port(), osc.get_traj_input_port("COM"))
builder.Connect(base_traj_src.get_output_port(), osc.get_traj_input_port("torso"))
builder.Connect(plant.get_state_output_port(), osc.get_state_input_port()) 
builder.Connect(osc.torque_output_port, plant.get_actuation_input_port())

# Add the visualizer
vis_params = MeshcatVisualizerParams(publish_period=0.0005)
MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params=vis_params)
#simulate
diagram = builder.Build()
graph = (pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=2))[0].create_svg())

with open("OSCgraph.svg", "wb") as f:
    f.write(graph)
################

sim_time = 20
simulator = Simulator(diagram)
simulator.Initialize(); simulator.set_target_realtime_rate(1)

# State Description: q = [x, z, planar_roty, left_hip, left_knee, right_hip, right_knee]
plant_context = diagram.GetMutableSubsystemContext(plant, simulator.get_mutable_context())

q = np.zeros((plant.num_positions(),))
q[0] = 0; q[1] = 1/2
theta = -np.arccos(q[1])
q[3] = theta; q[4] = -2 * theta
q[5] = theta;   q[6] = -2 * theta
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
plt.savefig(f"./logs/{time}plots.png")
plt.show()

