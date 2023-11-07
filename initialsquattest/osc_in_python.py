import pydot
import numpy as np
from IPython.display import SVG, display

from pydrake.all import Simulator, DiagramBuilder, AddMultibodyPlantSceneGraph,\
                        Parser, RigidTransform, MeshcatVisualizer, MeshcatVisualizerParams, \
                        ConstantVectorSource, ConstantValueSource, PiecewisePolynomial,\
                        AbstractValue, HalfSpace, CoulombFriction
import footstep_planner
import osc
import importlib
importlib.reload(osc)
importlib.reload(footstep_planner)
from osc import OperationalSpaceWalkingController, OscGains

from pydrake.all import StartMeshcat

meshcat = StartMeshcat()

# Build the block diagram for the simulation
builder = DiagramBuilder()

# Add a planar walker to the simulation
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0000005)
X_WG = HalfSpace.MakePose(np.array([0,0, 1]), np.zeros(3,))
plant.RegisterCollisionGeometry(
    plant.world_body(), 
    X_WG, HalfSpace(), 
    "collision", 
    CoulombFriction(1.0, 1.0)
)
parser = Parser(plant)
parser.AddModels("planar_walker.urdf")
plant.WeldFrames(
    plant.world_frame(),
    plant.GetBodyByName("base").body_frame(),
    RigidTransform.Identity()
)
plant.Finalize()


# Build the controller diagram
Kp = np.diag([100, 0, 100])
Kd = np.diag([10, 0, 10])
W = np.diag([1, 0, 1])

Wcom = np.zeros((3,3))
Wcom[2,2] = 1

gains = OscGains(
        Kp, Kd, Wcom,
        Kp, Kd, W,
        np.eye(1), np.eye(1), np.eye(1),
        0.00001
    )


# TODO: Adjust target walking speed here
walking_speed = 0.3 # walking speed in m/s

osc = builder.AddSystem(OperationalSpaceWalkingController(gains))
planner = builder.AddSystem(footstep_planner.LipTrajPlanner())
speed_src = builder.AddSystem(ConstantVectorSource(np.array([walking_speed])))
base_traj_src = builder.AddSystem(
    ConstantValueSource(AbstractValue.Make(PiecewisePolynomial(np.zeros(1,))))
)

# Wire planner inputs 
builder.Connect(plant.get_state_output_port(), 
                planner.get_state_input_port())
builder.Connect(speed_src.get_output_port(), 
                planner.get_walking_speed_input_port())

# Wire OSC inputs
builder.Connect(plant.get_state_output_port(), 
                osc.get_state_input_port()) 
builder.Connect(planner.get_swing_foot_traj_output_port(), 
                osc.get_traj_input_port("swing_foot_traj"))
builder.Connect(planner.get_com_traj_output_port(), 
                osc.get_traj_input_port("com_traj"))
builder.Connect(base_traj_src.get_output_port(), 
                osc.get_traj_input_port("base_joint_traj"))

# Add the visualizer
vis_params = MeshcatVisualizerParams(publish_period=0.01)
MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params=vis_params)

# Wire OSC to plant
builder.Connect(osc.get_output_port(), 
                plant.get_actuation_input_port())

#simulate
diagram = builder.Build()
# display(SVG(pydot.graph_from_dot_data(
#     diagram.GetGraphvizString(max_depth=2))[0].create_svg()))

# NOTE - if you make changes, you should re-run the cell above this one

sim_time = 0.03
simulator = Simulator(diagram)
simulator.Initialize()
simulator.set_target_realtime_rate(1)

# Set the robot state
plant_context = diagram.GetMutableSubsystemContext(
    plant, simulator.get_mutable_context())
q = np.zeros((plant.num_positions(),))
q[1] = 0.8
theta = -np.arccos(q[1])
q[3] = theta
q[4] = -2 * theta
q[5] = theta
q[6] = -2 * theta
plant.SetPositions(plant_context, q)

# Simulate the robot
simulator.AdvanceTo(sim_time)