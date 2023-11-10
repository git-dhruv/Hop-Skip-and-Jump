import pydot
import numpy as np
from IPython.display import SVG, display

from pydrake.all import Simulator, DiagramBuilder, AddMultibodyPlantSceneGraph,\
                        Parser, RigidTransform, MeshcatVisualizer, MeshcatVisualizerParams, \
                        ConstantVectorSource, ConstantValueSource, PiecewisePolynomial,\
                        AbstractValue, HalfSpace, CoulombFriction
from pydrake.all import StartMeshcat 

from pydrake.visualization import MeshcatPoseSliders
meshcat = StartMeshcat()
# Build the block diagram for the simulation
builder = DiagramBuilder()
# Add a planar walker to the simulation
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.005)
X_WG = HalfSpace.MakePose(np.array([0,0, 1]), np.zeros(3,))
plant.RegisterCollisionGeometry(
    plant.world_body(), 
    X_WG, HalfSpace(), 
    "collision", 
    CoulombFriction(1.0, 1.0))
parser = Parser(plant)
parser.AddModels("/home/dhruv/final/models/fivelink2.urdf")
plant.WeldFrames(
    plant.world_frame(),
    plant.GetBodyByName("base").body_frame(),
    RigidTransform.Identity()
)


# meshcat.DeleteAllButtonsAndSliders()
teleop = builder.AddSystem(MeshcatPoseSliders(meshcat))
plant.Finalize()

# Add the visualizer
vis_params = MeshcatVisualizerParams(publish_period=0.01)
MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params=vis_params)
# builder.Connect(teleop.get_output_port(0), plant.get_input_port(0))
torques = builder.AddSystem(ConstantVectorSource(np.zeros(plant.num_actuators())))
builder.Connect(torques.get_output_port(), plant.get_actuation_input_port())

#simulate
diagram = builder.Build()

sim_time = 5
simulator = Simulator(diagram)
simulator.Initialize()
simulator.set_target_realtime_rate(1)

# Set the robot state
plant_context = diagram.GetMutableSubsystemContext(
    plant, simulator.get_mutable_context())
q = np.zeros((plant.num_positions(),))
q[0] = 1
q[1] = 1
q[2] = 0

q[5] = .00000001
plant.SetPositions(plant_context, q)

# Simulate the robot
simulator.AdvanceTo(sim_time)