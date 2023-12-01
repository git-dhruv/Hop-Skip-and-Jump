import matplotlib.pyplot as plt
import numpy as np
import importlib
from pydrake.all import (DiagramBuilder, Simulator, FindResourceOrThrow, MultibodyPlant, PiecewisePolynomial, SceneGraph,
    Parser, JointActuatorIndex, MathematicalProgram, Solve)
from scipy.constants import g
# import kinematic_constraints
import dynamics_constraints
from pydrake.math import RigidTransform
# importlib.reload(kinematic_constraints)
importlib.reload(dynamics_constraints)
# from kinematic_constraints import (
#   AddFinalLandingPositionConstraint
# )
from dynamics_constraints import (
  AddCollocationConstraints,
  EvaluateDynamics
)

def find_throwing_trajectory(N, initial_state, jumpheight, tf, jumpheight_tol=5e-2):
  '''
  Parameters:
    N - number of knot points
    initial_state - starting configuration
    distance - target distance to throw the ball

  '''

  builder = DiagramBuilder()
  plant = builder.AddSystem(MultibodyPlant(0.0))
  file_name = "/home/anirudhkailaje/Documents/01_UPenn/02_MEAM5170/03_FinalProject/src/planar_walker.urdf"
  file_name = "/home/dhruv/Hop-Skip-and-Jump/models/planar_walker.urdf"

  Parser(plant=plant).AddModels(file_name)
  plant.WeldFrames(plant.world_frame(),plant.GetBodyByName("base").body_frame(),RigidTransform.Identity())
  plant.Finalize()
  robot = plant.ToAutoDiffXd()

  plant_context = plant.CreateDefaultContext()
  context = robot.CreateDefaultContext()

  # Dimensions specific to the robot
  n_q = robot.num_positions()
  n_v = robot.num_velocities()
  n_x = n_q + n_v
  n_u = robot.num_actuators()
  
  # Store the actuator limits here
  effort_limits = np.array([robot.get_joint_actuator(JointActuatorIndex(act_idx)).effort_limit() for act_idx in range(n_u)])
  """Joint limits specified in the order [planar_x: m, planar_z:m, roty(torso_angle): radians, left_hip:radians, right_hip: radians, left_knee: radians, right_knee: radians]"""
  joint_limit_lower = np.array([-1, 0, -0.9, -1.39, -1.39, 0, 0])
  joint_limit_upper = np.array([1, 5, 0.6, 0.78, 0.78, 2.5, 2.5])
  # vel_limits = 15 * np.ones(n_v)

  # Create the mathematical program
  prog = MathematicalProgram()
  #These are going to be the decision variables for the mathematical program
  x = np.zeros((N, n_x), dtype="object"); u = np.zeros((N, n_u), dtype="object"); lambda_c = np.zeros((N, 6), dtype="object")
  for i in range(N):
    x[i] = prog.NewContinuousVariables(n_x, "x_" + str(i))
    u[i] = prog.NewContinuousVariables(n_u, "u_" + str(i))
    lambda_c[i] = prog.NewContinuousVariables(6, f"lambda_c_{i}")

  # t_jump = prog.NewContinuousVariables(1, "t_jump") #Decision variable for the take-off maneuvour duration
  # prog.AddConstraint(t_jump[0]<=tf)

  t0 = 0.0; timesteps = np.linspace(t0, tf, N)
  x0 = x[0]; xf = x[-1]

  # Add the kinematic constraints (initial state, final state)
  # TODO: Add constraints on the initial state
  initial_state_constraint = prog.AddLinearEqualityConstraint(x0, initial_state)
  initial_state_constraint.evaluator().set_description("Initial State Constraint")

  required_velocity = (2*g*jumpheight)**0.5
  velocity_tol = np.array([(2*g*jumpheight_tol)**0.5])
  # prog.AddLinearEqualityConstraint(xf[n_q+1], final_configuration)
  # final_zvel_constraint = prog.AddBoundingBoxConstraint(required_velocity-velocity_tol, required_velocity+velocity_tol, xf[n_q+1])
  # final_zvel_constraint.evaluator().set_description("Final Z velocity constraint")
  # robot.SetPositionsAndVelocities(context, xf)
  # com_pos = robot.CalcCenterOfMassPositionInWorld(context).ravel()  

  """TODO: Add Bounding Box constraint on Final Angular Momentum"""
  


  # Add the collocation aka dynamics constraints
  AddCollocationConstraints(prog, robot, context, N, x, u, lambda_c, timesteps)

  # TODO: Add the cost function here
  for i in range(N-1):
       prog.AddQuadraticCost(0.5*(timesteps[i+1]-timesteps[i])*((u[i].T@u[i])+(u[i+1].T@u[i+1])))
       prog.AddQuadraticCost(0.5*(x[i][2]-jumpheight)**2)
       
  # prog.AddQuadraticCost(0.5*(xf[n_q+1]-required_velocity)**2) #Cost on error in z-velocity

  # TODO: Add bounding box constraints on the inputs and qdot 
  joint_limit_constraints = []

  mu = 1
  A_fric = np.array([[1, 0, -mu, 0, 0, 0], # Friction constraint for left foot, positive x-direction
                  [-1, 0, -mu, 0, 0, 0],   # Friction constraint for left foot, negative x-direction
                  [0, 0, 0, 1, 0, -mu],    # Friction constraint for right foot, positive x-direction
                  [0, 0, 0, -1, 0, -mu]])  # Friction constraint for right foot, negative x-direction
  for i in range(N):
      # joint_limit_constraints.append(prog.AddBoundingBoxConstraint(joint_limit_lower, joint_limit_upper, x[i, :n_q]))
      # joint_limit_constraints[i].evaluator().set_description(f"Joint limit @ knot point {i}")
      # prog.AddBoundingBoxConstraint(-vel_limits, vel_limits, x[i, n_q:n_q+n_v])
      prog.AddBoundingBoxConstraint(-effort_limits, effort_limits, u[i])
      # The constraint is applied to the 6x1 lambda_c vector
      prog.AddLinearConstraint(A_fric @ lambda_c[i].reshape(6, 1), -np.inf * np.ones((4, 1)), np.zeros((4, 1)))
      prog.AddLinearEqualityConstraint(lambda_c[i][1] == 0)
      prog.AddLinearEqualityConstraint(lambda_c[i][4] == 0)


  



  # TODO: give the solver an initial guess for x and u using prog.SetInitialGuess(var, value) - Tricky Problem to solve
  x_guess = np.load("/home/anirudhkailaje/Documents/01_UPenn/02_MEAM5170/03_FinalProject/src/traj.npy")
  x_init = x_guess[:, ::(x_guess.shape[1])//N][:,:N].T


  u_init = np.random.uniform(low = -effort_limits, high = effort_limits, size=(N, n_u))

  lambda_init = np.zeros((N, 6))
  
  prog.SetInitialGuess(x, x_init)
  prog.SetInitialGuess(u, u_init)
  prog.SetInitialGuess(lambda_c, lambda_init)

  # Set up solver
  result = Solve(prog)
  
  x_sol = result.GetSolution(x)
  u_sol = result.GetSolution(u)
  lambda_sol = result.GetSolution(lambda_c)
  # t_land_sol = result.GetSolution(t_land)

  print('optimal cost: ', result.get_optimal_cost())
  print('x_sol: ', x_sol)
  print('u_sol: ', u_sol)
  # print('t_land: ', t_land_sol)

  print(result.get_solution_result())

  # infeasible_constraints = result.GetInfeasibleConstraints(prog)
  # for c in infeasible_constraints:
  #     print(f"infeasible constraint: {c}")

  # Reconstruct the trajectory
  xdot_sol = np.zeros(x_sol.shape)
  for i in range(N):
    xdot_sol[i] = EvaluateDynamics(plant, plant_context, x_sol[i], u_sol[i], lambda_sol[i])[0]
  
  x_traj = PiecewisePolynomial.CubicHermite(timesteps, x_sol.T, xdot_sol.T)
  u_traj = PiecewisePolynomial.ZeroOrderHold(timesteps, u_sol.T)

  return x_traj, u_traj, prog, prog.GetInitialGuess(x), prog.GetInitialGuess(u)
  
if __name__ == '__main__':
  N = 50
  initial_state = np.zeros(14)
  # final_configuration = np.array([np.pi, 0])
  tf = 3.0
  x_traj, u_traj, prog,  _, _ = find_throwing_trajectory(N, initial_state, jumpheight = 1.5, tf=3, jumpheight_tol=5e-2)
