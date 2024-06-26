import matplotlib.pyplot as plt
import numpy as np
import importlib
from pydrake.all import (DiagramBuilder, Simulator, FindResourceOrThrow, MultibodyPlant, PiecewisePolynomial, SceneGraph,
    Parser, JointActuatorIndex, MathematicalProgram, Solve, SolverType)
from scipy.constants import g
from pydrake.math import RigidTransform
from dynamics_constraints_dev_explicit import (
  AddCollocationConstraints,
  EvaluateDynamics
)

from dataclasses import dataclass
from pydrake.multibody.tree import Frame

@dataclass
class PointOnFrame:
    frame: Frame
    pt: np.ndarray

def get_foot_pos(context, plant):
    contact_points = {
    0: PointOnFrame(
        plant.GetBodyByName("left_lower_leg").body_frame(),
        np.array([0, 0, -0.5])
    ),
    1: PointOnFrame(
        plant.GetBodyByName("right_lower_leg").body_frame(),
        np.array([0, 0, -0.5])
    )
    }
    ft = np.zeros((3,2))
    i =0
    for fsm in [0,1]:
        pt_to_track = contact_points[fsm]
        ft[:,i] = plant.CalcPointsPositions(context, pt_to_track.frame,
                                        pt_to_track.pt, plant.world_frame()).ravel()
        i+=1
    return ft    


def find_throwing_trajectory(N, initial_state, jumpheight, tf, jumpheight_tol=5e-2):
  '''
  Parameters:
    N - number of knot points
    initial_state - starting configuration
    distance - target distance to throw the ball

  '''

  builder = DiagramBuilder()
  plant = builder.AddSystem(MultibodyPlant(0.0))
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
  joint_limit_lower = np.array([-1e-1, 0, -1e-1, -3.14, 0, -3.14, 0])
  joint_limit_upper = np.array([1e-1, 1, 1e-1, 3.14, 3.14, 3.14, 3.14])
  # vel_limits = 15 * np.ones(n_v)

  # Create the mathematical program and decision variables
  prog = MathematicalProgram()
  x = np.array([prog.NewContinuousVariables(n_x, f"x_{i}") for i in  range(N)])
  u = np.array([prog.NewContinuousVariables(n_u, f"u_{i}") for i in  range(N)])
  lambda_c_col = np.array([prog.NewContinuousVariables(6, f"lambda_c_col{i}") for i in  range(N-1)])
  lambda_c = np.array([prog.NewContinuousVariables(6, f"lambda_c_{i}") for i in range(N)])
  

  # t_jump = prog.NewContinuousVariables(1, "t_jump") #Decision variable for the take-off maneuvour duration
  # prog.AddConstraint(t_jump[0]<=tf)

  t0 = 0.0; timesteps = np.linspace(t0, tf, N)
  x0 = x[0]; xf = x[-1]

  # Add the kinematic constraints (initial state, final state)
  # TODO: Add constraints on the initial state
  initial_state_constraint = prog.AddLinearEqualityConstraint(x0, initial_state)
  initial_state_constraint.evaluator().set_description("Initial State Constraint")

  # prog.AddLinearEqualityConstraint(xf, final_state)

  """Jump velocity constraints"""
  required_velocity = (2*g*jumpheight)**0.5
  velocity_tol = np.array([(2*g*jumpheight_tol)**0.5])
  velocity_tol = 1e-2
  # prog.AddLinearEqualityConstraint(xf[n_q+1], final_configuration)
  final_zvel_constraint = prog.AddBoundingBoxConstraint(required_velocity-velocity_tol, required_velocity+velocity_tol, xf[n_q+1])
  final_zvel_constraint.evaluator().set_description("Final Z velocity constraint")

  """TODO: Add Bounding Box constraint on Final Angular Momentum"""

  # Add the collocation aka dynamics constraints
  AddCollocationConstraints(prog, robot, context, N, x, u, lambda_c, lambda_c_col, timesteps)

  # TODO: Add the cost function here
  for i in range(N-1):
       prog.AddQuadraticCost(0.5*(u[i] - u[i+1]).T @ (u[i] - u[i+1]).T)
       prog.AddLinearConstraint(x[i][1], 0.6, 1.25)
       prog.AddLinearConstraint(x[i][0], -1e-2, 1e-2)
       prog.AddLinearConstraint(x[i][2], -1e-4, 1e-4)
       
       if i>=N-3:
        prog.AddLinearConstraint(x[i+1][n_q+1] - x[i][n_q+1] , -9, 30)
       else:
        prog.AddLinearConstraint(x[i+1][n_q+1] - x[i][n_q+1], -20, 9)
        prog.AddLinearConstraint(x[i][n_q] - x[i+1][n_q], -.01, .01)        

  

  # TODO: Add bounding box constraints on the inputs and qdot 

  mu = 1
  A_fric = np.array([[1, 0, -mu, 0, 0, 0], # Friction constraint for left foot, positive x-direction
                  [-1, 0, -mu, 0, 0, 0],   # Friction constraint for left foot, negative x-direction
                  [0, 0, 0, 1, 0, -mu],    # Friction constraint for right foot, positive x-direction
                  [0, 0, 0, -1, 0, -mu]])  # Friction constraint for right foot, negative x-direction
  for i in range(N):
      prog.AddBoundingBoxConstraint(joint_limit_lower, joint_limit_upper, x[i, :n_q])
      # joint_limit_constraints[i].evaluator().set_description(f"Joint limit @ knot point {i}")
      # prog.AddBoundingBoxConstraint(-vel_limits, vel_limits, x[i, n_q:n_q+n_v])
      prog.AddBoundingBoxConstraint(-effort_limits, effort_limits, u[i])
      # The constraint is applied to the 6x1 lambda_c vector
      prog.AddLinearConstraint(A_fric @ lambda_c[i].reshape(6, 1), -np.inf * np.ones((4, 1)), np.zeros((4, 1)))
      prog.AddLinearEqualityConstraint(lambda_c[i][1] == 0)
      prog.AddLinearEqualityConstraint(lambda_c[i][4] == 0)
  for i in range(N-1):
      prog.AddLinearConstraint(A_fric @ lambda_c_col[i].reshape(6, 1), -np.inf * np.ones((4, 1)), np.zeros((4, 1)))
      prog.AddLinearEqualityConstraint(lambda_c_col[i][1] == 0)
      prog.AddLinearEqualityConstraint(lambda_c_col[i][4] == 0)


  



  # TODO: give the solver an initial guess for x and u using prog.SetInitialGuess(var, value) - Tricky Problem to solve
  # x_guess = np.load("/home/anirudhkailaje/Documents/01_UPenn/02_MEAM5170/03_FinalProject/src/traj.npy")
  # x_init = x_guess[:, ::(x_guess.shape[1])//N][:,:N].T

  x_init = np.linspace(initial_state, initial_state, N)
  u_init = np.random.uniform(low = -effort_limits, high = effort_limits, size=(N, n_u))/1e1
  lambda_init = np.zeros((N, 6))
  lambda_c_col_init = np.zeros((N-1, 6))
  
  prog.SetInitialGuess(x, x_init)
  prog.SetInitialGuess(u, u_init)
  prog.SetInitialGuess(lambda_c, lambda_init)
  prog.SetInitialGuess(lambda_c_col, lambda_c_col_init)

  print("Starting the solve")
  prog.SetSolverOption(SolverType.kSnopt, "Major iterations limit", 3e7) #30000
  prog.SetSolverOption(SolverType.kSnopt, "Minor feasibility tolerance", 1e-5)
  prog.SetSolverOption(SolverType.kSnopt, "Major optimality tolerance", 1e-3)

  # Set up solver
  result = Solve(prog)
  
  x_sol = result.GetSolution(x)
  u_sol = result.GetSolution(u)
  lambda_sol = result.GetSolution(lambda_c)
  # t_land_sol = result.GetSolution(t_land)

  print('optimal cost: ', result.get_optimal_cost())
  print('x_sol: ', x_sol.round(2))
  print('u_sol: ', u_sol.round())
  print(required_velocity)
  print(x_sol[-1][n_q+1])
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

  return x_traj, u_traj, prog, x_sol, u_sol 
  
if __name__ == '__main__':
  N = 3
  initial_state = np.zeros(14)
  q = np.zeros((7,))
  q[0] = 0; q[1] = 0.8
  theta = -np.arccos(q[1])
  q[3] = theta/2; q[4] = -2 * theta
  q[5] = theta;   q[6] = -2 * theta/2
  initial_state[:7] = q
  final_state = initial_state
  # final_configuration = np.array([np.pi, 0])
  tf = 3.0
  x_traj, u_traj, prog,  _, _ = find_throwing_trajectory(N, initial_state, 1.3, tf=1, jumpheight_tol=5e-2)
  # x_traj, u_traj, prog,  _, _ = find_throwing_trajectory(N, initial_state, final_state=final_state, tf=1, jumpheight_tol=5e-2)
  print("Done")