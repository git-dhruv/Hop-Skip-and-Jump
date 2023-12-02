import matplotlib.pyplot as plt
import numpy as np, os
import importlib, logging
from pydrake.all import (DiagramBuilder, Simulator, FindResourceOrThrow, MultibodyPlant, PiecewisePolynomial, SceneGraph,
    Parser, JointActuatorIndex, MathematicalProgram, Solve, SolverType)
from scipy.constants import g
from pydrake.math import RigidTransform

from dynamics_constraints_dev import (
  AddCollocationConstraints,
  EvaluateDynamics, 
  AddAngularMomentumConstraint,
)

from dataclasses import dataclass
from pydrake.multibody.tree import Frame


logger = logging.getLogger('DIRCOL')
def dir_col(N, initial_state, jumpheight, tf, jumpheight_tol=5e-2):

  builder = DiagramBuilder()
  plant = builder.AddSystem(MultibodyPlant(0.0))
  for root, dirs, files in os.walk('../'):
      for file in files:
          if "planar_walker.urdf" in file:
            file_name = os.path.join(root, file)
  
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
  
  # Create the mathematical program and decision variables
  prog = MathematicalProgram()
  x = np.array([prog.NewContinuousVariables(n_x, f"x_{i}") for i in  range(N)])
  u = np.array([prog.NewContinuousVariables(n_u, f"u_{i}") for i in  range(N)])
  lambda_c_col = np.array([prog.NewContinuousVariables(8, f"lambda_c_col{i}") for i in  range(N-1)])
  lambda_c = np.array([prog.NewContinuousVariables(8, f"lambda_c_{i}") for i in range(N)])
  gamma = np.array([prog.NewContinuousVariables(6, f"gamma_{i}") for i in range(N)])
  gamma_col = np.array([prog.NewContinuousVariables(6, f"gamma_col_{i}") for i in range(N-1)])

  
  t0 = 0.0; timesteps = np.linspace(t0, tf, N)
  x0 = x[0]; xf = x[-1]

  initial_state_constraint = prog.AddLinearEqualityConstraint(x0, initial_state)
  initial_state_constraint.evaluator().set_description("Initial State Constraint")

  """Jump velocity constraints"""
  required_velocity = (2*g*jumpheight)**0.5
  velocity_tol = np.array([(2*g*jumpheight_tol)**0.5])
  velocity_tol = 1e-2
  # prog.AddLinearEqualityConstraint(xf[n_q+1], final_configuration)
  final_zvel_constraint = prog.AddBoundingBoxConstraint(required_velocity-velocity_tol, required_velocity+velocity_tol, xf[n_q+1]) 
  final_zvel_constraint.evaluator().set_description("Final Z velocity constraint")


  # Add the collocation aka dynamics constraints
  AddCollocationConstraints(prog, robot, context, N, x, u, lambda_c, lambda_c_col, gamma, gamma_col, timesteps)
  
  for i in range(N-1):
       prog.AddQuadraticCost(0.5*(u[i] - u[i+1]).T @ (u[i] - u[i+1]).T)
       prog.AddLinearConstraint(x[i][1], 0.6, 1.2)
       prog.AddLinearConstraint(x[i][0], -1e-2, 1e-2)
       prog.AddLinearConstraint(x[i][2], -1e-4, 1e-4)
       
       if i>=N-3:
        prog.AddLinearConstraint(x[i+1][n_q+1] - x[i][n_q+1] , -9, 30)
        # Adding costs for angular momentum
        # AddAngularMomentumConstraint(prog, robot, context, x[i], 1000000)
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
      # prog.AddBoundingBoxConstraint(-vel_limits, vel_limits, x[i, n_q:n_q+n_v])
      prog.AddBoundingBoxConstraint(-effort_limits, effort_limits, u[i])
      # The constraint is applied to the 6x1 lambda_c vector
      prog.AddLinearConstraint(mu*lambda_c[i][3]-lambda_c[i][0]-lambda_c[i][1], 0, np.inf)
      prog.AddLinearConstraint(mu*lambda_c[i][7]-lambda_c[i][4]-lambda_c[i][5], 0, np.inf)
      prog.AddBoundingBoxConstraint(np.zeros(8), np.ones(8)*np.inf, lambda_c[i])
      prog.AddBoundingBoxConstraint(np.zeros(6), np.ones(6)*np.inf, gamma[i])
      prog.AddLinearEqualityConstraint(lambda_c[i][2] == 0)
      prog.AddLinearEqualityConstraint(lambda_c[i][5] == 0)
  for i in range(N-1):
      prog.AddBoundingBoxConstraint(np.zeros(8), np.ones(8)*np.inf, lambda_c_col[i])
      prog.AddBoundingBoxConstraint(np.zeros(6), np.ones(6)*np.inf, gamma_col[i])
      prog.AddLinearEqualityConstraint(lambda_c_col[i][2] == 0)
      prog.AddLinearEqualityConstraint(lambda_c_col[i][5] == 0)

  lambda_init = np.zeros((N, 8))
  lambda_c_col_init = np.zeros((N-1, 8))
  
  prog.SetInitialGuess(lambda_c, lambda_init)
  prog.SetInitialGuess(lambda_c_col, lambda_c_col_init)

  logger.debug("Starting the solve")
  
  prog.SetSolverOption(SolverType.kSnopt, "Major iterations limit", 30000) #30000
  prog.SetSolverOption(SolverType.kSnopt, "Minor feasibility tolerance", 1e-3)
  prog.SetSolverOption(SolverType.kSnopt, "Major optimality tolerance", 1e-4)
  
  result = Solve(prog)
  
  x_sol = result.GetSolution(x); 
  u_sol = result.GetSolution(u); 
  lambda_sol = result.GetSolution(lambda_c)

  logger.debug(f'optimal cost: {result.get_optimal_cost()}')
  logger.debug(f'x_sol: {x_sol.round(2)}')
  logger.debug(f'u_sol: {u_sol.round()}' )
  logger.debug(f'Required Velocity: {required_velocity}')
  logger.debug(f'Achieved Dircol Velocity: {x_sol[-1][n_q+1]}')
  logger.debug(result.get_solution_result())

  print(f'optimal cost: {result.get_optimal_cost()}')
  print(f'x_sol: {x_sol.round(2)}')
  print(f'u_sol: {u_sol.round()}' )
  print(f'Required Velocity: {required_velocity}')
  print(f'Achieved Dircol Velocity: {x_sol[-1][n_q+1]}')
  print(result.get_solution_result())

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
  
  tf = 3.0
  x_traj, u_traj, prog,  _, _ = dir_col(N, initial_state, 0.3, tf=1, jumpheight_tol=5e-2)
  logger.debug("Done")
