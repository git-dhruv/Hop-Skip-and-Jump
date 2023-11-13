import numpy as np
from pydrake.multibody.all import JacobianWrtVariable
import pydrake.math
from pydrake.autodiffutils import AutoDiffXd
from utils import *

def CalculateContactJacobian( fsm: int, plant,plant_context) :
    """
        For a given finite state, LEFT_STANCE or RIGHT_STANCE, calculate the
        Jacobian terms for the contact constraint, J and Jdot * v.

        As an example, see CalcJ and CalcJdotV in PointPositionTrackingObjective

        use contact_points to get the PointOnFrame for the current stance foot
    """
    LEFT_STANCE = 0
    RIGHT_STANCE = 1


    contact_points = {
            LEFT_STANCE: PointOnFrame(
                plant.GetBodyByName("left_lower_leg").body_frame(),
                np.array([0, 0, -0.5])
            ),
            RIGHT_STANCE: PointOnFrame(
                plant.GetBodyByName("right_lower_leg").body_frame(),
                np.array([0, 0, -0.5])
            )
        }
    # TODO - STUDENT CODE HERE:
    pt_to_track = contact_points[fsm]
    J = plant.CalcJacobianTranslationalVelocity(
        plant_context, JacobianWrtVariable.kV, pt_to_track.frame,
        pt_to_track.pt, plant.world_frame(), plant.world_frame()
    )

    JdotV = plant.CalcBiasTranslationalAcceleration(
        plant_context, JacobianWrtVariable.kV, pt_to_track.frame,
        pt_to_track.pt, plant.world_frame(), plant.world_frame()
    ).ravel()

    return J, JdotV

def EvaluateDynamics(planar_arm, context, x, u, lambda_c):
  # Computes the dynamics xdot = f(x,u)

  planar_arm.SetPositionsAndVelocities(context, x)
  n_v = planar_arm.num_velocities()

  M = planar_arm.CalcMassMatrixViaInverseDynamics(context)
  B = planar_arm.MakeActuationMatrix()
  g = planar_arm.CalcGravityGeneralizedForces(context)
  C = planar_arm.CalcBiasTerm(context)

  J_c, J_c_dot_v = CalculateContactJacobian(0, planar_arm, context)
  J_c_2, J_c_dot_v_2 = CalculateContactJacobian(1,planar_arm, context)
  J_c = np.row_stack((J_c, J_c_2))
  J_c_dot_v = np.row_stack((J_c_dot_v.reshape(-1,1), J_c_dot_v_2.reshape(-1,1)))
  

  M_inv = np.zeros((n_v,n_v)) 
  if(x.dtype == AutoDiffXd):
    M_inv = pydrake.math.inv(M)
  else:
    M_inv = np.linalg.inv(M)
  v_dot = M_inv @ (B @ u + g - C + J_c.T@lambda_c)

  contact = J_c_dot_v + (J_c@v_dot).reshape(-1,1)
  return np.hstack((x[-n_v:], v_dot)), contact

def CollocationConstraintEvaluator(planar_arm, context, dt, x_i, u_i, x_ip1, u_ip1, lambda_c_i, lambda_c_ip1, lambda_c_halfway):
  n_x = planar_arm.num_positions() + planar_arm.num_velocities()
  h_i = np.zeros(n_x,)
  # TODO: Add a dynamics constraint using x_i, u_i, x_ip1, u_ip1, dt
  # You should make use of the EvaluateDynamics() function to compute f(x,u)
  fi,_ = EvaluateDynamics(planar_arm, context, x_i, u_i, lambda_c_i)
  fi1, _ = EvaluateDynamics(planar_arm, context, x_ip1, u_ip1, lambda_c_ip1)

  s_halfway = (x_i+x_ip1)*0.5 - 0.125*(dt)*(fi1-fi)
  sdot_halfway = 1.5*(x_ip1-x_i)/dt - 0.25*(fi+fi1)
  u_halfway = (u_i+u_ip1)*0.5

  dyn, contact = EvaluateDynamics(planar_arm, context, s_halfway, u_halfway, lambda_c_halfway)
  h_i = sdot_halfway - dyn
  

  return h_i, contact

def AddCollocationConstraints(prog, planar_arm, context, N, x, u, lambda_c, lambda_c_col, timesteps):
  n_u = planar_arm.num_actuators()
  n_x = planar_arm.num_positions() + planar_arm.num_velocities()
  n_lambda = 6

  def CollocationConstraintHelper_1(vars):
      x_i = vars[:n_x]
      x_ip1 = vars[n_x:2*n_x]
      u_i = vars[2*n_x: 2*n_x + n_u]
      u_ip1 = vars[2*n_x+n_u:2*(n_x+n_u)]
      lambda_c_i = vars[2*(n_x+n_u):2*(n_x+n_u)+6]
      lambda_c_ip1 = vars[2*(n_x+n_u)+6:2*(n_x+n_u)+12]
      lambda_c_halfway = vars[2*(n_x+n_u)+12:]

      return CollocationConstraintEvaluator(planar_arm, context, timesteps[i+1] - timesteps[i], x_i, u_i, x_ip1, u_ip1, lambda_c_i, lambda_c_ip1, lambda_c_halfway)[0]

  def CollocationConstraintHelper_2(vars):
    x_i = vars[:n_x]
    x_ip1 = vars[n_x:2*n_x]
    u_i = vars[2*n_x: 2*n_x + n_u]
    u_ip1 = vars[2*n_x+n_u:2*(n_x+n_u)]
    lambda_c_i = vars[2*(n_x+n_u):2*(n_x+n_u)+6]
    lambda_c_ip1 = vars[2*(n_x+n_u)+6:2*(n_x+n_u)+12]
    lambda_c_halfway = vars[2*(n_x+n_u)+12:]

    return CollocationConstraintEvaluator(planar_arm, context, timesteps[i+1] - timesteps[i], x_i, u_i, x_ip1, u_ip1, lambda_c_i, lambda_c_ip1, lambda_c_halfway)[1]
  
  for i in range(N - 1):
    # TODO: Within this loop add the dynamics constraints for segment i (aka collocation constraints)
    #       to prog
    # Hint: use prog.AddConstraint(CollocationConstraintHelper, lb, ub, vars)
    # where vars = hstack(x[i], u[i], ...)
    lower_bound = np.zeros(n_x)
    upper_bound = lower_bound
    eps = 1e-4
    contact_lower_bound = np.zeros(6)
    contact_upper_bound = contact_lower_bound

    prog.AddConstraint(CollocationConstraintHelper_1, lower_bound-eps, upper_bound+eps,np.hstack([x[i], x[i+1], u[i], u[i+1], lambda_c[i], lambda_c[i+1], lambda_c_col[i]]) )
    prog.AddConstraint(CollocationConstraintHelper_2, contact_lower_bound-eps, contact_upper_bound+eps,np.hstack([x[i], x[i+1], u[i], u[i+1], lambda_c[i], lambda_c[i+1], lambda_c_col[i]]))
    # prog.AddLinearEqualityConstraint(contact, np.zeros((6,1)))
