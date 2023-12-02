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

def EvaluateDynamics(robot, context, x, u, lambda_c):
  # Computes the dynamics xdot = f(x,u)

  robot.SetPositionsAndVelocities(context, x)
  n_v = robot.num_velocities()

  M = robot.CalcMassMatrixViaInverseDynamics(context)
  B = robot.MakeActuationMatrix()
  g = robot.CalcGravityGeneralizedForces(context)
  C = robot.CalcBiasTerm(context)

  J_c, J_c_dot_v = CalculateContactJacobian(0, robot, context)
  J_c_2, J_c_dot_v_2 = CalculateContactJacobian(1,robot, context)
  J_c = np.row_stack((J_c, J_c_2))
  J_c_dot_v = np.row_stack((J_c_dot_v.reshape(-1,1), J_c_dot_v_2.reshape(-1,1)))
  
  M_inv = np.zeros((n_v,n_v)) 
  if(x.dtype == AutoDiffXd):
    M_inv = pydrake.math.inv(M)
  else:
    M_inv = np.linalg.inv(M)

  contact_force = np.array([lambda_c[0]-lambda_c[1], lambda_c[2], lambda_c[3], lambda_c[4]-lambda_c[5], lambda_c[6], lambda_c[7]])
  v_dot = M_inv @ (B @ u + g - C + J_c.T@contact_force)
  
  state = fetchStates(context=context, plant=robot)
  foot = np.concatenate((state['left_leg'], state['right_leg']))
  foot_vel = np.concatenate((state['leftVel'],state['rightVel']))

  return np.hstack((x[-n_v:], v_dot)), foot, foot_vel


def CollocationConstraintEvaluator(robot, context, dt, x_i, u_i, x_ip1, u_ip1, lambda_c_i, lambda_c_ip1, lambda_c_halfway):
  n_x = robot.num_positions() + robot.num_velocities()
  h_i = np.zeros(n_x,)
  """Adding dynamics constraint using x_i, u_i, x_ip1, u_ip1, dt """
  fi,foot_i ,foot_vel_i= EvaluateDynamics(robot, context, x_i, u_i, lambda_c_i)
  fi1, foot_i1, foot_vel_i1 = EvaluateDynamics(robot, context, x_ip1, u_ip1, lambda_c_ip1)

  s_halfway = (x_i+x_ip1)*0.5 - 0.125*(dt)*(fi1-fi)
  sdot_halfway = 1.5*(x_ip1-x_i)/dt - 0.25*(fi+fi1)
  u_halfway = (u_i+u_ip1)*0.5

  dyn, foot_halfway, foot_vel_halfway = EvaluateDynamics(robot, context, s_halfway, u_halfway, lambda_c_halfway)
  h_i = sdot_halfway - dyn

  feet_pos = np.concatenate((foot_i, foot_halfway, foot_i1))
  feet_vel = np.concatenate((foot_vel_i, foot_vel_halfway, foot_vel_i1))
  
  return h_i, feet_pos, feet_vel

def AddCollocationConstraints(prog, robot, context, N, x, u, lambda_c, lambda_c_col, gamma, gamma_col, timesteps):
  n_u = robot.num_actuators()
  n_x = robot.num_positions() + robot.num_velocities()
  n_lambda = 6

  for i in range(N - 1):
    lower_bound = np.zeros(n_x)
    upper_bound = lower_bound
    eps = 1e-4
    feet_lb = np.zeros((6*3,1))
    feet_ub = np.ones((6*3,1))*np.inf

    def CollocationConstraintHelper_1(vars):
      x_i = vars[:n_x]
      x_ip1 = vars[n_x:2*n_x]
      u_i = vars[2*n_x: 2*n_x + n_u]
      u_ip1 = vars[2*n_x+n_u:2*(n_x+n_u)]
      lambda_c_i = vars[2*(n_x+n_u):2*(n_x+n_u)+8]
      lambda_c_ip1 = vars[2*(n_x+n_u)+8:2*(n_x+n_u)+16]
      lambda_c_halfway = vars[2*(n_x+n_u)+16:2*(n_x+n_u)+24]
      gamma_i = vars[2*(n_x+n_u)+24: 2*(n_x+n_u)+30]
      gamma_i1 = vars[2*(n_x+n_u)+30:2*(n_x+n_u)+36]
      gamma_halfway = vars[2*(n_x+n_u)+36:]

      return CollocationConstraintEvaluator(robot, context, timesteps[i+1] - timesteps[i], x_i, u_i, x_ip1, u_ip1, lambda_c_i, lambda_c_ip1, lambda_c_halfway)[0]

    def CollocationConstraintHelper_2(vars):
      x_i = vars[:n_x]
      x_ip1 = vars[n_x:2*n_x]
      u_i = vars[2*n_x: 2*n_x + n_u]
      u_ip1 = vars[2*n_x+n_u:2*(n_x+n_u)]
      lambda_c_i = vars[2*(n_x+n_u):2*(n_x+n_u)+8]
      lambda_c_ip1 = vars[2*(n_x+n_u)+8:2*(n_x+n_u)+16]
      lambda_c_halfway = vars[2*(n_x+n_u)+16:]
      gamma_i = vars[2*(n_x+n_u)+24: 2*(n_x+n_u)+30]
      gamma_i1 = vars[2*(n_x+n_u)+30:2*(n_x+n_u)+36]
      gamma_halfway = vars[2*(n_x+n_u)+36:]

      return CollocationConstraintEvaluator(robot, context, timesteps[i+1] - timesteps[i], x_i, u_i, x_ip1, u_ip1, lambda_c_i, lambda_c_ip1, lambda_c_halfway)[1]

    def CollocationConstraintHelper_3(vars):
      x_i = vars[:n_x]
      x_ip1 = vars[n_x:2*n_x]
      u_i = vars[2*n_x: 2*n_x + n_u]
      u_ip1 = vars[2*n_x+n_u:2*(n_x+n_u)]
      lambda_c_i = vars[2*(n_x+n_u):2*(n_x+n_u)+8]
      lambda_c_ip1 = vars[2*(n_x+n_u)+8:2*(n_x+n_u)+16]
      lambda_c_halfway = vars[2*(n_x+n_u)+16:]
      gamma_i = vars[2*(n_x+n_u)+24: 2*(n_x+n_u)+30]
      gamma_i1 = vars[2*(n_x+n_u)+30:2*(n_x+n_u)+36]
      gamma_halfway = vars[2*(n_x+n_u)+36:]

      feet_vel = CollocationConstraintEvaluator(robot, context, timesteps[i+1] - timesteps[i], x_i, u_i, x_ip1, u_ip1, lambda_c_i, lambda_c_ip1, lambda_c_halfway)[2]
      foot_vel_i = feet_vel[:6]; foot_vel_halfway = feet_vel[6:12]

      slack_constraint = np.concatenate((gamma_i-foot_vel_i, gamma_halfway-foot_vel_halfway))
      return slack_constraint
    
    def CollocationConstraintHelper_4(vars):
      x_i = vars[:n_x]
      x_ip1 = vars[n_x:2*n_x]
      u_i = vars[2*n_x: 2*n_x + n_u]
      u_ip1 = vars[2*n_x+n_u:2*(n_x+n_u)]
      lambda_c_i = vars[2*(n_x+n_u):2*(n_x+n_u)+8]
      lambda_c_ip1 = vars[2*(n_x+n_u)+8:2*(n_x+n_u)+16]
      lambda_c_halfway = vars[2*(n_x+n_u)+16:]
      gamma_i = vars[2*(n_x+n_u)+24: 2*(n_x+n_u)+30]
      gamma_i1 = vars[2*(n_x+n_u)+30:2*(n_x+n_u)+36]
      gamma_halfway = vars[2*(n_x+n_u)+36:]

      feet_pos, feet_vel = CollocationConstraintEvaluator(robot, context, timesteps[i+1] - timesteps[i], x_i, u_i, x_ip1, u_ip1, lambda_c_i, lambda_c_ip1, lambda_c_halfway)[1:]
      foot_vel_i = feet_vel[:6]; foot_vel_halfway = feet_vel[6:12]

      mu = 1
      friction_force = lambda x: np.array([mu*x[2]-x[0]-x[1]])
      frictioncone_constraint1 = friction_force(lambda_c_i[:4]) * gamma_i[:3] #Shape 1
      frictioncone_constraint2 = friction_force(lambda_c_i[4:]) * gamma_i[3:]
      frictioncone_constraint3 = friction_force(lambda_c_halfway[:4]) * gamma_halfway[:3]
      frictioncone_constraint4 = friction_force(lambda_c_halfway[4:]) * gamma_halfway[3:]
      frictioncone_constraints = np.concatenate((frictioncone_constraint1, frictioncone_constraint2, frictioncone_constraint3, frictioncone_constraint4))

      lambda_x = np.array([lambda_c_i[0],lambda_c_i[0],lambda_c_i[0], lambda_c_i[4], lambda_c_i[4],lambda_c_i[4]])
      lambda_x_halfway = np.array([lambda_c_halfway[0],lambda_c_halfway[0],lambda_c_halfway[0], lambda_c_halfway[4], lambda_c_halfway[4],lambda_c_halfway[4]])
      lambda_z = np.array([lambda_c_i[3],lambda_c_i[3],lambda_c_i[3], lambda_c_i[7], lambda_c_i[7],lambda_c_i[7]])
      lambda_z_halfway = np.array([lambda_c_halfway[3],lambda_c_halfway[3],lambda_c_halfway[3], lambda_c_halfway[7], lambda_c_halfway[7],lambda_c_halfway[7]])
      
      lambda_complementaryconstraint1 = feet_pos[:2*3] @ lambda_z
      lambda_complementaryconstraint2 = feet_pos[2*3:2*3*2] @ lambda_z_halfway
      lambda_complementaryconstraints = np.array([lambda_complementaryconstraint1, lambda_complementaryconstraint2])
      
      slidingfriction_constraint1 = (gamma_i + foot_vel_i)@lambda_x
      slidingfriction_constraint2 = (gamma_halfway + foot_vel_halfway)@lambda_x_halfway
      slidingfriction_constraints = np.array([slidingfriction_constraint1, slidingfriction_constraint2])

      additional_constraint1 = (gamma_i+foot_vel_i)@lambda_x
      additional_constraint2 = (gamma_halfway+foot_vel_halfway)@lambda_x_halfway
      additional_constraint3 = (gamma_i-foot_vel_i)@lambda_x
      additional_constraint4 = (gamma_halfway-foot_vel_halfway)@lambda_x_halfway
      additional_constraints = np.array([additional_constraint1, additional_constraint2, additional_constraint3, additional_constraint4])

      return np.concatenate((frictioncone_constraints, slidingfriction_constraints, additional_constraints, lambda_complementaryconstraints))


    
    prog.AddConstraint(CollocationConstraintHelper_1, lower_bound-eps, upper_bound+eps,np.hstack([x[i], x[i+1], u[i], u[i+1], lambda_c[i], lambda_c[i+1], lambda_c_col[i], gamma[i], gamma[i+1], gamma_col[i]]) )
    prog.AddConstraint(CollocationConstraintHelper_2, feet_lb, feet_ub, np.hstack([x[i], x[i+1], u[i], u[i+1], lambda_c[i], lambda_c[i+1], lambda_c_col[i], gamma[i], gamma[i+1], gamma_col[i]]))
    prog.AddConstraint(CollocationConstraintHelper_3, np.zeros(12), np.zeros(12),np.hstack([x[i], x[i+1], u[i], u[i+1], lambda_c[i], lambda_c[i+1], lambda_c_col[i], gamma[i], gamma[i+1], gamma_col[i]]))
    prog.AddConstraint(CollocationConstraintHelper_4, np.zeros((20,1)), np.zeros((20,1)),np.hstack([x[i], x[i+1], u[i], u[i+1], lambda_c[i], lambda_c[i+1], lambda_c_col[i], gamma[i], gamma[i+1], gamma_col[i]]))

def AddAngularMomentumConstraint(prog, robot, context, x, Lthd):

  def AngularMomentumHelper(vars):
    x = vars
    base_point = robot.CalcPointsPositions(context, robot.GetFrameByName("base"), np.array([0,0,0]), robot.world_frame())
    L = robot.CalcSpatialMomentumInWorldAboutPoint(context, base_point).rotational().ravel()
    return L

  prog.AddConstraint(AngularMomentumHelper, np.zeros(3)-Lthd/2, np.zeros(3)+Lthd/2, x)  
  