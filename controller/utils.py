import numpy as np
from pydrake.all import JacobianWrtVariable

def calculateDoubleContactJacobians(plant, context):
    Jleft = plant.CalcJacobianTranslationalVelocity(context, JacobianWrtVariable.kV, 
                                                    plant.GetFrameByName("left_lower_leg"), np.array([0,0, -0.5]), plant.world_frame(), plant.world_frame())
    Jright = plant.CalcJacobianTranslationalVelocity(context, JacobianWrtVariable.kV, 
                                                     plant.GetFrameByName("right_lower_leg"), np.array([0,0, -0.5]), plant.world_frame(), plant.world_frame())
    JdotVleft = plant.CalcBiasTranslationalAcceleration(context, JacobianWrtVariable.kV,
                                                        plant.GetFrameByName("left_lower_leg"), np.array([0,0,-0.5]),plant.world_frame(), plant.world_frame())
    JdotVright = plant.CalcBiasTranslationalAcceleration(context, JacobianWrtVariable.kV,
                                                         plant.GetFrameByName("right_lower_leg"), np.array([0,0,-0.5]),plant.world_frame(), plant.world_frame())    
    J_c = np.row_stack((Jleft, Jright))
    J_c_dot_v = np.row_stack((JdotVleft, JdotVright))
    return J_c, J_c_dot_v