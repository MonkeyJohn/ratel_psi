import yaml
import numpy as np
import math

from carmodel import BrushTireModel, LinearTireModel
from renderer import _Renderer

def control(X, params):

    beta_eq = -12*math.pi/180
    ux_eq = 8
    r_eq = 0.6
    
    FxR_eq = 2293
    FyF_eq = 3807
    FyR_eq = 4469

    K_beta = 2
    K_r    = 4
    K_Ux   = 0.846
    
    v_x = X[3]
    v_y = X[4]
    
    beta = np.arctan2(v_y,v_x)
    e_beta = beta - beta_eq
    
    r = X[5]
    r_des = r_eq + K_beta*e_beta

    e_r = r - r_des
    e_v_x = v_x - ux_eq

    k1 = params['L_f']/params['I_z'] - K_beta/(params['m'] * v_x)
    k2 = params['L_r']/params['I_z'] - K_beta/(params['m'] * v_x)

    FxR_des = FxR_eq - params['m'] * K_Ux * e_v_x
    v_x_des = params['C_x'] * v_x /(params['C_x'] - FxR_des)



    action = np.array([15, 0.3])
    return action 




def simulation():
    stream = open('/home/xingjiansheng/Documents/src/workplace_xjs/drift/ratel_psi/src/car.yaml','r')
    params = yaml.load(stream)
    model = BrushTireModel(params, 1.37, 1.96)
    renderer = _Renderer(params)
    # print(params)

    state0 = np.zeros(6)
    action_hardcode = np.array([15, 0.0])
    
    iterations = 1500
    dt = 0.03
    flag = True
    for i in range(iterations):
        print(state0[3])
        if state0[3] > 12 and flag:
            action_hardcode = control(state0, params)
            flag = False
        state0 = model.state_transition(state0, action_hardcode, dt)
        renderer.update(state0,action_hardcode)


if __name__ == "__main__":
    simulation()
