import yaml
import numpy as np
import math
from scipy.integrate import solve_ivp

from carmodel import BrushTireModel, LinearTireModel
from renderer import _Renderer

def FialaFront(Ca, mu, Fz, Fx, alpha):
    z = math.tan(alpha)
    xi = 1
    tan_alpha_sl = 3*xi*mu*Fz/Ca
    alpha_sl = np.arctan(tan_alpha_sl)

    if alpha < alpha_sl:
        Fy = -Ca * z + (Ca**2/(3*xi*mu*Fz))*np.abs(z)*z - (Ca**3/(27*(xi**2)*(mu**2)*(Fz**2)))*z**3
    else:
        Fy = -xi*mu*Fz*np.sign(alpha)
    return Fy

def FialaRear(Ca, mu, Fz, Fx, alpha):
    z = math.tan(alpha)
    xi = np.sqrt((mu * Fz)**2 - Fx**2)/(mu * Fz)
    tan_alpha_sl = 3*xi*mu*Fz/Ca
    alpha_sl = np.arctan(tan_alpha_sl)

    if alpha < alpha_sl:
        Fy = -Ca * z + (Ca**2/(3*xi*mu*Fz))*np.abs(z)*z - (Ca**3/(27*(xi**2)*(mu**2)*(Fz**2)))*z**3
    else:
        Fy = -xi*mu*Fz*np.sign(alpha)
    return Fy

def InverseFiala(FyF, params):
    alphaF_LUT = []
    FyF_LUT = []
    for i in range(60):
        alphatmp = -30* math.pi/180 + i * math.pi/180
        alphaF_LUT.append(alphatmp)
        FyFtmp = FialaFront(params['C_alpha'], params['mu'], params['load_f'], 0 , alphatmp )
        FyF_LUT.append(FyFtmp)
    c = np.abs(FyF - np.array(FyF_LUT))
    index = np.argmin(c)
    alphaF = alphaF_LUT[index]
    return alphaF

def FyF2delta(beta, r, Ux, FyF, params):
    a = params['L_f']
    alphaF = InverseFiala(FyF,params )
    delta = -alphaF + math.atan(beta + a/Ux*r) #todo
    return delta

def wraptopi(val):
    """
    Wrap radian value to the interval [-pi, pi].
    """
    pi = np.pi
    val = val - 2*pi*np.floor((val+pi)/(2*pi))
    return val

def state_transition(state, control, dt, params):
    t = np.array([0, dt])
    X_new = solve_ivp(
        fun=(lambda t, state: Dynamic(state, t, control, params)),
        t_span=t, y0=state, atol=1e-5)
    return X_new.y[:,-1]

    # state_new = state + dt*Dynamic(state, dt, control, params)
    # return state_new

def Dynamic(state,t,  control, params):
    
    pos_x = state[0]
    pos_y = state[1]
    pos_yaw = state[2]
    Beta    = state[3]
    r       = state[4]
    Ux      = state[5]
    FxR     = control[0]
    delta   = control[1]

    g = params['g']
    m = params['m']
    Iz = params['I_z']
    a = params['L_f']
    b = params['L_r']
    L = params['L_f'] + params['L_r']
    CaF = params['C_alpha']
    CaR = params['C_alpha']
    mu = params['mu']
    FzF = params['load_f']
    FzR = params['load_r']

    # if Ux <= 0.0001 and Ux >= 0:
    #     Ux = 0.0001
    # if Ux >= -0.0001 and Ux <= 0:
    #     Ux = -0.0001

    alphaF = np.arctan(Beta + a/Ux*r) - delta
    if alphaF > math.pi/2 or alphaF < -math.pi/2:
        print("alphaF = %.2f"%(alphaF))
    FyF = FialaFront(CaF, mu , FzF, FxR, alphaF)
    # print("FyF = %.2f"%(FyF))

    alphaR = np.arctan(Beta - b/Ux*r)
    if alphaR > math.pi/2 or alphaR < -math.pi/2:
        print("alphaR = %.2f"%(alphaR))
    FyR = FialaRear(CaF, mu , FzR, FxR, alphaR)
    # print("FyR = %.2f"%(FyR))

    Uy = Ux*np.tan(Beta)
    v = np.sqrt(Ux**2 + Uy**2)
    pos_x_dot = v*np.cos(Beta+pos_yaw)
    pos_y_dot = v*np.sin(Beta+pos_yaw)

    state_dot = np.zeros(6)
    state_dot[0] = pos_x_dot
    state_dot[1] = pos_y_dot
    state_dot[2] = r
    state_dot[3] = 1/(m*Ux)*(FyF + FyR) - r
    state_dot[4] = 1/Iz*(a*FyF-b*FyR)
    state_dot[5] = 1/m*(FxR-FyF*np.sin(delta))+r*Ux*Beta
    return state_dot

def control(X, U , params):

    action = np.array([15, 0.3])
    
    # params['load_f'] = params['L_r']*params['m']*params['g']/(params['L_f'] + params['L_r'])
    # params['load_r'] = params['L_f']*params['m']*params['g']/(params['L_f'] + params['L_r'])

    beta_eq = -20.44*math.pi/180
    ux_eq = 8
    r_eq = 0.6
    
    FyF_eq = 3807
    FxR_eq = 2293
    FyR_eq = 4469

    delta_max = 35*math.pi/180
    FxR_max = params['mu'] * params['load_r']

    K_beta = 2
    K_r    = 4
    K_Ux   = 0.846
    
    beta = X[3]
    r = X[4]
    v_x = X[5]
    if v_x < 0.0001:
        v_x = 0.0001

    e_beta = beta - beta_eq
    r_des = r_eq + K_beta*e_beta

    e_r = r - r_des
    e_v_x = v_x - ux_eq

    k1 = params['L_f']/params['I_z'] - K_beta/(params['m'] * v_x) #todo
    k2 = params['L_r']/params['I_z'] + K_beta/(params['m'] * v_x)

    # mode 1
    FxR_des = FxR_eq - params['m'] * K_Ux * e_v_x
    FxR_des = np.min([FxR_des , FxR_max])
    FxR_des = np.max([FxR_des , 0])
    
    alphaR = np.arctan(beta - params['L_r']/v_x*r) #todo
    FyR_des = FialaRear(params['C_alpha'], params['mu'], params['load_r'], FxR_des , alphaR )

    FyF_des = 1/k1 * ( k2 * FyR_des - K_beta**2 * e_beta - K_beta * r_eq - (K_beta + K_r) * e_r )


    if np.abs(FyF_des) <= params['mu'] * (params['load_f']): #todo
        delta_des = FyF2delta(beta, r, v_x, FyF_des, params)
        action[0] = FxR_des
        action[1] = delta_des
        print("Mode 1 , FyF %.2f , Fxr %.2f , Fyr %.2f , Delta %.2f "%(FyF_des , FxR_des, FyR_des ,delta_des))
    # mode 2
    else:
        alphaF = np.arctan(beta + params['L_f']/v_x*r) - U[1]
        FyF_des = -params['mu'] * params['load_f']* np.sign(alphaF)
        FyR_des = 1/k2 * ( k1 * FyF_des + K_beta**2 * e_beta + K_beta * r_eq + (K_beta + K_r) * e_r )
        FyR_des = np.min([FyR_des , params['mu'] * params['load_r']])
        FyR_des = np.max([FyR_des , -params['mu'] * params['load_r']])
        
        FxR_des = np.sqrt((params['mu']*params['load_r'])**2 - (FyR_des)**2)
        FxR_des = np.min([FxR_des , FxR_max])
        FxR_des = np.max([FxR_des , 0])

        delta_des = FyF2delta(beta, r,v_x,FyF_des,params)
        action[0] = FxR_des
        action[1] = delta_des
        print("Mode 2 , FyF %.2f , Fxr %.2f , Fyr %.2f , Delta %.2f "%(FyF_des , FxR_des, FyR_des ,delta_des))

    delta_des = np.min([delta_des , delta_max])
    delta_des = np.max([delta_des , -delta_max])
    action[1] = delta_des
    # if FxR_des >= 0:
    #     v_x_des = params['C_x'] * v_x /(params['C_x'] - FxR_des)
    #     action[0] = v_x_des
    # else:
    #     v_x_des = (params['C_x'] + FxR_des) * v_x / params['C_x']
    #     action[0] = v_x_des

    return action

def simulation():
    stream = open('/home/xingjiansheng/Documents/src/workplace_xjs/drift/ratel_psi/src/car.yaml','r')
    params = yaml.load(stream)
    model = BrushTireModel(params, 0.55, 0.55)
    renderer = _Renderer(params)

    state0 = np.zeros(6)
    action_hardcode = np.array([2293.0, 0])
    # action_hardcode = np.array([2293.0, -12*math.pi/180])
    # state0[3] = 0 #(-20.44)*math.pi/180
    # state0[4] = 0.600+0.2
    state0[5] = 8
    iterations = 1500
    dt = 0.02
    flag = False
    for i in range(iterations):
        # print(state0)
        if state0[5] > 12 :
            action_hardcode[0] = 0
            action_hardcode[1] = 0.3
        
        # action_hardcode = control(state0, action_hardcode, params)
        
        state0 = state_transition(state0, action_hardcode, dt, params)
        renderer.update(state0, action_hardcode)


if __name__ == "__main__":
    simulation()
