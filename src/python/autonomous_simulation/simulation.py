import yaml
import numpy as np
import math
from scipy.integrate import solve_ivp

from car_dynamics import DynamicBicycleModel
from renderer import _Renderer

def wraptopi(val):
    """
    Wrap radian value to the interval [-pi, pi].
    """
    pi = np.pi
    val = val - 2*pi*np.floor((val+pi)/(2*pi))
    return val

def control(X, U , params):
    action = np.array([15, 0.3])

    return action

def cost(x, u):

    xd = np.zeros((6,1),dtype=float)
    xd[0,0] = 1
    xd[1,0] = 1

    # x_curr = np.zeros((6,1),dtype=float)
    # x_curr[:,0] = x
    # u_curr = np.zeros((2,1),dtype=float)
    # u_curr[:,0] = u

    cu = np.array([[0.001 , 0.001]])
    lu = cu*np.square(u)
    ldu = 0.0

    cx = np.array([[0.5, 0.1, 0.4, 0.05, 0.005, 0.002]])
    lx = cx * np.square(x - xd)
    ldx = 0.0

    total_cost = lu[0,0] + lx[0,0]
    return total_cost

def final_cost(x):
    res_cost = 0.0
    return res_cost

def clamp_to_limits(u_curr, params):
    u_clamped = np.zeros(2,dtype=float)
    u_clamped[0] = min(params['u0_max'],max(params['u0_min'],u_curr[0,0]))
    u_clamped[1] = min(params['u1_max'],max(params['u1_min'],u_curr[1,0]))
    return u_clamped

def forward_pass(x0, u_s, T, model, params):
    xnew = np.zeros((T+1,6,1),dtype=float)
    unew = np.zeros((T,2,1),dtype=float)
    xnew[0] = x0
    total_cost = 0

    u_curr = np.zeros((2,1),dtype=float)
    x_curr = x0
    for i in range(T):
        u_curr = u_s[i]

        # todo this

        unew[i] = clamp_to_limits(u_curr, params)
        dt = 0.02
        state = model.state_transition(x_curr[:,0], unew[i], dt)
        total_cost += cost(x_curr, unew[i])
        
        x_curr[:,0] = state
        xnew[i+1] = x_curr

    total_cost += final_cost(x_curr[:,0])
    return xnew, total_cost

def backward_pass(cx, cu, cxx, cxu, cuu, fx, fu, us, Vx, Vxx, k, K, dV):
    Vx[T] = cx[T]
    Vxx[T] = cxx[T]

    Qx = np.zeros((6,1),dtype=float)
    Qu = np.zeros((2,1),dtype=float)
    Qxx = np.zeros((6,6),dtype=float)
    Qux = np.zeros((2,6),dtype=float)
    Quu = np.zeros((2,2), dtype=float)
    k_i = np.zeros((2,1), dtype=float)
    K_i = np.zeros((2,6), dtype=float)

    for i in range(T-1,-1,-1):
        Qx = cx[i] + fx[i].T * Vx[i+1]
        Qu = cu[i] + fu[i].T * Vx[i+1]

        Qxx = cxx[i] + fx[i].T * Vxx[i+1] * fx[i]
        Qux = cxu[i].T +fu[i].T * Vxx[i+1] * fx[i]
        Quu = cuu[i] + fu[i].T * Vxx[i+1]* fu[i]

        Vxx_reg = Vxx[i+1]
        Qux_reg = cxu[i].T + fu[i].T * Vxx_reg * fx[i]
        lambda = 1.0
        Eye2 = np.identity(2)
        QuuF = cuu[i] + fu[i].T * Vxx_reg * fu[i] + lambda * Eye2



def planner(model, params):
    x0 = np.zeros((6,1),dtype=float)
    xd = np.zeros((6,1),dtype=float)
    xd[0,0] = 1
    xd[1,0] = 1

    T = 50  # horizon
    us = np.zeros((T, 2, 1), dtype=float)

    for i in range(T):
        us[i,0,0] = 1.0
        us[i,1,0] = 0.3
    
    xs , cost_cur = forward_pass(x0, us, T, model, params)

    fx = np.zeros((T+1, 6, 6), dtype=float)
    fu = np.zeros((T+1, 6, 2), dtype=float)
    cx = np.zeros((T+1, 6, 1), dtype=float)
    cu = np.zeros((T+1, 2, 1), dtype=float)
    cxx = np.zeros((T+1, 6, 6), dtype=float)
    cxu = np.zeros((T+1, 6, 2), dtype=float)
    cuu = np.zeros((T+1, 2, 2), dtype=float)
    Vx = np.zeros((T+1, 6, 1), dtype=float)
    Vxx = np.zeros((T+1, 6, 6), dtype=float)
    L = np.zeros((T, 2, 6), dtype=float)
    l= np.zeros((T, 2), dtype=float)
    dV = np.zeros((2, 1), dtype=float)

    maxIter = 30
    stop = False
    for i in range(maxIter):
        if stop:
            break

        #derivatives
        eps = 1e-4
        dt = 0.02
        for t in range(T):

            # dynamics_derivatives
            for i in range(6):
                plus = minus = xs[t]
                plus[i,0] += eps
                minus[i,0] -= eps
                fx[t,:,i] = (model.state_transition(plus[:,0], us[t,:,0], dt) - model.state_transition(minus[:,0], us[t,:,0], dt))/(2*eps)

            for i in range(2):
                plus = minus = us[t]
                plus[i,0] += eps
                minus[i,0] -= eps
                fu[t,:,i] = (model.state_transition(xs[t,:,0], plus[:,0], dt) - model.state_transition(xs[t,:,0], minus[:,0], dt))/(2*eps)

            # get_cost_derivatives
            for i in range(6):
                plus = minus = xs[t]
                plus[i,0] += eps
                minus[i,0] -= eps
                cx[t,i,0] = (cost(plus[:,0], us[t,:,0]) - cost(minus[:,0], us[t,:,0]))/(2*eps)

            for i in range(2):
                plus = minus = us[t]
                plus[i,0] += eps
                minus[i,0] -= eps
                cu[t,i,0] = (cost(xs[t,:,0], plus[:,0]) - cost(xs[t,:,0], minus[:,0]))/(2*eps)

            #get_cost_2nd_derivatives
            #cxx
            for i in range(6):
                for j in range(i,6):
                    pp = pm = mp = mm = xs[t]
                    pp[i,0] = pm[i,0] += eps
                    mp[i,0] = mm[i,0] -= eps
                    pp[j,0] = mp[j,0] += eps
                    pm[j,0] = mm[j,0] -= eps
                    cxx[t,i,j] = cxx[t,j,i] = (cost(pp[:,0],us[t,:,0]) - cost(pm[:,0],us[t,:,0]) - cost(mp[:,0],us[t,:,0]) + cost(mm[:,0],us[t,:,0]))/(4*eps*eps)

            #cxu
            for i in range(6):
                for j in range(2):
                    px = mx = xs[t]
                    pu = mu = us[t]
                    px[i,0] += eps
                    mx[i,0] -= eps
                    pu[j,0] += eps
                    mu[j,0] -= eps
                    cxu[t,i,j] = (cost(px[:,0],pu[:,0]) - cost(mx[:,0],pu[:,0]) - cost(px[:,0],mu[:,0]) + cost(mx[:,0],mu[:,0]))/(4*eps*eps)

            #cuu
            for i in range(2):
                for j in range(2):
                    pp = pm = mp = mm = us[t]
                    pp[i,0] = pm[i,0] += eps
                    mp[i,0] = mm[i,0] -= eps
                    pp[j,0] = mp[j,0] += eps
                    pm[j,0] = mm[j,0] -= eps
                    cuu[t,i,j] = cuu[t,j,i] = (cost(xs[t,:,0], pp[:,0]) - cost(xs[t,:,0], mp[:,0]) - cost(xs[t,:,0],pm[:,0]) + cost(xs[t,:,0],mm[:,0]))/(4*eps*eps)

        backPassDone = False
        while backPassDone==False:
            backward_pass(cx, cu, cxx, cxu, cuu, fx, fu, u, Vx, Vxx, l, L, dV)


def simulation():
    stream = open('/home/xingjiansheng/Documents/src/workplace_xjs/drift/ratel_psi/src/python/autonomous_simulation/car.yaml','r')
    params = yaml.load(stream)
    model = DynamicBicycleModel(params)
    renderer = _Renderer(params)

    state0 = np.zeros(6)
    action_hardcode = np.array([8.0, 0.2])
    iterations = 1500
    dt = 0.02

    planner(model, params)

    for i in range(iterations):
        # print(state0)
        # if state0[5] > 12 :
        #     action_hardcode[0] = 0
        #     action_hardcode[1] = 0.3
        
        action_hardcode = control(state0, action_hardcode, params)
        
        state0 = model.state_transition(state0, action_hardcode, dt)
        # print("state0 , bate %.2f , r %.2f , vx %.2f "%(state0[3]/3.14*180 , state0[4], state0[5]))
        
        renderer.update(state0, action_hardcode)


if __name__ == "__main__":
    simulation()
