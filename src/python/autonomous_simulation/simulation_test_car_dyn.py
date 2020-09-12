import yaml
import numpy as np
import math
from scipy.integrate import solve_ivp

from car_dynamics import DynamicBicycleModel
from iLQR import iLQR

def clamp_to_limits(u_curr, params):
    u_clamped = np.zeros((2,1),dtype=float)
    u_clamped[0,0] = min(params['u0_max'],max(params['u0_min'],u_curr[0,0]))
    u_clamped[1,0] = min(params['u1_max'],max(params['u1_min'],u_curr[1,0]))
    return u_clamped

class CarParkingEnv:
    def __init__(self, model, params):
        self.model = model
        self.params = params
        # self.T = 50
        self.u_lims = np.array([[-1, 4],[-0.76,0.68]])  # control limits

    def dynamics(self, x, u):
        dt = 0.02
        n = x.shape[0]
        m = u.shape[0]
        N = x.shape[1]
        state = np.zeros((n, N), dtype = float)
        for i in range(N):
            state[:,i] = self.model.state_transition(x[:,i], u[:,i], dt)
        return state

    def cost(self, x , u):

        n = x.shape[0]
        m = u.shape[0]
        N = x.shape[1]

        xd = np.zeros((n,N),dtype=float)
        xd[0][N-1] = 1
        xd[1][N-1] = 1
        delta_x = x - xd

        cu = np.array([0.001 , 0.001])
        lu = u[0] ** 2 * cu[0] + u[1] ** 2 * cu[1]

        cx = np.array([0.5, 0.1, 0.4, 0.05, 0.005, 0.002])
        lx = delta_x[0] ** 2 * cx[0] + delta_x[1] ** 2 * cx[1] + delta_x[2] ** 2 * cx[2] + delta_x[3] ** 2 * cx[3] + delta_x[4] ** 2 * cx[4] + delta_x[5] ** 2 * cx[5]

        total_cost = lu + lx

        return total_cost

    def dyn_cst(self, x, u):

        f = self.dynamics(x, u)
        c = self.cost(x, u)

        n = x.shape[0]
        m = u.shape[0]
        N = x.shape[1]

        fx = np.zeros((N, n, n), dtype=float)
        fu = np.zeros((N, n, m), dtype=float)
        cx = np.zeros((n, N), dtype=float)
        cu = np.zeros((m, N), dtype=float)
        cxx = np.zeros((N, n, n), dtype=float)
        cux = np.zeros((N, m, n), dtype=float)
        cuu = np.zeros((N, m, m), dtype=float)

        eps = 0.001
        dt = 0.02
        # T = self.T
        model = self.model

        print("*****x")
        print(x)
        print("*****u")
        print(u)
        for t in range(N):
            # xs = np.array([[x[0,t]],[x[1,t]],[x[2,t]],[x[3,t]],[x[4,t]],[x[5,t]]])
            # us = np.array([[u[0,t]],[u[1,t]]])

            # dynamics_derivatives
            for i in range(n):
                plus = minus = x[:,t]
                plus[i] += eps
                minus[i] -= eps
                fx[t,:,i] = (model.state_transition(plus, u[:,t], dt) - model.state_transition(minus, u[:,t], dt))/(2*eps)

            for i in range(m):
                plus = minus = u[:,t]
                plus[i] += eps
                minus[i] -= eps
                fu[t,:,i] = (model.state_transition(x[:,t], plus, dt) - model.state_transition(x[:,t], minus, dt))/(2*eps)
        print("*****fx")
        print(fx)
        # get_cost_derivatives
        for i in range(n):
            plus = minus = x
            plus[i,:] += eps
            minus[i,:] -= eps
            cx[i,:] = (self.cost(plus, u) - self.cost(minus, u))/(2*eps)
        
        for i in range(m):
            plus = minus = u
            plus[i,:] += eps
            minus[i,:] -= eps
            cu[i,:] = (self.cost(x, plus) - self.cost(x, minus))/(2*eps)

        #cxx
        for i in range(n):
            for j in range(i,n):
                pp = pm = mp = mm = x
                pm[i,:] += eps
                pp[i,:] = pm[i,:]
                mm[i,:] -= eps
                mp[i,:] = mm[i,:]
                mp[j,:] += eps
                pp[j,:] = mp[j,:]
                mm[j,:] -= eps
                pm[j,:] = mm[j,:]
                cxx[:,i,j] = cxx[:,j,i] = (self.cost(pp, u) - self.cost(pm,u) - self.cost(mp,u) + self.cost(mm,u))/(4*eps*eps)
        #cuu
        for i in range(m):
            for j in range(i,m):
                pp = pm = mp = mm = u
                pm[i,:] += eps
                pp[i,:] = pm[i,:]
                mm[i,:] -= eps
                mp[i,:] = mm[i,:]
                mp[j,:] += eps
                pp[j,:] = mp[j,:]
                mm[j,:] -= eps
                pm[j,:] = mm[j,:]
                cuu[:,i,j] = cuu[:,j,i] = (self.cost(x, pp) - self.cost(x, mp) - self.cost(x,pm) + self.cost(x,mm))/(4*eps*eps)
        #cux
        for i in range(n):
            for j in range(m):
                px = mx = x
                pu = mu = u
                px[i,:] += eps
                mx[i,:] -= eps
                pu[j,:] += eps
                mu[j,:] -= eps
                cux[:,j,i] = (self.cost(px,pu) - self.cost(mx,pu) - self.cost(px,mu) + self.cost(mx,mu))/(4*eps*eps)

        return f, c, fx, fu, cx, cu, cxx, cuu, cux

def test_planner():
    stream = open('/home/xingjiansheng/Workplace/project/MotionPlanner/Drift/ratel_psi/src/python/autonomous_simulation/car.yaml','r')
    params = yaml.load(stream)
    model = DynamicBicycleModel(params)
    env = CarParkingEnv(model, params)
    
    x0 = np.zeros((6,1),dtype=float)
    T = 50  # horizon
    u0 = np.zeros((2, T), dtype=float)
    for i in range(T):
        u0[0,i] = 1.0
        u0[1,i] = 0.3

    DYNCST = lambda x, u: env.dyn_cst(x, u)
    # === run the optimization
    x, u, cost = iLQR(DYNCST, x0, u0, env.u_lims)
    print(x)
    print(u)

def test_model():
    stream = open('/home/xingjiansheng/Documents/src/workplace_xjs/2020_final/ratel_psi/src/python/autonomous_simulation/car.yaml','r')
    params = yaml.load(stream)
    model = DynamicBicycleModel(params)    

    state0 = np.zeros(6)
    action_hardcode = np.array([1.0, 0.3])
    dt = 0.02
    state0 = model.state_transition(state0, action_hardcode, dt)
    # print(state0)
    state0[1] += 0.01
    state1 = model.state_transition(state0, action_hardcode, dt)
    # print(state1)
    delta_state = state1 -state0
    print(delta_state)


if __name__ == "__main__":
    test_model()
    # test_planner()
