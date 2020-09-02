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
    u_clamped = np.zeros((2,1),dtype=float)
    u_clamped[0,0] = min(params['u0_max'],max(params['u0_min'],u_curr[0,0]))
    u_clamped[1,0] = min(params['u1_max'],max(params['u1_min'],u_curr[1,0]))
    return u_clamped

def forward_pass(x0, u_s, T, x, L, init_forward, model, params):
    xnew = np.zeros((T+1,6,1),dtype=float)
    unew = np.zeros((T,2,1),dtype=float)
    xnew[0] = x0
    total_cost = 0

    u_curr = np.zeros((2,1),dtype=float)
    x_curr = x0
    for i in range(T):
        u_curr = u_s[i]

        if not init_forward:
            dx = xnew[i] - x[i]
            u_curr += L[i] @ dx

        unew[i] = clamp_to_limits(u_curr, params)
        dt = 0.02
        state = model.state_transition(x_curr[:,0], unew[i], dt)
        total_cost += cost(x_curr, unew[i])
        x_curr[:,0] = state
        xnew[i+1] = x_curr

    total_cost += final_cost(x_curr[:,0])
    return xnew, unew , total_cost

def boxQP(H, g, x0, params):
    
    Hfree = np.zeros((2,2),dtype=float)
    free_v = np.zeros((2,1),dtype=int)

    maxIter = 100
    minGrad = 1e-8
    minRelImprove = 1e-8
    stepDec = 0.6
    minStep = 1e-22
    Armijo = 0.1

    lower = np.array([[params['u0_min']],
                    [params['u1_min']]])
    upper = np.array([[params['u0_max']],
                    [params['u1_max']]])

    x = clamp_to_limits(x0,params)

    value = np.dot(x.T,g) + 0.5 * (x.T @ H @ x)
    # print("*****Start BoxQp initvalue: %.2f  "%(value[0,0]))

    result = 0
    nfactor = 0
    oldvalue = np.zeros((1,1),dtype=float)
    factorize = False

    clamped = np.zeros((2,1),dtype=int)
    old_clamped = np.zeros((2,1),dtype=int)

    for iter in range(maxIter):
        if result != 0:
            break

        if (iter > 0) and (oldvalue - value)[0,0] < minRelImprove * np.abs(oldvalue[0,0]):
            result = 4
            break

        oldvalue = value
        
        # Get gradient
        grad = g + H @ x

        old_clamped = clamped
        clamped = np.zeros((2,1),dtype=int)
        free_v = np.ones((2,1),dtype=int)

        for i in range(2):
            if np.abs(x[i,0] - lower[i,0]) < 1e-3 and grad[i,0] > 0:
                clamped[i,0] = 1
                free_v[i,0] = 0
            elif np.abs(x[i,0] - upper[i,0]) < 1e-3 and grad[i,0] < 0:
                clamped[i,0] = 1
                free_v[i,0] = 0
        
        if clamped.sum() == 2:
            result = 6
            break

        if iter == 0:
            factorize = True
        elif (old_clamped - clamped).sum() != 0:
            factorize = True
        else:
            factorize = False

        if factorize:
            n_free = free_v.sum()
            # print("*****n_free : %d  "%(n_free))
            # print("*****H")
            # print(H)

            if free_v[0,0] == 1:
                Hf = H[0:n_free,0:n_free]
                # print("first")
            else:
                Hf = H[1:(n_free+1),1:(n_free+1)]
                # print("second")
            
            Hfree = np.zeros((n_free,n_free), dtype=float)

            # print("*****Hf")
            # print(Hf)
            Hfree = np.linalg.cholesky(Hf).T

            nfactor+=1
        
        gnorm = np.linalg.norm(grad * free_v)
        if gnorm < minGrad:
            result = 5
            break

        grad_clamped = g + H @ (x * clamped)


        grad_clamped_all = grad_clamped[free_v > 0]
        # print("*****grad_clamped_all")
        # print(grad_clamped_all)
        grad_clamped_need = grad_clamped_all.reshape(grad_clamped_all.shape[0],1)

        x_all = x[free_v > 0]
        # print("*****x_all")
        # print(x_all)
        x_need = x_all.reshape(x_all.shape[0],1)

        search = np.zeros((2,1), dtype=float)

        # print("*****Hfree")
        # print(Hfree)

        # print("*****np.linalg.inv(Hfree)")
        # print(np.linalg.inv(Hfree))

        search_need = -np.linalg.inv(Hfree) @ (np.linalg.inv(Hfree.T) @ grad_clamped_need) - x_need

        if free_v[0,0] == 1 and free_v[1,0] == 1:
            search = search_need
        elif free_v[0,0] == 1:
            search[0,0] = search_need[0,0]
        elif free_v[1,0] == 1:
            search[1,0] = search_need[0,0]

        sdotg = (search * grad).sum()

        if sdotg >= 0:
            break


		#Armijo line search
        step = 1
        nstep = 0
        reach = x + step * search
        xc = clamp_to_limits(reach,params)
        vc = xc.T @ g + 0.5 * ( xc.T @ H @ xc)

        while (vc[0,0] - oldvalue[0,0])/(step * sdotg) < Armijo :
            step *= stepDec
            nstep+= 1
            reach = x + step * search
            xc = clamp_to_limits(reach,params)
            vc = xc.T @ g + 0.5 * ( xc.T @ H @ xc)
            if step < minStep :
                result = 2
                break
        x = xc
        value = vc

    return result, x, Hfree, free_v

def backward_pass(cx, cu, cxx, cxu, cuu, fx, fu, us, T, lamb, params):
    Vx = np.zeros((T+1, 6, 1), dtype=float)
    Vxx = np.zeros((T+1, 6, 6), dtype=float)
    K = np.zeros((T, 2, 6), dtype=float)
    k = np.zeros((T, 2, 1), dtype=float)

    Vx[T] = cx[T]
    Vxx[T] = cxx[T]

    Qx = np.zeros((6,1),dtype=float)
    Qu = np.zeros((2,1),dtype=float)
    Qxx = np.zeros((6,6),dtype=float)
    Qux = np.zeros((2,6),dtype=float)
    Quu = np.zeros((2,2), dtype=float)
    k_i = np.zeros((2,1), dtype=float)
    K_i = np.zeros((2,6), dtype=float)
    dV = np.zeros((2,1), dtype=float)

    for i in range(T-1,-1,-1):
        Qx = cx[i] + fx[i].T @ Vx[i+1]
        Qu = cu[i] + fu[i].T @ Vx[i+1]

        Qxx = cxx[i] + fx[i].T @ Vxx[i+1] @ fx[i]
        Qux = cxu[i].T + fu[i].T @ Vxx[i+1] @ fx[i]
        Quu = cuu[i] + fu[i].T @ Vxx[i+1] @ fu[i]

        Vxx_reg = Vxx[i+1]
        Qux_reg = cxu[i].T + fu[i].T @ Vxx_reg @ fx[i]
        # lambda1 = 1.0
        Eye2 = np.identity(2)
        QuuF = cuu[i] + fu[i].T @ Vxx_reg @ fu[i] + lamb * Eye2

        result, k_i, R, free_v = boxQP(QuuF, Qu, k[min(i+1,T-1)], params)

        if result < 1:
            print("@@@@@ result < 1")
            result = i
            break
        

        if free_v[0,0] == 1 and free_v[1,0] == 1:
            Lfree = -np.linalg.inv(R)@(np.linalg.inv(R.T) @ Qux_reg)
            K_i = Lfree
        elif free_v[0,0] == 1:
            Qux_reg_need = Qux_reg[0,:]
            Lfree = -np.linalg.inv(R)@(np.linalg.inv(R.T) @ Qux_reg_need)
            K_i[0,:] = Lfree
        elif free_v[1,0] == 1:
            Qux_reg_need = Qux_reg[1,:]
            Lfree = -np.linalg.inv(R)@(np.linalg.inv(R.T) @ Qux_reg_need)
            K_i[1,:] = Lfree

        dV[0,0] = k_i.T @ Qu
        dV[1,0] = 0.5 *(k_i.T @ Quu @ k_i)

        Vx[i] = Qx + K_i.T @ Quu @ k_i + K_i.T @ Qu + Qux.T @ k_i
        Vxx[i] = Qxx + K_i.T @ Quu @ K_i + K_i.T @ Qux + Qux.T @ K_i
        Vxx[i] = 0.5 * (Vxx[i] + Vxx[i].T)
    
        k[i] = k_i
        K[i] = K_i

    result = 0
    return result, Vx, Vxx, k , K, dV

def adjust_u(u,l,alpha):
    new_u = u
    new_u += alpha * l
    return new_u

def planner(model, params):
    x0 = np.zeros((6,1),dtype=float)
    xd = np.zeros((6,1),dtype=float)
    xd[0,0] = 1
    xd[1,0] = 1
    xd[2,0] = np.pi*3/2
    cost_s = 0.0

    T = 500  # horizon
    us = np.zeros((T, 2, 1), dtype=float)
    for i in range(T):
        us[i,0,0] = 1.0
        us[i,1,0] = 0.3

    xs = np.zeros((T+1,6,1),dtype=float)

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
    l= np.zeros((T, 2, 1), dtype=float)
    dV = np.zeros((2, 1), dtype=float)
    Alpha = np.array([1.0000, 0.5012, 0.2512, 0.1259, 0.0631, 0.0316, 0.0158, 0.0079, 0.0040, 0.0020, 0.0010])
    tolFun = 1e-6

    maxIter = 30
    dcost = 0.0
	# z = 0.0
    zMin = 0.0
	# expected = 0.0
    stop = False
    diverge = 0
    lamb = 1.0
    lambdaFactor = 1.6
    lambdaMax = 1e11
    lambdaMin = 1e-8
    dlambda = 1.0
    flgChange = True

    init_forward = True
    xs , us , cost_s = forward_pass(x0, us, T,  xs, L, init_forward, model, params)
    print("***** Initial cost : %.2f  "%(cost_s))

    x = xs
    u = us

    for iter in range(maxIter):
        if stop:
            break

        # STEP 1: Differentiate dynamics and cost along new trajectory
        # derivatives
        if flgChange :

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
                        pm[i,0] += eps
                        pp[i,0] = pm[i,0]
                        mm[i,0] -= eps
                        mp[i,0] = mm[i,0]
                        mp[j,0] += eps
                        pp[j,0] = mp[j,0]
                        mm[j,0] -= eps
                        pm[j,0] = mm[j,0]
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
                    for j in range(i,2):
                        pp = pm = mp = mm = us[t]
                        pm[i,0] += eps
                        pp[i,0] = pm[i,0]
                        mm[i,0] -= eps
                        mp[i,0] = mm[i,0]
                        mp[j,0] += eps
                        pp[j,0] = mp[j,0]
                        mm[j,0] -= eps
                        pm[j,0] = mm[j,0]
                        cuu[t,i,j] = cuu[t,j,i] = (cost(xs[t,:,0], pp[:,0]) - cost(xs[t,:,0], mp[:,0]) - cost(xs[t,:,0],pm[:,0]) + cost(xs[t,:,0],mm[:,0]))/(4*eps*eps)
            flgChange = False

        # STEP 2: Backward pass, compute optimal control law and cost-to-go  
        backPassDone = False
        while not backPassDone:
            diverge, Vx, Vxx, l , L, dV  = backward_pass(cx, cu, cxx, cxu, cuu, fx, fu, u, T, lamb, params)
            if diverge != 0:
                dlambda = max(dlambda * lambdaFactor, lambdaFactor)
                lamb = max(lamb * dlambda, lambdaMin)
                if lamb > lambdaMax:
                    break
                continue
            backPassDone = True

        # STEP 3: Forward pass / line-search to find new control sequence, trajectory, cost
        fwdPassDone = False
        if backPassDone:
            for i in range(Alpha.shape[0]):
                alpha = Alpha[i]
                xnew , unew, new_cost = forward_pass(x0, adjust_u(us,l,alpha), T, x, L, init_forward, model, params)
                
                print("***** Iter %d  cost : %.2f  "%(iter ,cost_s))
                
                dcost = cost_s - new_cost
                expected = -alpha * (dV[0,0] + alpha * dV[1,0])

                if expected > 0:
                    z = dcost/expected
                else:
                    z = np.sign(dcost)

                if z > zMin:
                    fwdPassDone = True
                    break
            if not fwdPassDone:
                alpha = 0.0

        # STEP 4: accept step (or not), print status
        if fwdPassDone:

            #decrease lambda   
            dlambda = min(dlambda / lambdaFactor, 1/lambdaFactor)
            if lamb > lambdaMin:
                templamb = 1
            else:
                templamb = 0
            lamb = lamb * dlambda * templamb           
            # accept changes
            us = unew
            xs = xnew
            cost_s = new_cost
            flgChange = True

            if dcost < tolFun:
                print("SUCCESS: cost change < tolFun")
                break
        else :
            dlambda = max(dlambda * lambdaFactor, lambdaFactor)
            lamb = max(lamb * dlambda, lambdaMin)
            if lamb > lambdaMax:
                break
    print(xs)
    print("@@@@@@@@@@@@@@@@@@@")
    print(us)

def control(X, U , params):
    action = np.array([15, 0.3])

    return action


def simulation():
    stream = open('/home/xingjiansheng/Workplace/project/MotionPlanner/Drift/ratel_psi/src/python/autonomous_simulation/car.yaml','r')
    params = yaml.load(stream)
    model = DynamicBicycleModel(params)
    renderer = _Renderer(params)

    state0 = np.zeros(6)
    action_hardcode = np.array([8.0, 0.2])
    iterations = 1500
    dt = 0.02

    # planner(model, params)

    for i in range(iterations):
        # print(state0)
        # if state0[5] > 12 :
        #     action_hardcode[0] = 0
        #     action_hardcode[1] = 0.3
        
        action_hardcode = control(state0, action_hardcode, params)
        
        state0 = model.state_transition(state0, action_hardcode, dt)
        # print("state0 , bate %.2f , r %.2f , vx %.2f "%(state0[3]/3.14*180 , state0[4], state0[5]))
        
        renderer.update(state0, action_hardcode)

def test_box_QP():
    stream = open('/home/xingjiansheng/Workplace/project/MotionPlanner/Drift/ratel_psi/src/python/autonomous_simulation/car.yaml','r')
    params = yaml.load(stream)
    g = np.array([[0],
                  [0]])
    H = np.array([[1 , 0],
                  [0 , 1]])
    H = H @ H.T

    x0 = np.array([[-1.5],
                    [-2]])
    result, x, Hfree, free_v = boxQP(H, g, x0, params)

    print("result :")
    print(result)
    print(" x : ")
    print(x)
    print(Hfree)
    print(free_v)

def test_planner():
    stream = open('/home/xingjiansheng/Workplace/project/MotionPlanner/Drift/ratel_psi/src/python/autonomous_simulation/car.yaml','r')
    params = yaml.load(stream)
    model = DynamicBicycleModel(params)
    planner(model, params)


if __name__ == "__main__":
    # simulation()
    # test_box_QP()
    test_planner()
