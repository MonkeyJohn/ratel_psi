import numpy as np
import yaml
from scipy.integrate import solve_ivp
from tire_dynamics import BrushTireModel
from renderer import _Renderer

class DynamicBicycleModel(object):
    """
    Vehicle modeled as a three degrees of freedom dynamic bicycle model.
    """

    def __init__(self, params):
        """
        Initialize model parameters from dictionary format to instance
        variables.
        """
        # Vehicle parameters
        self.m = params['m']                # Mass
        self.g = params['g']                # Mass
        self.L_f = params['L_f']            # CoG to front axle length
        self.L_r = params['L_r']            # CoG to rear axle length
        self.L = self.L_f + self.L_r        # Front to rear axle length
        self.load_f = self.m * self.g * self.L_r / self.L      # Load on front axle
        self.load_r = self.m * self.g * self.L_f / self.L      # Load on rear axle

        # Wheel parameters
        self.C_x = params['C_x']            # Longitudinal stiffness
        self.C_alpha = params['C_alpha']    # Cornering stiffness
        self.I_z = params['I_z']            # Moment of inertia

        # Static and kinetic coefficients of friction
        self.mu_static = params['mu_static']
        self.mu_slide = params['mu_slide']

        self.tire = BrushTireModel()
        # Slip ratio
        # self.kappa = None


    def state_transition(self, X, U, dt):
        """
        Update state after some timestep.
        """
        t = np.array([0, dt])
        X_new = solve_ivp(
            fun=(lambda t, X: self._dynamics(X, t, U)),
            t_span=t, y0=X, atol=1e-5)
        return X_new.y[:,-1]


    def _dynamics(self, X, t, U):
        """
        Use dynamics model to compute X_dot from X, U.
        """
        pos_x = X[0]
        pos_y = X[1]
        pos_yaw = self.wraptopi(X[2])
        v_x = X[3]
        v_y = X[4]
        yaw_rate = X[5]
        cmd_vx = U[0]
        delta = U[1]

        # Tire slip angle (zero when stationary)
        if np.abs(v_x) < 0.001 and np.abs(v_y) < 0.001:
            alpha_f = 0
            alpha_r = 0
        elif np.abs(v_x) < 0.001:
            alpha_f = np.pi / 2 * np.sign(v_y) - delta
            alpha_r = np.pi / 2 * np.sign(v_y)
        elif v_x < 0:
            alpha_f = np.arctan((v_y + self.L_f*yaw_rate)/np.abs(v_x)) + delta
            alpha_r = np.arctan((v_y - self.L_r*yaw_rate)/np.abs(v_x))
        else:
            alpha_f = np.arctan((v_y + self.L_f*yaw_rate)/np.abs(v_x)) - delta
            alpha_r = np.arctan((v_y - self.L_r*yaw_rate)/np.abs(v_x))
        
        alpha_f = self.wraptopi(alpha_f)
        alpha_r = self.wraptopi(alpha_r)

        # Compute forces on tires using brush tire model
        F_xf, F_yf = self.tire.tire_dynamics(v_x, v_x, self.mu_static, self.mu_slide, self.load_f, self.C_x, self.C_alpha, alpha_f)
        F_xr, F_yr = self.tire.tire_dynamics(v_x, cmd_vx, self.mu_static, self.mu_slide, self.load_r, self.C_x, self.C_alpha, alpha_r)

        # Find dX
        T_z = self.L_f * F_yf * np.cos(delta) - self.L_r * F_yr
        ma_x = F_xr - F_yf * np.sin(delta)
        ma_y = F_yf * np.cos(delta) + F_yr

        # Acceleration with damping
        # yaw_rate_dot = T_z/self.I_z - 0.02*yaw_rate
        # v_x_dot = ma_x/self.m + yaw_rate*v_y - 0.025*v_x
        # v_y_dot = ma_y/self.m - yaw_rate*v_x - 0.025*v_y

        yaw_rate_dot = T_z/self.I_z
        v_x_dot = ma_x/self.m + yaw_rate*v_y
        v_y_dot = ma_y/self.m - yaw_rate*v_x

        # Translate to inertial frame
        v = np.sqrt(v_x**2 + v_y**2)

        # if np.abs(v_x) < 0.001 and np.abs(v_y) < 0.001:
        #     beta = 0
        # elif np.abs(v_x) < 0.001:
        #     beta = np.pi / 2 * np.sign(v_y)
        # elif v_x < 0 and np.abs(v_y) < 0.001:
        #     beta = np.pi
        # elif v_x < 0:
        #     beta = np.sign(v_y) * np.pi - np.atan(v_y / abs(v_x))
        # else:
        #     beta = np.atan(v_y / abs(v_x))
        beta = np.arctan2(v_y,v_x)
        pos_x_dot = v*np.cos(beta+pos_yaw)
        pos_y_dot = v*np.sin(beta+pos_yaw)

        X_dot = np.zeros(6)
        X_dot[0] = pos_x_dot
        X_dot[1] = pos_y_dot
        X_dot[2] = yaw_rate
        X_dot[3] = v_x_dot
        X_dot[4] = v_y_dot
        X_dot[5] = yaw_rate_dot

        return X_dot

    def wraptopi(self, val):
        """
        Wrap radian value to the interval [-pi, pi].
        """
        pi = np.pi
        val = val - 2*pi*np.floor((val+pi)/(2*pi))
        return val

    def inc_steer_test(self,params):

        renderer = _Renderer(params)

        state0 = np.zeros(6)
        action_hardcode = np.array([8.0, 0.2])

        iterations = 1500
        dt = 0.02

        for i in range(iterations):
            state0 = self.state_transition(state0, action_hardcode, dt)
            renderer.update(state0, action_hardcode)


# if __name__ == "__main__":
#     stream = open('/home/xingjiansheng/Documents/src/workplace_xjs/drift/ratel_psi/src/python/autonomous_simulation/car.yaml','r')
#     params = yaml.load(stream)
#     car_dyn = DynamicBicycleModel(params)
#     car_dyn.inc_steer_test(params)