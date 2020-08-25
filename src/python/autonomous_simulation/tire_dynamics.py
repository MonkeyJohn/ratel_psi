import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class BrushTireModel():
    """
    Use a dynamic bicycle model with a brush tire model for tire dynamics.
    """

    # def __init__(self, params, mu_s, mu_k):
    def tire_dynamics(self, v_x, v_x_wheel, mu_static, mu_slide, Fz, C_x, C_alpha, alpha):
        
        # Find longitudinal wheel slip K (kappa)
        if (np.abs(v_x_wheel-v_x) < 0.001 or
                (np.abs(v_x_wheel) < 0.001 and np.abs(v_x) < 0.001)):
            K = 0
        # Infinite slip, longitudinal saturation
        elif abs(v_x) < 0.001:
            Fx = np.sign(v_x_wheel) * mu_static * Fz
            Fy = 0
            return Fx, Fy
        else:
            K = (v_x_wheel - v_x)/np.abs(v_x)

        # Instead of avoiding -1, now look for positive equivalent
        if K < 0:
            spin_dir = -1
            K = np.abs(K)
        else:
            spin_dir = 1

        # alpha > pi/2 is invalid because of the use of tan(). Since
        # alpha > pi/2 means vehicle moving backwards, Fy's sign has
        # to be reversed, hence we multiply by sign(alpha)
        if abs(alpha) > np.pi/2:
            alpha = (np.pi-abs(alpha))*np.sign(alpha)

        # Compute combined slip value gamma
        gamma = np.sqrt( C_x**2 * (K/(1+K))**2 + C_alpha**2 * (np.tan(alpha)/(1+K))**2 )

        if gamma <= 3 * mu_static * Fz:
            F = gamma - 1 / (3 * mu_static * Fz) * (2 - mu_slide / mu_static)*gamma**2 + \
                    1 / ( 9 * mu_static**2 * Fz**2) * (1 - (2/3) * (mu_slide / mu_static)) * gamma**3
        else:
        # more accurate modeling with peak friction value
            F = mu_slide * Fz

        if gamma == 0:
            Fx = 0
            Fy = 0
        else:
            Fx = C_x / gamma * ( K/(1 + K)) * F * spin_dir
            Fy = -C_alpha / gamma * (np.tan(alpha)/(1 + K)) * F

        return Fx, Fy
    
    def test(self):
        m = 2.35           # mass (kg)
        L = 0.257          # wheelbase (m)
        C_alpha = 197.0      # laternal stiffness
        C_x = 116.0          # longitude stiffness
        Iz = 0.025 # rotation inertia
        g = 9.81

        b = 0.14328        # CoG to rear axle
        a = L-b            # CoG to front axle

        G_front = m*g*b/L   # calculated load or specify front rear load directly
        G_rear = m*g*a/L
        mu_static = 1.31
        mu_slide = 0.5

        v_x = 1
        v_x_wheel_cmd =[]
        Fx =[]
        Fy =[]
        K = []
        alpha = []

        # for i in range(200):
        #     v_x_wheel_cmd.append(0.01 * i)
        #     fx , fy = self.tire_dynamics(v_x, v_x_wheel_cmd[i], mu_static, mu_slide, G_rear, C_x, C_alpha, 0)
        #     Fx.append(fx)
        #     Fy.append(fy)
        #     k = (v_x_wheel_cmd[i] - v_x) / np.abs(v_x)
        #     K.append(k)
        # plt.plot(K, Fx, ".r")

        for i in range(200):
            alpha.append(-1 + 0.01 * i)
            fx , fy = self.tire_dynamics(v_x, 1.5, mu_static, mu_slide, G_rear, C_x, C_alpha, alpha[i])
            Fx.append(fx)
            Fy.append(fy)
        plt.plot(alpha, Fy, ".r")
        plt.plot(alpha, Fx, ".b")

        # plt.plot(Fx, Fy, ".y")
        
        plt.pause(0.01)
        plt.show()

# if __name__ == "__main__":
#     tiremodel = BrushTireModel()
#     tiremodel.test()