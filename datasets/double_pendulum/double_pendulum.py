import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

class DoublePendulum:
    def __init__(self, L1=1.0, L2=1.0, M1=1.0, M2=1.0, G=9.81):
        self.L1, self.L2 = L1, L2  # lengths of rods
        self.M1, self.M2 = M1, M2  # masses of bobs
        self.G = G  # gravitational acceleration
        
    def derivatives(self, state, t):
        theta1, omega1, theta2, omega2 = state
        
        c = np.cos(theta1 - theta2)
        s = np.sin(theta1 - theta2)
        
        theta1_dot = omega1
        theta2_dot = omega2
        
        # Derived from Lagrangian mechanics
        omega1_dot = (-self.G*(2*self.M1 + self.M2)*np.sin(theta1) - 
                     self.M2*self.G*np.sin(theta1 - 2*theta2) -
                     2*s*self.M2*(omega2**2*self.L2 + 
                     omega1**2*self.L1*c)) / (self.L1*(2*self.M1 + 
                     self.M2 - self.M2*np.cos(2*(theta1 - theta2))))
        
        omega2_dot = (2*s*(omega1**2*self.L1*(self.M1 + self.M2) +
                     self.G*(self.M1 + self.M2)*np.cos(theta1) +
                     omega2**2*self.L2*self.M2*c)) / (self.L2*(2*self.M1 +
                     self.M2 - self.M2*np.cos(2*(theta1 - theta2))))
        
        return theta1_dot, omega1_dot, theta2_dot, omega2_dot
    
    def simulate(self, initial_state, t_span, dt):
        t = np.arange(0, t_span, dt)
        solution = odeint(self.derivatives, initial_state, t)
        return t, solution
    
    def get_coordinates(self, theta1, theta2):
        x1 = self.L1 * np.sin(theta1)
        y1 = -self.L1 * np.cos(theta1)
        
        x2 = x1 + self.L2 * np.sin(theta2)
        y2 = y1 - self.L2 * np.cos(theta2)
        
        return x1, y1, x2, y2


def generate_random_initial_state():
    # Generate random initial angles between -π and π
    theta1 = np.random.uniform(-np.pi, np.pi)
    theta2 = np.random.uniform(-np.pi, np.pi)
    
    # Generate random initial angular velocities between -2 and 2
    omega1 = np.random.uniform(-2, 2)
    omega2 = np.random.uniform(-2, 2)
    
    return [theta1, omega1, theta2, omega2]