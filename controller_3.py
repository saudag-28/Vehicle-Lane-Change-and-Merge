import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

class MPC:

    def __init__(self, N, dt, L, d_min, Q, R, Qf, y_f, x_f, a_max, a_min, init_state, delta_max, delta_min):

        self.N = N
        self.dt = dt
        self.L = L
        self.y_f = y_f
        self.x_f = x_f
        self.d_min = d_min
        self.Q = np.diag([Q]*4)
        self.R = np.diag([R]*2)
        self.Qf = np.diag([Qf]*4)
        self.a_max = a_max
        self.a_min = a_min
        self.init_state = init_state
        self.delta_max = delta_max
        self.delta_min = delta_min

        self.psi = 0
        self.v = 5

        # self.constraint = 0
        self.cost = 0


    def create_dynamics(self, state, control):

        x, y, psi, v = state
        a, delta = control

        beta = np.arctan2(1.25 * np.tan(delta), self.L)

        xd = v * np.cos(beta + psi)
        yd = v * np.sin(beta + psi)
        psid = a
        vd = v * np.sin(beta)/(self.L/2)

        return np.array([xd, yd, psid, vd])
    
    def rk4(self, state, control):

        k1 = self.create_dynamics(state, control)
        k2 = self.create_dynamics(state + self.dt/2.0 * k1, control)
        k3 = self.create_dynamics(state + self.dt/2.0 * k2, control)
        k4 = self.create_dynamics(state + self.dt * k3, control)

        return state + self.dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
    
    def linear_model(self):

        A = np.array([[1, 0, -self.v*np.sin(self.psi)*self.dt, np.cos(self.psi)*self.dt],
                      [0, 1, self.v*np.cos(self.psi)*self.dt, np.sin(self.psi)*self.dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        
        B = np.array([[0, 0],
                      [0, 0],
                      [self.dt, 0],
                      [0, self.dt]])
        return A, B
    
    def obstacle_vehicles(self, state, control):
        states = [state]
        controls = [control]

        for i in range(self.N-1):

            next_states = self.rk4(states[-1], controls[-1])
            states.append(next_states)
            controls.append(control) 

        return np.array(states), np.array(controls)
    
    def _setup_problem(self, A, B, xic, xg, N_mpc):
        
        X = cp.Variable((4, N_mpc))
        U = cp.Variable((2, N_mpc-1))
        cost = 0

        # cost function
        for k in range(N_mpc-1):
            
            cost += 0.5 * cp.quad_form(X[:,k] - xg, self.Q)
            cost += 0.5 * cp.quad_form(U[:,k], self.R)
        
        cost += 0.5 * cp.quad_form(X[:,-1] - xg, self.Qf)

        # initial condition constraint - will keep updating
        constraints = [X[:,0] == xic]
        print(xic)

        # dynamics constraint
        for k in range(N_mpc-1):
            constraints += [X[:,k+1] == A @ X[:,k] + B @ U[:,k]]

        # control bound constraints
        for k in range(N_mpc-1):
            constraints += [U[0,k] <= self.a_max]
            constraints += [U[0,k] >= self.a_min]
            constraints += [U[1,k] <= self.delta_max]
            constraints += [U[1,k] >= self.delta_min]

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            raise ValueError("Solver did not converge!")

        X = X.value
        U = U.value

        return U[:,0]

    
    def plot(self, vehicles,  labels):

        fig, ax = plt.subplots()

        max_x = max(np.max(vehicle[:,0]) for vehicle in vehicles) + 10
        min_y = min(np.min(vehicle[:,1]) for vehicle in vehicles) - 10
        max_y = max(np.max(vehicle[:,1]) for vehicle in vehicles) + 10

        ax.set_xlim(0, max_x)
        ax.set_ylim(min_y, max_y)
        ax.grid(True)

        colors = ['blue', 'green']
        rects = []

        for idx, vehicle in enumerate(vehicles):
            width, height = 5.0, 2.0
            rect = Rectangle((vehicle[0, 0] - width / 2, vehicle[0, 1] - height / 2), width, height, fill=False, color=colors[idx])
            ax.add_patch(rect)
            rects.append(rect)

        def animate(i):
            for idx, rect in enumerate(rects):
                vehicle = vehicles[idx]
                rect.set_xy((vehicle[i, 0] - width / 2, vehicle[i, 1] - height / 2))
            return rects

        ani = FuncAnimation(fig, animate, frames=len(vehicles[0]), blit=True, interval=100, repeat=False)

        for i, label in enumerate(labels):
            rects[i].set_label(label)

        ax.legend()
        plt.show()

L = 2.5
d_min = 8
Q, R, Qf = 1, 0.1, 10
y_f = 0
x_f = 0
a_max = 1
a_min = -1
delta_max = 0.5
delta_min = -0.5
init_state = np.array([0, 5, 0, 5])  # x, y, psi, v - obstacle vehicle
init_control = np.array([0, 0])

# problem size
N = 100
# MPC window size
N_mpc = 20
# sim size
N_sim = N + 20
dt = 1.0

mpc = MPC(N, dt, L, d_min, Q, R, Qf, y_f, x_f, a_max, a_min, init_state, delta_max, delta_min)

# obstacle vehicle
Xsim_obs, Usim_obs = mpc.obstacle_vehicles(init_state, init_control)
# ref trajectory
Xref, Uref = mpc.obstacle_vehicles(np.array([0, 0, 0, 6]), np.array([0, 0]))

# ego vehicle
xi = np.array([0, -5, 0, 8])
xg = np.array([Xsim_obs[-1,0] - d_min, 5, 0, Xsim_obs[-1,3]])    # Goal state of ego vehicle is behind the obstacle vehicle

A, B = mpc.linear_model()

X_sim = [xi]
U_sim = []

for i in range (N_sim-1):

    # solving the optimization problem for the MPC window
    U_opt = mpc._setup_problem(A, B, X_sim[i], xg, N_mpc)
    U_sim.append(U_opt)

    # simulate one step
    X_next = mpc.rk4(X_sim[i], U_opt)
    X_sim.append(X_next)
    
X_sim = np.array(X_sim)

# print(X_sim)

mpc.plot([Xsim_obs, X_sim], ['Obstacle Vehicle','Ego Vehicle'])
