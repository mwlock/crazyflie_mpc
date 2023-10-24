import osqp
import numpy as np
import scipy as sp
from scipy import sparse

import control
import casadi as ca

import matplotlib.pyplot as plt

FREQ = 40
dt = 1/FREQ

Ac = ca.DM.zeros(7, 7)
Bc = ca.DM.zeros(7, 3)

Ac[0, 3] = 1
Ac[1, 4] = 1
Ac[2, 5] = 1

Ac[5, 5] = 1 # disturbance


Bc[3, 0] = 1
Bc[4, 1] = 1
Bc[5, 2] = 1

continous_system = control.StateSpace(Ac, Bc, ca.DM.eye(7), ca.DM.zeros(7, 3))
discrete_system = continous_system.sample(dt)


print(f"continous_system:\n {continous_system}")
print(f"discrete_system:\n {discrete_system}")

Ad = sparse.csc_matrix(discrete_system.A)
Bd = sparse.csc_matrix(discrete_system.B)

[nx, nu] = Bd.shape

print(f"nx: {nx}")
print(f"nu: {nu}")

# Constraints

x_max = 0.5
y_max = 0.5
z_min = 0
z_max = 3

vx_max = 0.5
vy_max = 0.5
vz_max = 1

fx_max = 0.5
fy_max = 0.5
fz_max = 1

umin = np.array([-fx_max, -fy_max, -fz_max])
umax = np.array([ fx_max,  fy_max,  fz_max])
xmin = np.array([-x_max, -y_max, z_min, -vx_max, -vy_max, -vz_max, -10])
xmax = np.array([ x_max,  y_max, z_max,  vx_max,  vy_max,  vz_max,  10])

# # Objective function
Q = sparse.diags([1., 1., 1., 0., 0., 0., 0.])
QN = Q
R = 0.1*sparse.eye(3)

# Initial and reference states
x0 = np.array([0., 0., 0., 0., 0., 0., -1])
xr = np.array([0., 0., 1., 0., 0., 0., 0.])

# Prediction horizon
N = 100

# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
# - quadratic objective
P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                       sparse.kron(sparse.eye(N), R)], format='csc')

# - linear objective
q = np.hstack([np.kron(np.ones(N), -Q@xr), -QN@xr, np.zeros(N*nu)])

# - linear dynamics
Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
Aeq = sparse.hstack([Ax, Bu])
leq = np.hstack([-x0, np.zeros(N*nx)])
ueq = leq

# - input and state constraints
Aineq = sparse.eye((N+1)*nx + N*nu)
lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])

# - OSQP constraints
A = sparse.vstack([Aeq, Aineq], format='csc')
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace
prob.setup(
    P, q, A, l, u, 
    warm_start=True,
    verbose=False
    )

# Simulate in closed loop
nsim = 1000
sim_results = np.zeros((nx, nsim))
for i in range(nsim):
    # Solve
    res = prob.solve()

    # Check solver status
    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')
    
    # Print solve time 
    print(f"Solve time : {res.info.solve_time}")
    print(f"solve freq {1/res.info.solve_time}")

    # Apply first control input to the plant
    ctrl = res.x[-N*nu:-(N-1)*nu]
    x0 = Ad@x0 + Bd@ctrl

    # Store results
    sim_results[:, i] = x0

    # Update initial state
    l[:nx] = -x0
    u[:nx] = -x0
    prob.update(l=l, u=u)

# Plot results
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(sim_results[0, :], label='x')
plt.plot(sim_results[1, :], label='y')
plt.plot(sim_results[2, :], label='z')
plt.plot(sim_results[6, :], label='disturbance')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(sim_results[3, :], label='vx')
plt.plot(sim_results[4, :], label='vy')
plt.plot(sim_results[5, :], label='vz')
plt.legend()
plt.show()
