from control import dlqr
import random
import numpy as np
import os

y_max = 10
v_max = 10
f_max = 10
g_y = 1
g_v = 1
g_f = 1
sigma_d = 0.1
A = np.array([[1,1],[0,1]])
B = np.array([[0],[1]])
B_w = np.array([[0],[1]])
Q = np.diag([g_y/(y_max**2),g_v/(v_max**2)])
R = np.array(g_f/(f_max**2))
out = dlqr(A,B,Q,R)
K = out[0]
S = out[1]

def timestep(x,K,w,A,B,B_w):
    u = -K@x
    return A@x + B@u + B_w@w

def simulate(x_0,n_steps,noise=True):
    trajectory = [x_0]
    x = x_0
    for n in range(n_steps):
        if noise:
            random.seed()
            w = np.array([[random.gauss(sigma=sigma_d)]])
        else:
            w = np.array([[0]])
        x = timestep(x,K,w,A,B,B_w)
        trajectory.append(x)
    return trajectory

if __name__ == "__main__":
    x_0 = np.array([[10],[0]])
    n_steps = 20
    trajectory = simulate(x_0,n_steps,noise=False)
    try:
        os.makedirs("numberline/results")
    except FileExistsError:
        pass

    filename = "numberline/results/trajectory.txt"
    with open(filename,"w") as results_file:
        results_file.write("y, v\n")
        for x in trajectory:
            results_file.write(f"{x[0,0]}, {x[1,0]}\n")