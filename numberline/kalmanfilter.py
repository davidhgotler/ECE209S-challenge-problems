import numpy as np
import matplotlib.pyplot as plt

# Define constants
dt = 1  # time step
var_process = 0.001   # process noise variance
var_meas_y = 10.0   # position measurement noise variance
var_meas_v = 0.001    # velocity measurement noise variance

# System dynamics
F = np.array([[1, dt], [0, 1]])             # state transition matrix
Q = var_process * np.array([[0,0],[0,1]])   # process noise covariance
H = np.eye(2)                               # observation matrix
R = np.diag([var_meas_y,var_meas_v])        # non-correlated measurement noise covariance
print(R)
def run_sim(num_steps=50):
    '''Simulate the particle motion with measurements'''
    # Initialize
    X = np.array([[0], [1]])  # initial position=0, velocity=1
    X_list = [X]
    # initial measurement
    v = np.random.multivariate_normal([0, 0], R).reshape(2, 1)
    z = H @ X + v
    z_list = [z]
    
    for t in range(1,num_steps+1):
        # System synamics evolution (w/ some process noise)
        w = np.random.multivariate_normal([0, 0], Q).reshape(2, 1)
        X = F @ X + w
        X_list.append(X)

        # Measurement with noise
        v = np.random.multivariate_normal([0, 0], R).reshape(2, 1)
        z = H @ X + v
        z_list.append(z)
    return np.array(X_list),np.array(z_list)

def kalman_filter(z_arr):
    # # Initialize
    X_est = z_arr[0,:,:]
    P_est = 10*np.eye(2)  # initial estimate covariance
    X_est_list = [X_est]
    for z in z_arr[1:,:,:]:

        # Kalman filter prediction
        x_pred = F @ X_est
        P_pred = F @ P_est @ F.T + Q

        # Kalman gain
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Measurement update
        y = z - H @ x_pred
        X_est = x_pred + K @ y
        P_est = (np.eye(2) - K @ H) @ P_pred

        # Store estimates
        X_est_list.append(X_est)
    return np.array(X_est_list)

def stats_sim(num_runs=100,num_steps=50):
    pass

def plot_result(X_arr,z_arr,X_est_arr):
    # Plot results
    fig,(y_ax,v_ax) = plt.subplots(1,2,figsize=(8,4))
    y_ax.plot(X_arr[:,0,:], label="$y_{true}$")
    y_ax.plot(z_arr[:,0,:], 'o', label="$z_y$", markersize=4)
    y_ax.plot(X_est_arr[:,0,:], label="$y_{Kalman}$")
    y_ax.set_xlabel("$t$")
    y_ax.set_xlim(0,len(X_arr)-1)
    y_ax.set_ylabel("position")
    y_ax.legend()

    v_ax.plot(X_arr[:,1,:], label="$v_{true}$")
    v_ax.plot(z_arr[:,1,:], 'o', label="$z_v$", markersize=4)
    v_ax.plot(X_est_arr[:,1,:], label="$v_{Kalman}$")
    v_ax.set_xlabel("$t$")
    v_ax.set_xlim(0,len(X_arr)-1)
    v_ax.set_ylabel("velocity")
    v_ax.legend()

    fig.suptitle("Kalman Filter Estimate")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    X_arr,z_arr = run_sim(100)
    X_est_arr = kalman_filter(z_arr)
    plot_result(X_arr,z_arr,X_est_arr)
