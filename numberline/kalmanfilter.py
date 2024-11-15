import numpy as np
import matplotlib.pyplot as plt

# Define constants
dt = 1  # time step
var_process_y = 0.1     # position process noise variance
var_process_v = 0.1   # velocity process noise variance
var_meas_y = 1          # position measurement noise variance
var_meas_v = 0.01       # velocity measurement noise variance
var_meas_yv = 0     # correlated measurement noise variance

# System dynamics
F = np.array([[1, dt], [0, 1]])             # state transition matrix
Q = np.diag([var_process_y,var_process_v])  # process noise covariance
H = np.eye(2)                               # observation matrix
R = np.array([[var_meas_y,var_meas_yv],[var_meas_yv,var_meas_v]])   # non-correlated measurement noise covariance

def run_sim(num_steps):
    '''Simulate the particle motion with measurements'''
    
    # Initialize
    X = np.array([[0], [1]])  # initial position=0, velocity=1
    X_list = [X]

    
    rng_w = np.random.default_rng()
    rng_v = np.random.default_rng()

    # Velocity-dependent noise
    Q = np.diag([var_process_y,var_process_v])*X[1,0]**2
    R = np.array([[var_meas_y,var_meas_yv],[var_meas_yv,var_meas_v]])*X[1,0]**2

    # initial measurement
    v = np.random.multivariate_normal([0, 0], R).reshape(2, 1)
    z = H @ X + v
    z_list = [z]
    for t in range(1,num_steps+1):
        # Velocity-dependent noise
        Q = np.diag([var_process_y,var_process_v])*X[1,0]**2
        R = np.array([[var_meas_y,var_meas_yv],[var_meas_yv,var_meas_v]])*X[1,0]**2
        # System synamics evolution (w/ some process noise)
        w = rng_w.multivariate_normal([0, 0], Q).reshape(2, 1)
        X = F @ X + w
        X_list.append(X)

        # Measurement with noise
        v = rng_v.multivariate_normal([0, 0], R).reshape(2, 1)
        z = H @ X + v
        z_list.append(z)
    return np.array(X_list),np.array(z_list)

def kalman_filter(z_arr):
    # # Initialize
    X = z_arr[0]
    P = 100*np.eye(2)  # initial estimate covariance
    X_est_list = []
    P_est_list = []
    for z in z_arr:
        # Kalman filter prediction
        # Velocity-dependent noise estimate
        Q = np.diag([var_process_y,var_process_v])*X[1,0]**2
        R = np.array([[var_meas_y,var_meas_yv],[var_meas_yv,var_meas_v]])*X[1,0]**2
        X = F @ X
        P = F @ P @ F.T + Q

        # Measurement update
        # Kalman gain
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        y = z - H @ X
        X = X + K @ y
        P = (np.eye(2) - K @ H) @ P

        # Store estimates
        X_est_list.append(X)
        P_est_list.append(P)
    return np.array(X_est_list)

def stats_sim(num_runs,num_steps):
    X_n_list     = []
    z_n_list     = []
    X_est_n_list = []
    for n in range(num_runs):
        X_n,z_n = run_sim(num_steps)
        X_est_n = kalman_filter(z_n)
        X_n_list.append(X_n)
        z_n_list.append(z_n)
        X_est_n_list.append(X_est_n)
    return np.array(X_n_list),np.array(z_n_list),np.array(X_est_n_list)

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

def plot_stats(X_n_arr,z_n_arr,X_est_n_arr,fig=None):
    n = len(X_n_arr)
    t = np.arange(X_n_arr.shape[1])
    # Get statistics
    X_mean = np.mean(X_n_arr,axis=0)
    X_var = np.var(X_n_arr,axis=0)
    z_mean = np.mean(z_n_arr,axis=0)
    z_var = np.var(z_n_arr,axis=0)
    X_est_mean = np.mean(X_est_n_arr,axis=0)
    X_est_var = np.var(X_est_n_arr,axis=0)

    if type(fig) is None:
        fig,(y_ax,v_ax) = plt.subplots(1,2,figsize=(8,4))
    else:
        y_ax,v_ax = fig.axes
    # position plot
    # Plot means
    y_ax.plot(t,X_mean[:,0,0], label=r"$\bar{y}_{true}$",color='green')
    y_ax.errorbar(t,z_mean[:,0,0],yerr=np.sqrt(z_var[:,0,0]),fmt ='o', label=r"$\bar{z}_y$", capsize=5,markersize=4,color='black')
    y_ax.plot(t,X_est_mean[:,0,0], label=r"$\bar{y}_{Kalman}$",color='blue')
    # Plot variances
    y_ax.fill_between(t,X_mean[:,0,0]+np.sqrt(X_var[:,0,0]),X_mean[:,0,0]-np.sqrt(X_var[:,0,0]),color='green',alpha=0.2)
    y_ax.fill_between(t,X_est_mean[:,0,0]+np.sqrt(X_est_var[:,0,0]),X_est_mean[:,0,0]-np.sqrt(X_est_var[:,0,0]),color='blue',alpha=0.2)
    y_ax.set_xlabel("$t$")
    y_ax.set_xlim(0,len(X_mean)-1)
    y_ax.set_ylabel("Position")
    y_ax.legend()
    # velocity plot
    # Plot means
    v_ax.plot(t,X_mean[:,1,0], label=r"$\bar{v}_{true}$",color='green')
    v_ax.errorbar(t,z_mean[:,1,0],yerr=np.sqrt(z_var[:,1,0]),fmt ='o', label=r"$\bar{z}_v$", capsize=4,markersize=4,color='black')
    v_ax.plot(t,X_est_mean[:,1,0], label=r"$\bar{v}_{Kalman}$",color='blue')
    # Plot variances
    v_ax.fill_between(t,X_mean[:,1,0]+np.sqrt(X_var[:,1,0]),X_mean[:,1,0]-np.sqrt(X_var[:,1,0]),color='green',alpha=0.2)
    v_ax.fill_between(t,X_est_mean[:,1,0]+np.sqrt(X_est_var[:,1,0]),X_est_mean[:,1,0]-np.sqrt(X_est_var[:,1,0]),color='blue',alpha=0.2)
    # v_ax.plot(t,np.max(X_n_arr,axis=0)[:,1,0],'--',label='min/max $v_{true}$',color='black')
    # v_ax.plot(t,np.min(X_n_arr,axis=0)[:,1,0],'--',color='black')
    # v_ax.plot(t,np.max(X_est_n_arr,axis=0)[:,1,0],':',label='min/max $v_{Kalman}$',color='black')
    # v_ax.plot(t,np.min(X_est_n_arr,axis=0)[:,1,0],':',color='darkblue')

    v_ax.set_xlabel("$t$")
    v_ax.set_xlim(0,len(X_mean)-1)
    v_ax.set_ylabel("Velocity")
    # v_ax.set_ylim(1-.2,1+.2)
    # v_ax.set_ylim(1-2*np.sqrt(var_meas_v),1+2*np.sqrt(var_meas_v))
    v_ax.legend()

    fig.suptitle(f"$n = {n}$ trials")
    # fig.tight_layout()
    # plt.show()
    return fig
    fig.savefig(f'numberline/results/pos_vel_stats_n={n}',dpi=300)

if __name__ == "__main__":
    # Simulate 1 run
    # X_arr,z_arr = run_sim(20)
    # X_est_arr = kalman_filter(z_arr)
    # plot_result(X_arr,z_arr,X_est_arr)

    # Simulate at different statistics levels and plot
    fig = plt.figure(layout='constrained', figsize=(20,8))
    subfigs = fig.subfigures(2, 2,wspace=0.05,hspace=0.05)
    subfigs = [subfigs[i][j] for i in range(2) for j in range(2)]
    for n,fig_n in zip([10,100,1000,10000],subfigs):
        print(f'simulating for n = {n} trials')
        X_n_arr,z_n_arr,X_est_n_arr = stats_sim(n,25)
        print('plotting statistics')
        axes = fig_n.subplots(1,2)
        plot_stats(X_n_arr,z_n_arr,X_est_n_arr,fig=fig_n)
    fig.suptitle(f"Kalman Filter Estimate - process noise: $q_y^2={var_process_y}v^2, q_v^2={var_process_v}v^2,q_{{yv}}=0$, sensor noise: $r_y^2={var_meas_y}v^2,r_v^2={var_meas_v}v^2,r_{{yv}}={var_meas_yv}$",fontsize='xx-large')
    fig.savefig(f'numberline/results/pos_vel_stats_plot_vel-dep-noise_no-min-max',dpi=300)
    # plt.show()