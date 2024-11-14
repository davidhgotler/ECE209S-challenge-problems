import numpy as np
import matplotlib.pyplot as plt

# Define constants
dt = 1.0  # time step
process_variance = 1e-3  # process noise variance
measurement_variance = 1.0  # measurement noise variance

# System dynamics
F = np.array([[1, dt], [0, 1]])  # state transition matrix
Q = process_variance * np.eye(2)  # process noise covariance
H = np.array([[1, 0]])            # observation matrix
R = np.array([[measurement_variance]])  # measurement noise covariance

# Initial estimates
x_est = np.array([[0], [1]])  # initial position=0, velocity=1
P_est = np.eye(2)  # initial estimate covariance

# Simulate the particle motion with measurements
num_steps = 50
true_positions = []
measurements = []
estimates = []

for t in range(num_steps):
    # True system evolution (assuming some process noise)
    w = np.random.multivariate_normal([0, 0], Q).reshape(2, 1)
    x_true = F @ x_est + w
    true_positions.append(x_true[0, 0])

    # Measurement with noise
    v = np.random.normal(0, np.sqrt(measurement_variance))
    z = H @ x_true + v
    measurements.append(z[0, 0])

    # Kalman filter prediction
    x_pred = F @ x_est
    P_pred = F @ P_est @ F.T + Q

    # Kalman gain
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    # Measurement update
    y = z - H @ x_pred
    x_est = x_pred + K @ y
    P_est = (np.eye(2) - K @ H) @ P_pred

    # Store estimates
    estimates.append(x_est[0, 0])

# Plot results
plt.plot(true_positions, label="True Position")
plt.plot(measurements, 'o', label="Measurements", markersize=4)
plt.plot(estimates, label="Kalman Filter Estimate")
plt.xlabel("Time Step")
plt.ylabel("Position")
plt.legend()
plt.show()
