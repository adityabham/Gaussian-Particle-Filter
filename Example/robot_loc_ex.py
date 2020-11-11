import pf as pf
import numpy as np
from numpy.linalg import norm
from filterpy.monte_carlo import systematic_resample
from matplotlib import pyplot as plt

N = 500  # number of particles
x_dim = (0, 20)
y_dim = (0, 20)
std = (.1, .1)
dt = 1  # time step
known_locations = np.array([[-1, 0], [2, 3], [-1, 15]])
# Test Parameters
num_of_iterations = 40

particles = pf.create_particles(N=N, x=x_dim, y=y_dim)
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
weights = np.zeros(N)

robot_position = np.array([0., 0.])
for iteration in range(num_of_iterations):
    # Increment robot position
    robot_position += (1, 1)

    # Distance from robot to each known location
    diff = known_locations - robot_position
    observation = (norm(diff, axis=1))

    pf.predict(particles=particles, std=std, dt=dt)

    pf.update(particles=particles, weights=weights, observation=observation,
              known_locations=known_locations)

    pf.gaussian_process_reg(particles=particles, weights=weights)

    # state estimation
    mean = pf.estimate(particles, weights)
    p1 = plt.scatter(robot_position[0], robot_position[1], marker='+', color='k', s=180, lw=3)
    p2 = plt.scatter(mean[0], mean[1], marker='s', color='r')

    # Resampling
    if pf.effective_particles(weights) < N / 2:
        print('Resampling... %s' % iteration)
        indexes = systematic_resample(weights)
        pf.resample(particles, weights, indexes)

    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

# Results
plt.legend([p1, p2], ['Actual', 'GPPF_matlab_NEU'], loc=4, numpoints=1)
plt.show()