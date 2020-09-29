import pf_filt as pf
import numpy as np
from numpy.random import uniform, randn
from numpy.linalg import norm
from filterpy.monte_carlo import systematic_resample
from matplotlib import pyplot as plt

N = 500
x_dim = (0, 20)
y_dim = (0, 20)
control_input = (1, 1)
std = (.1, .1)
dt = 1
landmarks = np.array([[-1, 0], [2, 3], [-1, 15]])
# Test Parameters
num_of_iterations = 20
sensor_std = .2

particles = pf.create_particles(N=N, x=x_dim, y=y_dim)
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
weights = np.zeros(N)

robot_position = np.array([0., 0.])
for iteration in range(num_of_iterations):
    robot_position += (1, 1)
    diff = landmarks - robot_position
    observation = (norm(diff, axis=1) +
                   (randn(len(landmarks)) * sensor_std))

    pf.predict(particles=particles, control_input=control_input, std=std, dt=dt)

    pf.update(particles=particles, weights=weights, observation=observation, sensor_std=sensor_std,
              landmarks=landmarks)

    # pf.particle_vis(particles, weights)

    mean = pf.estimate(particles, weights)
    p1 = plt.scatter(robot_position[0], robot_position[1], marker='+',
                     color='k', s=180, lw=3)
    p2 = plt.scatter(mean[0], mean[1], marker='s', color='r')

    # Initial Resample
    # if pf.effective_particles(weights) < N / 2:
    #     print('Resampling... %s' % iteration)
    #     indexes = systematic_resample(weights)
    #     pf.resample(particles, weights, indexes)

    pf.gaussian_process_reg(particles, weights)

    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1)
plt.show()
