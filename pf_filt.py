import numpy as np
from numpy.random import uniform, randn
from numpy.linalg import norm
import scipy.stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_particles(N, x, y):
    particles = np.empty((N, 2))
    particles[:, 0] = uniform(x[0], x[1], size=N)
    particles[:, 1] = uniform(y[0], y[1], size=N)
    return particles


def predict(particles, control_input, std, dt):
    num_of_particles = len(particles)

    xdist = (control_input[0] * dt) + (randn(num_of_particles) * std[0])
    ydist = (control_input[1] * dt) + (randn(num_of_particles) * std[1])

    particles[:, 0] += xdist
    particles[:, 1] += ydist


def update(particles, weights, observation, sensor_std, landmarks):
    weights.fill(1.)

    for i, landmark in enumerate(landmarks):
        distance = norm(particles[:, 0:2] - landmark, axis=1)
        pdf_in = observation[i]
        pdfs = scipy.stats.norm(distance, sensor_std).pdf(pdf_in)
        weights *= pdfs

    weights += 1.e-300
    weights /= sum(weights)  # normalize


def effective_particles(gp_weights):
    return 1. / np.sum(np.square(gp_weights))


def estimate(particles, weights):
    mean = np.average(particles[:], weights=weights, axis=0)
    return mean


def resample(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights.fill(1.0 / len(weights))


def gaussian_process_reg(particles, weights):
    kernel = 1.0 * RBF(1.0)
    gp = GaussianProcessRegressor(kernel=kernel, random_state=0)
    gp.fit(particles, weights)

    sampled_weights_array = gp.sample_y(particles)
    gp_weights = []
    for item in sampled_weights_array:
        gp_weights.append(item[0])

    weights[:] = gp_weights[:]


def particle_vis(particles, weights):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(particles[:, 0], particles[:, 1], weights, marker='o', color='orange')

    ax.set_xlabel('Part X')
    ax.set_ylabel('Part Y')
    ax.set_zlabel('Weights')

    plt.show()
