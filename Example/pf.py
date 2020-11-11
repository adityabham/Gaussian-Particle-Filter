import numpy as np
from numpy.random import uniform, randn
from numpy.linalg import norm
import scipy.stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def create_particles(N, x, y):
    # Create a set of Particles

    particles = np.empty((N, 2))
    particles[:, 0] = uniform(x[0], x[1], size=N)
    particles[:, 1] = uniform(y[0], y[1], size=N)
    return particles


def predict(particles, std, dt):
    # Propagate particles forward one time step accounting for standard deviation

    x = dt + (randn(len(particles)) * std[0])
    y = dt + (randn(len(particles)) * std[1])

    particles[:, 0] += x
    particles[:, 1] += y


def update(particles, weights, observation, known_locations):
    # Update particle weights based on observations

    weights.fill(1.)

    for i, known_locations in enumerate(known_locations):
        diff = norm(particles[:, 0:2] - known_locations, axis=1)
        pdf_in = observation[i]
        pdfs = scipy.stats.norm(diff).pdf(pdf_in)
        weights *= pdfs

    weights += 1.e-300
    weights /= sum(weights)


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
    # GP regression
    kernel = 1.0 * RBF(1.0)
    gp = GaussianProcessRegressor(kernel=kernel, random_state=0)
    gp.fit(particles, weights)

    sampled_weights_array = gp.sample_y(particles)
    gp_weights = []
    for item in sampled_weights_array:
        gp_weights.append(item[0])

    weights[:] = gp_weights[:]

