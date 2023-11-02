import numpy as np

T = 2  # time horizon
N = 2000  # total number of samples
pts = 20  # points in each trajectory
x_dim = 4  # feature dimension
y_dim = 1  # label dimension

t = np.linspace(0, T, pts).reshape((1, -1)).repeat(N, axis=0)  # time
x = np.full((N, pts, x_dim), 0.0)
y = np.full((N, pts, y_dim), 0.0)
mask = np.ones((N, pts))

rng = np.random.default_rng(0)  # rng with seed 0

x0 = 0 * rng.normal(0, 0.1, (N, 1))  # x0 start from 0
w1 = rng.normal(0.3, 0.1, (N, 1))  # different speeds of growth
w2 = rng.normal(0.7, 0.1, (N, 1))  # different speeds of growth
n1 = int(0.9 * N)
n2 = N - n1

tau = rng.exponential(1.0, (N, 1))
dt = t - tau
x[:n1, :, 0] = (x0 + (np.exp(w1 * dt) - w1 * dt - 1) * (dt > 0))[:n1]
x[n1:, :, 0] = (x0 + (np.exp(w2 * dt) - w2 * dt - 1) * (dt > 0))[n1:]

x[:n1, :, 1] = (x0 + w1 * dt * (dt > 0))[:n1]
x[n1:, :, 1] = (x0 + w2 * dt * (dt > 0))[n1:]

phase = rng.normal(0, 1, (N, 1))
x[:, :, 2] = np.sin(3 * t + phase)
x[:, :, 3] = np.cos(3 * t + phase)

x_max = 1.0
clip = x[:, :, 0] > x_max
x[clip, 0] = x_max

x[:, :, 0] += rng.normal(0, 0.01, (N, pts))
x[:, :, 1] += rng.normal(0, 0.1, (N, pts))
x[:, :, 2] += rng.normal(0, 0.1, (N, pts))
x[:, :, 3] += rng.normal(0, 0.1, (N, pts))

p = np.exp(-3 * np.square(x_max - x[:, :, 0]))
sample_y = rng.binomial(1, p)
y[:, :, 0] = sample_y

np.savez_compressed("synthetic_data.npz", t=t, x=x, y=y, mask=mask)
