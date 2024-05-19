import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

name = '3d_spring_single_body'

dt = 0.03
t = 10

import os
if not os.path.exists(name):
    os.makedirs(name)

# Тело
body_x = np.array([1.0, 1.0, 3.0])
body_v = np.array([0.1, 0.1, 0.3])
body_phi = np.array([0.0, 0.0, 0.0])
body_w = np.array([0.0, 0.0, 0.0])

# Пружина
spring_k = 1.0
spring_l = 3.0
spring_rel_shift = np.array([0.35, 0.35, 0.0])
fixed_point = np.array([0.0, 0.0, 0.0])

def dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def direction(x1, x2):
    a = x2 - x1
    a_len = dist(np.zeros(a.shape), a)
    return a / a_len

def spring(start, end, nodes, width, rotation=0):
    nodes = max(int(nodes), 1)
    spring_coords = np.zeros((3, nodes + 2))
    spring_coords[:,0], spring_coords[:,-1] = start, end

    length = dist(start, end)
    u_t = direction(start, end)
    u_n = direction(np.zeros(3), np.array([u_t[0], -u_t[1], rotation]))
    normal_dist = np.sqrt(max(0, width**2 - (length**2 / nodes**2))) / 2
    for i in range(1, nodes + 1):
        spring_coords[:,i] = (
            start
            + ((length * (2 * i - 1) * u_t) / (2 * nodes))
            + (normal_dist * (-1)**i * u_n))

    return spring_coords[0,:], spring_coords[1,:], spring_coords[2,:]

def draw_body(ax, pos, phi=np.array([0, 0, 0], dtype=float)):
    r = 0.5
    phi_vals = np.arange(1, 10, 2) * np.pi / 4
    Phi, Theta = np.meshgrid(phi_vals, phi_vals)

    x = np.cos(Phi) * np.sin(Theta) * r
    y = np.sin(Phi) * np.sin(Theta) * r
    z = np.cos(Theta) / np.sqrt(2) * r

    rot = R.from_rotvec(phi)

    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j], y[i][j], z[i][j] = rot.apply(np.array([x[i][j], y[i][j], z[i][j]])) + pos

    ax.plot_surface(x, y, z, color="r")

def draw_spring(ax, x, y):
    ax.plot(*spring(x, y, 6, 1.5, 0), color="g")

def draw_all(i):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    ax.set_aspect("equal")

    draw_body(ax, body_x)
    rot = R.from_rotvec(body_phi).apply(spring_rel_shift)
    draw_spring(ax, body_x + rot, fixed_point)

    plt.savefig(f'{name}/{i:03d}.png', dpi=300)
    plt.close()

# Симуляция
for time_i in range(int(t / dt)):
    # Обновление ускорения
    r_i = R.from_rotvec(body_phi).apply(spring_rel_shift)
    body_dist = dist(body_x + r_i, fixed_point)
    delta = body_dist - spring_l
    force = spring_k * delta * direction(body_x + r_i, fixed_point)
    body_a = force
    body_eps = np.cross(r_i, force)

    # Обновление скорости и координаты
    body_v += body_a * dt
    body_x += body_v * dt - body_a * dt * dt / 2

    body_w += body_eps * dt
    body_phi += body_w * dt - body_eps * dt * dt / 2

    draw_all(time_i)

import glob
import contextlib
from PIL import Image

# filepaths
fp_in = f"{name}/*.png"
fp_out = f"{name}/{name}.gif"

with contextlib.ExitStack() as stack:

    imgs = (stack.enter_context(Image.open(f))
            for f in sorted(glob.glob(fp_in)))
    img = next(imgs)

    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=30, loop=0)
