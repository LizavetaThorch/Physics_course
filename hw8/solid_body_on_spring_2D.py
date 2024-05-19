import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations

name = '2d_spring_single_body'

dt = 0.03
t = 10

import os
if not os.path.exists(name):
    os.makedirs(name)

# Тело
body_x = np.array([0.0, 0.0])
body_v = np.array([1.0, 0.0]) / 3
body_phi = np.array([0.0])
body_w = np.array([0.0])

# Пружина
spring_k = 1.0
spring_l = 3.1
spring_rel_shift_body = np.array([0.5, 0.5])
spring_other_end = np.array([3.0, 3.0])

def dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def direction(x1, x2):
    a = x2 - x1
    a_len = dist(np.zeros(a.shape), a)
    return a / a_len

def spring(start, end, nodes, width):
    nodes = max(int(nodes), 1)
    spring_coords = np.zeros((2, nodes + 2))
    spring_coords[:,0], spring_coords[:,-1] = start, end

    length = dist(start, end)
    u_t = direction(start, end)
    u_n = np.array([u_t[1], -u_t[0]])  # corrected u_n direction
    normal_dist = np.sqrt(max(0, width**2 - (length**2 / nodes**2))) / 2
    for i in range(1, nodes + 1):
        spring_coords[:,i] = (
            start
            + ((length * (2 * i - 1) * u_t) / (2 * nodes))
            + (normal_dist * (-1)**i * u_n))

    return spring_coords[0,:], spring_coords[1,:]

def rotate2d(coordinates, angle):
    x, y = coordinates
    newx = float(x * np.cos(angle[0]) - y * np.sin(angle[0]))
    newy = float(x * np.sin(angle[0]) + y * np.cos(angle[0]))
    return np.array([newx, newy])

import matplotlib.patches as patches

r = 0.5

def draw_body(ax, pos, phi=np.array([0], dtype=float)):
    rect = patches.Rectangle(
        (-r + pos[0], -r + pos[1]), 2 * r, 2 * r,
        angle=phi[0] / np.pi * 180, rotation_point='center', color="r")

    ax.add_patch(rect)

def draw_spring(ax, x, y):
    ax.plot(*spring(x, y, 6, 1), color="g")

def draw_all(i):
    fig, ax = plt.subplots()
    ax.set_xlim([-1, 11])
    ax.set_ylim([-1, 11])
    ax.set_aspect("equal")

    r_i = rotate2d(spring_rel_shift_body, body_phi)
    draw_spring(ax, body_x + r_i, spring_other_end)
    draw_body(ax, body_x, body_phi)

    plt.savefig(f'{name}/{i:03d}.png', dpi=300)
    plt.close()

# Симуляция
for time_i in range(int(t / dt)):
    # Обновляем ускорения
    body_a = np.zeros(body_x.shape, float)
    body_eps = np.zeros(body_phi.shape, float)

    r_i = rotate2d(spring_rel_shift_body, body_phi)
    body_dist = dist(body_x + r_i, spring_other_end)
    if body_dist >= 1e-4:
        delta = body_dist - spring_l
        force = spring_k * delta * direction(body_x + r_i, spring_other_end)
        body_a += force
        body_eps += np.cross(r_i, force) / r

    # Обновляем скорости и координаты
    body_v += body_a * dt
    # Предполагаем равноускоренное движение, чтобы повысить точность
    body_x += body_v * dt - body_a * dt * dt / 2

    body_w += body_eps * dt
    # Предполагаем равноускоренное движение, чтобы повысить точность
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

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=30, loop=0)
