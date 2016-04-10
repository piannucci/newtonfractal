#!/usr/bin/env python
#
# Newton's algorithm on a function f:
#   next(x) = x - f(x) / f'(x)
#
# Suppose we near enough to a zero x0 that we can treat f' and f'' as constant,
# then
#   next(x0 + dx) = x0 + dx - f(x0 + dx) / f'(x0 + dx)
#       ~= x0 + dx^2 f''(x0) / f'(x0)
# so if the current distance from x to the zero is d, then the next distance
# is cd^2 where c = |f''(x0) / f'(x0)| when x is close enough.
#
# Therefore the distance from the zero at the n-th step is approximately
#   d_n ~= (1 / c) [ (c epsilon) ^ (2 ^ (n - steps)) ]
# where epsilon is chosen so that f' and f'' are nearly constant within
# epsilon of the zero and (c epsilon) << 1. Here steps is the number of
# steps it takes to get within epsilon of the zero, which might not be an
# integer. Using this formula we can interpolate noninteger steps. Specifically,
#
#   steps ~= n - log [ log (c d_n) / log(c epsilon) ] / log(2)
#
# Note that (1 / c) is approximately the scale at which quadratic behavior
# of f dominates over linear behavior near a zero, so epsilon << (1 / c) is
# necessary for good convergence anyhow.
#

import time
import numpy as np
import imageio
import profile

default_epsilon = 0.00001

np.seterr(divide='ignore', invalid='ignore')

class Region:
    def __init__(self, center, width, px_wide, px_tall):
        self.center = center
        self.width = width
        self.A = px_wide
        self.B = px_tall

    def dx(self):
        return self.width / self.A

    def ratio(self):
        return self.B / self.A

    def lowerleft(self):
        return (self.center - (self.width / 2) - 1j * (self.width / 2) * self.ratio()
            + (self.dx() / 2) * (1 + 1j))

class Poly:
    def __init__(self, roots):
        self.roots = np.array(list(roots))
        self.degree = len(self.roots)

    def eval(self, z):
        return (z[...,None] - self.roots).prod(-1)

    # compute f(z) / f'(z)
    def eval_NR(self, z):
        return 1 / (1 / (z[...,None] - self.roots)).sum(-1)

    # compute f''(r_i) / f'(r_i) where r_i is the i-th root
    def compute_ddf_df(self, i):
        ss = [self.roots[i] - r for r in self.roots]
        df = 1
        ddf = 0
        for j in range(self.degree):
            if i == j:
                continue

            df *= ss[j]

            ddf_j = 2
            for k in range(self.degree):
                if i == k or j == k:
                    continue
                ddf_j *= ss[k]
            ddf += ddf_j

        return (ddf / df)

class NF:
    def __init__(self, func, max_steps = 200, epsilon = default_epsilon):
        self.func = func

        self.max_steps = max_steps

        self.targets = np.array(list(func.roots))
        self.target_c = abs(np.array([func.compute_ddf_df(i) for i in range(func.degree)]))

        self.epsilon = min(epsilon, .01/self.target_c.max())

        self.total_queries = 0
        self.total_steps = 0
        self.most_steps = 0
        self.num_failed = 0

    # clobbers input
    def converge(self, z):
        self.total_queries += z.size
        z_shape = z.shape
        z = z.flatten()
        idx = np.arange(z.size, dtype=int)
        fsteps = np.full(z.size, -1, dtype=float)
        target = np.full(z.size, -1, dtype=int)

        iteration = 0
        while iteration < self.max_steps:
            if idx.size == 0:
                break

            # exclude NaNs
            dead_pixels = np.isnan(z[idx])
            idx = idx[~dead_pixels]

            # match iterand against targets
            err = abs(z[idx,None] - self.targets)
            converged_points, converged_to = (err < self.epsilon).nonzero()

            # process converged points
            if len(converged_points):
                i = converged_to
                j = idx[converged_points]
                c = self.target_c[i]
                err = err[converged_points, converged_to]
                fsteps[j] = iteration - np.log2(np.log(c * err) /
                                                np.log(c * self.epsilon))
                target[j] = i
                mask = np.ones(idx.size, dtype = bool)
                mask[converged_points] = False
                idx = idx[mask]

            self.total_steps += idx.size

            z -= self.func.eval_NR(z)
            iteration += 1

        self.num_failed += (target==-1).sum()
        self.most_steps = max(self.most_steps, iteration)

        return (target.reshape(z_shape), fsteps.reshape(z_shape))

class Colorizer:
    def __init__(self):
        black   = np.array([0, 0, 0], dtype = float)
        red     = np.array([1, 0, 0], dtype = float)
        green   = np.array([0, 1, 0], dtype = float)
        blue    = np.array([0, 0, 1], dtype = float)
        self.basecolors = np.array([
                blue,
                green,
                red,
                blue + green,
                red + blue,
                red + green,
                red + blue + green
            ])
        self.black = black

    # Given non-negative real, return a float from 0 to 1
    def intensity(self, steps):
        return (0.1 * np.exp(-steps / 2) +
                0.6 * np.exp(-steps / 5) +
                0.2 * (1 / (1 + np.exp((steps - 15) / 5))) +
                0.1 * (20 / (20 + steps))
            )

    # returns numpy array of length 3 with values from 0 to 1
    def color(self, result):
        target, fsteps = result
        output = self.basecolors[target % len(self.basecolors)] * self.intensity(fsteps)[...,None]
        output[target==-1] = self.black 
        return output

# antialiased == 0 suppresses antialiasing
def compute_image(nf, colorizer, region, antialiased = 3):
    A, B = region.A, region.B

    ll = region.lowerleft()
    dx = region.dx()

    # compute center of pixel
    z = ll + dx * (np.arange(A)[:,None] + 1j * np.arange(B)[None,:])
    result = nf.converge(z)

    out = colorizer.color(result)

    if antialiased >= 2:
        aa = np.linspace(-1, 1, antialiased + 2)[1:-1] / 2
        aa = (aa[:,None] + 1j*aa[None,:]).flatten()

        # Now, antialias the boundaries
        target = result[0]
        boundary = np.zeros((A, B), dtype = bool)
        boundary[1:, :]  |= (target[1:, :] != target[:-1, :])
        boundary[:-1, :] |= (target[1:, :] != target[:-1, :])
        boundary[:, 1:]  |= (target[:, 1:] != target[:, :-1])
        boundary[:, :-1] |= (target[:, 1:] != target[:, :-1])

        i, j = boundary.nonzero()
        z = ll + dx * (i[:,None] + 1j * j[:,None] + aa[None,:])
        out[i,j] = colorizer.color(nf.converge(z)).mean(1)

    return np.clip((out * 256).astype(int), 0, 255).astype(np.uint8)

def flip_data(data):
    return data.transpose(1,0,2)[::-1]

def save_image(data, filename):
    imageio.imwrite(filename, flip_data(data))

def picture(roots, region, filename = 'test.png', antialiased = 3):
    func = Poly(roots)
    nf = NF(func, max_steps = 400)

    start = time.monotonic()
    output = [None]
    profile.runctx('output[0] = compute_image(nf, Colorizer(), region, antialiased)', globals(), locals())
    image = output[0]
    elapsed = time.monotonic() - start

    save_image(image, filename)

    pixels = region.A * region.B
    print ("=== Function ===")
    print ("Degree", func.degree)
    print ("Roots", roots)
    print ("=== Domain ===")
    print ("Dimensions", region.A, "x", region.B)
    print ("Center", region.center)
    print ("Width", region.width)
    print ("=== Computation ===")
    print ("Convergence threshold", nf.epsilon)
    print ("Total iterations computed", nf.total_steps)
    print ("Average iterations computed per query", nf.total_steps / nf.total_queries)
    print ("Average iterations computed per pixel", nf.total_steps / pixels)
    print ("Average queries per pixel", nf.total_queries / pixels)
    print ("Number of queries that didn't converge", nf.num_failed)
    print ("Most steps taken to converge", nf.most_steps)
    print ("Time elapsed (seconds)", elapsed)
    print ("Time per pixel (microseconds)", 1e6 * elapsed / pixels)
    print ("")

def movie(roots_func, region, filename = 'test.mp4', fps = 20, seconds = 5, period = 1, antialiased = 3):
    colorizer = Colorizer()
    images = []

    N = int(fps * seconds)

    start = time.monotonic()

    for t in np.linspace(0, period, N, endpoint = False):
        roots = roots_func(t)
        func = Poly(roots)
        nf = NF(func, max_steps = 400)

        image = compute_image(nf, colorizer, region, antialiased)
        images.append(flip_data(image))

        print ("Done frame", len(images), "of", N, ",",
                int(time.monotonic() - start), "seconds elapsed")

    elapsed = time.monotonic() - start

    print ("Time elapsed (seconds)", elapsed)

    imageio.mimwrite(filename, images, fps = fps, quality = 10)

def root_of_unity(n):
    return np.exp(1j * 2 * np.pi / n)

if __name__ == "__main__":
    # roots = [1, -0.5 + 1j * 3**.5 / 2, -0.5 - 1j * 3**.5 / 2]
    # picture(roots, Region(0, 4, 1440, 900), 'cubic.png')
    # picture(roots, Region(0, 4, 3 * 1440, 3 * 900), 'cubic3.png')

    # roots = [1, 1.5, -1 + 1j, -1 -1j]
    # picture(roots, Region(0, 5, 1440, 900), 'p00.png')

    # w7 = root_of_unity(7)
    # roots = [1, w7, w7 ** 2, w7 ** 3, w7 ** 4, w7 ** 5, w7 ** 6]
    # picture(roots, Region(0, 4, 1440, 900), 'septic.png')

    # w5 = root_of_unity(5)
    # roots = [1, w5, w5 ** 2, w5 ** 3, w5 ** 4]
    # picture(roots, Region(0, 4, 1440, 900), 'quintic.png')

    # roots = [2, -2, 1j, -1j]
    # picture(roots, Region(0, 8, 1440, 900), 'p01.png')

    # roots = [-1, 1, -1 + 0.3j]
    # picture(roots, Region(0, 4, 1440, 900), 'p02.png')

    # roots = [2, -2, 0.5j, -0.5j]
    # picture(roots, Region(0, 8, 1440, 900), 'p03.png')

    # roots = [2, -2, 0.2j, -0.2j]
    # picture(roots, Region(0, 8, 1440, 900), 'p04.png')

    #roots = [2.3, -2.3, 1.j, -1.j]
    #picture(roots, Region(0, 8, 2*1440, 2*900), filename='p05-retina.png')

    roots = [2, -2, 0.02j, -0.02j]
    picture(roots, Region(0, 8, 1440, 900), filename='p05.png')

    #roots = (lambda t : [np.cos(t + np.pi / 2) + 1j * np.sin(2 * t + np.pi),
    #        np.cos(t) + 1j * np.sin(t),
    #        np.cos(t + np.pi) + 1j * np.sin(t + np.pi)])
    #movie(roots, Region(0, 4, 800, 800), 'm00.mp4', 30, 10, 2 * np.pi, 3)
