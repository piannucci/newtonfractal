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

import math
import cmath
import time
import numpy as np
import imageio

log = math.log
exp = math.exp
pi = math.pi
cos = math.cos
sin = math.sin

default_epsilon = 0.00001

class Region:
    def __init__(self, center, width, px_wide, px_tall):
        self.center = center
        self.width = width
        self.A = px_wide
        self.B = px_tall

    def pixels(self):
        return [(i, j) for i in range(self.A) for j in range(self.B)]

    def dx(self):
        return self.width / self.A

    def ratio(self):
        return self.B / self.A

    def lowerleft(self):
        return (self.center - (self.width / 2) - 1j * (self.width / 2) * self.ratio()
            + (self.dx() / 2) * (1 + 1j))

class Poly:
    def __init__(self, roots):
        self.roots = list(roots)
        self.degree = len(self.roots)

    def eval(self, z):
        result = 1
        for r in self.roots:
            result *= (z - r)
        return result

    # compute f(z) / f'(z)
    def eval_NR(self, z):
        result = 0
        for r in self.roots:
            result += 1 / (z - r)
        if result == 0:
            return float('nan') * 1j
        return 1 / result

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
    def __init__(self, func, max_steps = None, epsilon = None):
        self.func = func

        if max_steps is None:
            self.max_steps = 200
        else:
            self.max_steps = max_steps

        if epsilon is None:
            epsilon = default_epsilon

        self.targets = list(func.roots)
        self.target_c = [abs(func.compute_ddf_df(i)) for i in range(func.degree)]

        for c in self.target_c:
            if epsilon * c > 0.01:
                epsilon = 0.01 / c
        self.epsilon = epsilon

        self.total_queries = 0
        self.total_steps = 0
        self.most_steps = 0
        self.num_failed = 0


    def converge(self, z):
        self.total_queries += 1
        go = self.func.eval_NR
        epsilon = self.epsilon

        steps = 0
        while steps < self.max_steps:
            if cmath.isnan(z):
                break

            for i, t in enumerate(self.targets):
                if abs(z - t) < epsilon:
                    c = self.target_c[i]
                    fsteps = steps - log(log(c * abs(z - t)) / log(c * epsilon)) / log(2)

                    self.total_steps += steps
                    self.most_steps = max(self.most_steps, steps)
                    return (i, fsteps)

            z = z - go(z)
            steps += 1

        self.total_steps += steps
        self.num_failed += 1
        return None

class Colorizer:
    def __init__(self):
        black   = np.array([0, 0, 0], dtype = float)
        red     = np.array([1, 0, 0], dtype = float)
        green   = np.array([0, 1, 0], dtype = float)
        blue    = np.array([0, 0, 1], dtype = float)
        self.basecolors = [
                blue,
                green,
                red,
                blue + green,
                red + blue,
                red + green,
                red + blue + green
            ]
        self.black = black

    # Given non-negative integer, return a float from 0 to 1
    def intensity(self, steps):
        return (0.1 * exp(-steps / 2) +
                0.6 * exp(-steps / 5) +
                0.2 * (1 / (1 + exp((steps - 15) / 5))) +
                0.1 * (20 / (20 + steps))
            )

    # returns numpy array of length 3 with values from 0 to 1
    def color_converge(self, i, steps):
        bc = self.basecolors[i % len(self.basecolors)]
        return bc * self.intensity(steps)

    def color_noconverge(self):
        return self.black

    def color(self, result):
        if result is None:
            return self.color_noconverge()
        else:
            return self.color_converge(result[0], result[1])

    def mean(self, colors):
        s = 0
        for c in colors:
            s += c
        return s * (1 / len(colors))

    def quantize(self, colors):
        x = colors * 256
        x[x <= 0] = 0
        x[x >= 255] = 255
        return x.astype(dtype = np.uint8)

# antialiased == 0 suppresses antialiasing
def compute_image(nf, colorizer, region, antialiased = 3):
    A, B = region.A, region.B
    out = np.zeros((A, B, 3), dtype = float)

    ll = region.lowerleft()
    dx = region.dx()

    # -1 will be used for missing the target
    target  = np.zeros((A, B), dtype = int)
    steps   = np.zeros((A, B), dtype = float)

    # First, compute every pixel
    for i, j in region.pixels():
        # compute center of pixel
        z = ll + dx * (i + 1j * j)
        result = nf.converge(z)

        if result is None:
            target[i, j] = -1
            steps[i, j] = 0
        else:
            target[i, j] = result[0]
            steps[i, j] = result[1]

        out[i, j, :] = colorizer.color(result)

    if antialiased >= 2:
        aa = list(np.linspace(-1, 1, antialiased + 2)[1:-1])

        # Now, antialias the boundaries
        boundary = np.zeros((A, B), dtype = bool)
        boundary[1:, :]     |= np.not_equal(target[1:, :], target[:-1, :])
        boundary[:-1, :]    |= np.not_equal(target[1:, :], target[:-1, :])
        boundary[:, 1:]     |= np.not_equal(target[:, 1:], target[:, :-1])
        boundary[:, :-1]    |= np.not_equal(target[:, 1:], target[:, :-1])

        for i, j in region.pixels():
            if not boundary[i, j]:
                continue
            colors = []
            for i_ in aa:
                for j_ in aa:
                    z = ll + dx * (i + i_ + 1j * (j + j_))
                    result = nf.converge(z)
                    colors.append(colorizer.color(result))
            out[i, j, :] = colorizer.mean(colors)

    return colorizer.quantize(out)

def flip_data(data):
    width, height, x = data.shape
    assert (x == 3)

    data = (np.transpose(data, (1, 0, 2)))[::-1, :, :]
    assert (data.shape == (height, width, 3))

    return data

def save_image(data, filename):
    imageio.imwrite(filename, flip_data(data))

def picture(roots, region, filename = 'test.png'):
    func = Poly(roots)
    nf = NF(func, max_steps = 400)

    start = time.monotonic()
    image = compute_image(nf, Colorizer(), region)
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
    return cos(2 * pi / n) + 1j * sin(2 * pi / n)

if __name__ == "__main__":
    # roots = [1, -0.5 + 1j * math.sqrt(3) / 2, -0.5 - 1j * math.sqrt(3) / 2]
    # run(roots, 0, 4, 1440, 900, 'cubic.png')
    # run(roots, 0, 4, 3 * 1440, 3 * 900, 'cubic3.png')


    # roots = [1, 1.5, -1 + 1j, -1 -1j]
    # run(roots, 0, 5, 1440, 900, 'p00.png')

    # w7 = root_of_unity(7)
    # roots = [1, w7, w7 ** 2, w7 ** 3, w7 ** 4, w7 ** 5, w7 ** 6]
    # run(roots, 0, 4, 1440, 900, 'septic.png')

    # w5 = root_of_unity(5)
    # roots = [1, w5, w5 ** 2, w5 ** 3, w5 ** 4]
    # run(roots, 0, 4, 1440, 900, 'quintic.png')

    # roots = [2, -2, 1j, -1j]
    # run(roots, 0, 8, 1440, 900, 'p01.png')

    # roots = [-1, 1, -1 + 0.3j]
    # run(roots, 0, 4, 1440, 900, 'p02.png')

    # roots = [2, -2, 0.5j, -0.5j]
    # run(roots, 0, 8, 1440, 900, 'p03.png')

    # roots = [2, -2, 0.2j, -0.2j]
    # run(roots, 0, 8, 1440, 900, 'p04.png')

    # roots = [2, -2, 0.02j, -0.02j]
    # run(roots, 0, 8, 1440, 900, 'p05.png')

    roots = (lambda t : [cos(t + pi / 2) + 1j * sin(2 * t + pi),
            cos(t) + 1j * sin(t),
            cos(t + pi) + 1j * sin(t + pi)])
    movie(roots, Region(0, 4, 800, 800), 'm00.mp4', 30, 10, 2 * pi, 3)
