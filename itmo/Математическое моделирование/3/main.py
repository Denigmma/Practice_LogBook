# -*- coding: utf-8 -*-
import math

# From the plot: max |f| = 2.2 (dashed horizontal line)
f_max = 2.2

# r = |a| from f_max = 1/(1-r)
r = 1.0 - 1.0 / f_max

# Inflection at omega = 0 gives quadratic:
# r*c^2 + (1+r^2)*c - 3r = 0, where c = cos(theta)
A = r
B = 1.0 + r*r
C = -3.0 * r

disc = B*B - 4.0*A*C
c1 = (-B + math.sqrt(disc)) / (2.0*A)
c2 = (-B - math.sqrt(disc)) / (2.0*A)

# Choose the root that lies in [-1,1] and gives theta in (0, pi/2)
candidates = []
for c in (c1, c2):
    if -1.0 <= c <= 1.0:
        theta = math.acos(c)
        candidates.append((theta, c))

theta, c = min(candidates, key=lambda t: t[0])  # smaller theta in [0,pi]

# Min |f| = 1/(1+r)
f_min = 1.0 / (1.0 + r)

re_a = r * math.cos(theta)
im_a = r * math.sin(theta)

print("abs_a =", "{:.10f}".format(r))
print("min_abs_f =", "{:.10f}".format(f_min))
print("re_a =", "{:.10f}".format(re_a))
print("im_a =", "{:.10f}".format(im_a))

print("abs_a_4dp =", "{:.4f}".format(r))
print("min_abs_f_4dp =", "{:.4f}".format(f_min))
print("re_a_4dp =", "{:.4f}".format(re_a))
print("im_a_4dp =", "{:.4f}".format(im_a))