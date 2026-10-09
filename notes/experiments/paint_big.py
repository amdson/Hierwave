"""Paint the large branching root exemplar (coord.EXEMPLAR_BIG) from strokes and print it as ASCII."""
import numpy as np

H, W = 40, 84
g = np.zeros((H, W), np.int64)          # 0 earth, 1..3 mass, -1 sky
g[:2] = -1
TY, TX = 2, 42


def line(y0, x0, y1, x1):
    """4-connected staircase from (y0,x0) to (y1,x1)."""
    pts = [(y0, x0)]
    y, x = y0, x0
    dy, dx = y1 - y0, x1 - x0
    n = abs(dy) + abs(dx)
    sy, sx = np.sign(dy), np.sign(dx)
    ey, ex = 0.0, 0.0
    for _ in range(n):
        # choose the axis with the larger remaining fraction
        ry = abs(y1 - y) / max(abs(dy), 1)
        rx = abs(x1 - x) / max(abs(dx), 1)
        if (y != y1) and (x == x1 or ry >= rx):
            y += sy
        else:
            x += sx
        pts.append((y, x))
    return pts


def stroke(points, mass, width=1):
    for (y0, x0), (y1, x1) in zip(points[:-1], points[1:]):
        for y, x in line(y0, x0, y1, x1):
            for yy in range(y, y + width):
                for xx in range(x, x + width):
                    if 0 <= yy < H and 0 <= xx < W and g[yy, xx] >= 0:
                        g[yy, xx] = max(g[yy, xx], mass)


# taproot
stroke([(3, 41), (9, 41), (14, 42)], 3, 3)
stroke([(15, 43), (22, 44), (26, 45)], 2, 2)
stroke([(27, 45), (31, 46), (35, 47)], 1)
# left main lateral
stroke([(4, 40), (6, 32), (8, 26)], 3, 2)
stroke([(9, 26), (12, 19), (15, 14)], 2, 2)
stroke([(16, 14), (20, 9), (24, 6)], 1)
stroke([(10, 23), (16, 21), (22, 23)], 2)          # drop from the left lateral
stroke([(23, 23), (28, 26)], 1)
stroke([(13, 17), (17, 17), (21, 15)], 1)
stroke([(7, 30), (11, 30), (13, 33)], 1)
# right main lateral
stroke([(4, 44), (6, 52), (8, 58)], 3, 2)
stroke([(9, 58), (12, 65), (15, 70)], 2, 2)
stroke([(16, 71), (20, 76), (23, 80)], 1)
stroke([(9, 56), (15, 55), (21, 57)], 2)           # drop from the right lateral
stroke([(22, 57), (27, 60)], 1)
stroke([(12, 63), (16, 62), (20, 64)], 1)
stroke([(7, 50), (11, 51), (14, 49)], 1)
# mid laterals off the taproot
stroke([(12, 40), (15, 33), (18, 28)], 2, 2)
stroke([(19, 28), (23, 24), (27, 21)], 1)
stroke([(15, 36), (20, 36), (24, 38)], 1)
stroke([(13, 44), (16, 51), (19, 56)], 2, 2)
stroke([(20, 57), (24, 61), (28, 64)], 1)
stroke([(17, 53), (22, 52), (26, 54)], 1)
# lower laterals
stroke([(21, 43), (25, 38), (29, 35)], 1)
stroke([(23, 45), (27, 50), (31, 53)], 1)
stroke([(29, 46), (33, 42)], 1)
g[TY, TX] = 3
CH = {-1: " ", 0: ".", 1: "1", 2: "2", 3: "3"}
rows = ["".join(CH[v] for v in r) for r in g]
rows[TY] = rows[TY][:TX] + "T" + rows[TY][TX + 1:]
for r in rows:
    print(f'    "{r}",')
