"""Write a placeholder template file in the contract format (HWT1).

Not the real templates (thread C writes data/*.hwt); this only exercises the
Luanti pipeline end to end.  8 templates, N = 8, L = 24:
  0 deep water, 1 water/shore, 2 sand, 3 grass, 4 grass + tree,
  5 house, 6 road cross, 7 tower.

    uv run python luanti/tools/make_placeholder_hwt.py
"""

import struct
from pathlib import Path

import numpy as np

LUANTI = Path(__file__).resolve().parent.parent
OUT = LUANTI / "games" / "hierwave" / "mods" / "hierwave_gen" / "test_data" / "placeholder.hwt"
NAMES = (LUANTI / "palette.txt").read_text().split()
P = {n: i for i, n in enumerate(NAMES)}
N, L = 8, 24


def ground(top):
    t = np.zeros((L, N, N), np.uint8)
    t[0:3] = P["dirt"]
    t[3] = P[top]
    return t


def make():
    ts = []
    # 0 deep water: water surface at layer 2, deep water below
    t = np.zeros((L, N, N), np.uint8)
    t[0] = P["sand"]
    t[1] = P["deep_water"]
    t[2] = P["water"]
    ts.append(t)
    # 1 shallow water
    t = np.zeros((L, N, N), np.uint8)
    t[0:2] = P["sand"]
    t[2] = P["water"]
    ts.append(t)
    # 2 sand
    t = ground("sand")
    t[0:3] = P["sand"]
    ts.append(t)
    # 3 grass
    ts.append(ground("grass"))
    # 4 grass + tree
    t = ground("dark_grass")
    t[4:9, 3:5, 3:5] = P["trunk"]
    t[8:12, 1:7, 1:7] = P["leaves"]
    t[12, 2:6, 2:6] = P["leaves"]
    ts.append(t)
    # 5 house
    t = ground("grass")
    t[4:8, 1:7, 1:7] = P["house_wall"]
    t[4:7, 2:6, 2:6] = P["air"]
    t[5, 1, 3:5] = P["window"]
    t[4:6, 6, 3] = P["air"]  # door
    for h in range(4):
        t[8 + h, 1 + h:7 - h, 1 + h:7 - h] = P["house_roof"]
    t[8:11, 2, 5] = P["chimney"]
    ts.append(t)
    # 6 road cross
    t = ground("paving")
    t[3, 2:6, :] = P["asphalt"]
    t[3, :, 2:6] = P["asphalt"]
    t[3, 3:5, 0:2] = P["road_line"]
    t[3, 3:5, 6:8] = P["road_line"]
    t[4:9, 0, 0] = P["lamp"]
    ts.append(t)
    # 7 tower
    t = ground("concrete")
    t[4:20, 1:7, 1:7] = P["tower_wall"]
    t[4:20, 2:6, 2:6] = P["air"]
    for k in range(5, 20, 2):
        t[k, 1, 2:6] = P["glass"]
        t[k, 6, 2:6] = P["glass"]
    t[20, 1:7, 1:7] = P["tower_top"]
    ts.append(t)
    return np.stack(ts)


def main():
    data = make()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(b"HWT1" + struct.pack("<III", data.shape[0], N, L))
        f.write(np.ascontiguousarray(data).tobytes())
    print(f"wrote {data.shape[0]} templates to {OUT}")


if __name__ == "__main__":
    main()
