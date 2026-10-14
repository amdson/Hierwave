"""Violations per level by factor type (support / honourA / honourB, homed at the children's level), raw and final."""
import json, sys, re
for f in sys.argv[1:]:
    o = json.load(open(f))
    print("==", f, o["n_worlds"], "worlds")
    for kind in ("raw", "final"):
        agg = {}
        for name, (v, n) in o[kind]["factors"].items():
            m = re.match(r"(support|honourA|honourB|prior|smooth)(\d)", name)
            typ, l = m.group(1), int(m.group(2))
            home = l if typ == "support" else l - 1
            agg.setdefault(home, {}).setdefault(typ, 0)
            agg[home][typ] += v
        print(kind, "frac", [f"{x:.1e}" for x in o[kind]["frac"]], "lm %.3f" % o[kind]["line_match"])
        for l in sorted(agg):
            print("   L%d" % l, agg[l])
