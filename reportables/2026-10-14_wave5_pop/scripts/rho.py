import json,sys
for f in sys.argv[1:]:
    m=json.load(open(f)); L=m["blocks"]["levels"]
    print(f, "rho", [round(L[str(l)]["direct"]["rho_pool"],2) for l in range(len(L))],
          "starved%", [round(100*L[str(l)]["direct"]["starved_probe"],1) for l in range(len(L))])
