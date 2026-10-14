#!/bin/bash
# usage: peek.sh run  -> last iteration summary
grep -v Activ runs/$1.log | grep '"iter"' | tail -${2:-1} | python3 -c "
import json,sys
for l in sys.stdin:
  d=json.loads(l); p=d['pop']; o=d['own_raw']
  print(d['iter'],d['update'],'own d/lm %.3f %.3f'%(o['dirt'],o['line_match']),'| pop d/lm/valid/Btop %.3f %.3f %.2f %.3f'%(p['dirt'],p['line_match'],p['valid'],p.get('B_top_row',-1)),'| tt acc',d.get('acc_tt_L4'),d.get('acc_tt_L3'),'p2p',d.get('pure2part_tt_L4'),d.get('pure2part_tt_L3'),'| t %.1f %.1f'%(d['time_refine'],d['time_ml']))"
