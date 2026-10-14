#!/bin/bash
grep -v Activ runs/$1.log | grep '"iter"' | tail -${2:-1} | python3 -c "
import json,sys
for l in sys.stdin:
  d=json.loads(l); p=d['pop']
  o8=d.get('own8',{}); lv=d.get('lv_own'); lp=d.get('lv_pop')
  print(d['iter'],d['update'],'pop d/lm/valid %.3f %.3f %.2f'%(p['dirt'],p['line_match'],p['valid']),'| own8',{k:round(v,3) for k,v in o8.items()},'| lv_own',[round(x,5) for x in lv] if lv else None,'| rkl',[round(x,5) for x in d['rkl_raw_viol']] if 'rkl_raw_viol' in d else None,'lam',[round(x,1) for x in d['lam']] if 'lam' in d else None,'| t %.1f'%d['t'])"
