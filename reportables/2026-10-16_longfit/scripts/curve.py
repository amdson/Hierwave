import json, sys
rows=[json.loads(l) for l in open('runs/clongfit/train.jsonl')]
rows=[r for r in rows if 'update' in r and 'own8' in r]
print("| min | updates | upd/s | own8 line_match | dirt | offset L0..L4 | lv_own frac L0..L4 |")
print("|---|---|---|---|---|---|---|")
last=-1e9
for r in rows:
    if r['t']-last<540 and r is not rows[-1]: continue
    last=r['t']; o=r['own8']
    print(f"| {r['t']/60:.0f} | {r['update']} | {r['update']/max(r['t_train'],1):.1f} | {o['line_match']:.3f} | {o['dirt']:.3f} | {','.join(f'{x:+.2f}' for x in o['offset'])} | {','.join(f'{x:.4f}' for x in r['lv_own'])} |")
print()
print("| tag | updates | line_match raw | raw viol frac L0..L4 | rho_pool L0..L4 | starved_probe L0..L4 | starved_own L0..L4 |")
print("|---|---|---|---|---|---|---|")
for l in open('runs/clongfit/eval.jsonl'):
    m=json.loads(l); b=m['blocks']
    lv=b['levels'] if b else None
    def g(k):
        if not lv: return '-'
        return ','.join(f"{lv[str(i)]['direct'][k]:.3g}" for i in range(5))
    print(f"| {m['tag']} | {m['updates']} | {m['stats_raw'].get('line_match',[0])[0]:.3f} | {','.join(f'{v:.4f}' for v in m['viol_raw'].values())} | {g('rho_pool')} | {g('starved_probe')} | {g('starved_own')} |")
