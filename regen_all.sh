#!/bin/bash
set -e
cd /home/claude/repo_v11
echo "=== [1/5] chequeo Ejemplo 1 reset-vs-contacto ==="
python3 -u -c "
import sys; sys.path.insert(0, '.')
import model as M, numpy as np, time, json
res = {}
for sub in ('1.1','1.2','1.3','1.4','1.5'):
    r0, th0, rs, ts = M.states_ex1(sub)
    b = {'1.1':0.01,'1.2':0.5,'1.3':0.5,'1.4':0.01,'1.5':0.5}[sub]
    Kf = M.sched_ex1(sub)
    t0 = time.time()
    _, rR, _ = M.integrate_projected(Kf, b, r0=r0, th0=th0, r_star=rs, th_star=ts)
    _, rC, _, nev = M.integrate_contact(Kf, b, r0=r0, th0=th0, r_star=rs, th_star=ts, return_events=True)
    d = float(np.abs(rR - rC).max())
    res[sub] = dict(b=b, events=int(nev), max_dr=d)
    print(f'  sub-ej {sub} (b={b}): eventos={nev:5d}  max|dr|={d:.2e}  [{time.time()-t0:.0f}s]', flush=True)
json.dump(res, open('figs/ex1_boundary_check.json','w'), indent=1)" 
echo "=== [2/5] run_scenarios --boundary contact ==="
python3 -u run_scenarios.py all --boundary contact
echo "=== [3/5] run_baseline ==="
python3 -u run_baseline.py
echo "=== [4/5] run_convergence ==="
python3 -u run_convergence.py
echo "=== [5/5] run_sensitivity ==="
python3 -u run_sensitivity.py
echo "=== LOTE COMPLETO ==="
