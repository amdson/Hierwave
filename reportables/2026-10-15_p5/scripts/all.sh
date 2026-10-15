source reportables/2026-10-14_cr2_penalty/scripts/env.sh 1
for a in "tight s1 200" "tight f2 40" "tight f4 40" "chain s1 200" "chain f2 40" "chain f4 40" "tight s2 200" "tight b11 40" "chain s2 200" "chain b11 40"; do
  set -- $a
  while [ $(cat /sys/fs/cgroup/pids.current) -gt 1400 ]; do sleep 10; done
  echo "== $a"; taskset -c 4-5 python reportables/2026-10-15_p5/scripts/p5.py $1 $2 $3 2>&1 | tail -3
done
