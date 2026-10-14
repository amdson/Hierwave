lane=$1; shift
for job in "$@"; do set -- $job; taskset -c $CORES timeout 700 python ml_fit.py $1 $2 $3 ${4:-240} > /tmp/kernelc_$1_$2_$3.log 2>&1; done
