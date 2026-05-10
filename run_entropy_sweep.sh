#!/bin/bash

# Default values
START=0.5
END=3.0
INTERVAL=0.5

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--start) START="$2"; shift ;;
        -e|--end) END="$2"; shift ;;
        -i|--interval) INTERVAL="$2"; shift ;;
        -h|--help) 
            echo "Usage: ./run_entropy_sweep.sh -s <start_freq> -e <end_freq> -i <interval>"
            exit 0 
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "=========================================================="
echo " Starting Frequency Sweep (Entropy Config)"
echo " Start: $START | End: $END | Interval: $INTERVAL"
echo "=========================================================="

EXPERIMENTS=""
# Use seq to generate numbers formatted to 2 decimal places (e.g., 0.50)
for FREQ in $(seq -f "%.2f" $START $INTERVAL $END); do
    EXPERIMENTS="$EXPERIMENTS freq_ent${FREQ}"
done

echo ">>> Running benchmark suite for: $EXPERIMENTS <<<"
python benchmark.py --experiments $EXPERIMENTS --model-types qpinn --output-dir "./results/entropy_sweep"
python _plot.py

echo -e "\n✅ Sweep completed. All results saved in ./results/entropy_sweep/"
