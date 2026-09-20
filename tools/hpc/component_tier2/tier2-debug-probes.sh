#!/bin/bash -l
set -euo pipefail
for index in 0 1 2 3 4; do
 export SLURM_ARRAY_TASK_ID=$index
 bash /data/leuven/379/vsc37996/MCUFlowNet-component/control/tier2-run.sh probe
done
