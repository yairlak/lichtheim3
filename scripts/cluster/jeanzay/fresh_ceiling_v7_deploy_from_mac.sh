#!/bin/bash
# Mac side: push the V7 branch, then run the Jean-Zay submit script over the
# interactive SSH ControlMaster (Jean-Zay needs a password-authenticated
# master; open it once with:
#   ssh -M -S ~/.ssh/controlmasters/jz-autoresearch -o ControlPersist=yes jz true
# ).  Usage: fresh_ceiling_v7_deploy_from_mac.sh <V7_EXECUTABLE_COMMIT>
set -euo pipefail
COMMIT=${1:?V7 executable commit}
SOCK=${SOCK:-$HOME/.ssh/controlmasters/jz-autoresearch}
WT=$(cd "$(dirname "$0")/../../.." && pwd)
[[ "$(git -C "$WT" rev-parse HEAD)" == "$COMMIT" || -n "$(git -C "$WT" branch --contains "$COMMIT" 2>/dev/null)" ]] || { echo "FATAL: $COMMIT not in $WT"; exit 1; }
[[ -z "$(git -C "$WT" status --porcelain -uno)" ]] || { echo "FATAL: dirty tracked tree"; exit 1; }
git -C "$WT" push origin fresh/step0-ceiling-v7
[[ -S "$SOCK" ]] || { echo "FATAL: no live Jean-Zay ControlMaster at $SOCK"; exit 2; }
ssh -S "$SOCK" -o BatchMode=yes jz 'hostname; whoami'
ssh -S "$SOCK" -o BatchMode=yes jz "bash -lc 'mkdir -p \$WORK/l3_fresh_v7 && cat > \$WORK/l3_fresh_v7/fresh_ceiling_v7_submit.sh'" < "$WT/scripts/cluster/jeanzay/fresh_ceiling_v7_submit.sh"
ssh -S "$SOCK" -o BatchMode=yes jz "bash -lc 'bash \$WORK/l3_fresh_v7/fresh_ceiling_v7_submit.sh $COMMIT'"
