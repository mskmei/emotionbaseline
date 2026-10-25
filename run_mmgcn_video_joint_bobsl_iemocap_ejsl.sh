#!/usr/bin/env bash
set -euo pipefail
export BASELINE=MMGCN
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_video_joint_bobsl_iemocap_ejsl.sh" "$@"
