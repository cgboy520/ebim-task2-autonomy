#!/usr/bin/env bash
# Source ROS 2 Jazzy before running the autonomy node.
set -euo pipefail

if [ -f /opt/ros/jazzy/setup.bash ]; then
    # Relax -u while sourcing ROS setup.bash.
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
    set -u
fi

exec "$@"
