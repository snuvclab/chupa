#!/bin/bash
subject=$1
python scripts/render.py --dataset thuman -s "$subject" \
                         -i data/params/thuman/         \
                         -o data/normal_map/thuman/     \
                         --body --head --front --smpl --normal --egl --align_yaw