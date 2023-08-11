#!/bin/bash
subject=$1
python scripts/render.py --dataset renderpeople -s "$subject" \
                         -i data/params/renderpeople/         \
                         -o data/normal_map/renderpeople/     \
                         --body --head --front --smpl --normal --egl --align_yaw