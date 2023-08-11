#!/bin/bash
subject=$1
python scripts/render.py --dataset frankmocap -s "$subject" \
                         -i data/params/frankmocap/mocap    \
                         -o data/normal_map/frankmocap      \
                         --body --head --front --smpl --normal --egl --align_yaw