#!/usr/bin/env bash
# Used inside docker for:
# python3 dist-dqn-trainer.py --ps_hosts=localhost:2222 --worker_hosts=localhost:2223 --job_name=ps --task_index=0
# python3 dist-dqn-trainer.py --ps_hosts=localhost:2222 --worker_hosts=localhost:2223 --job_name=worker --task_index=0
python /dist-dqn-trainer.py $@