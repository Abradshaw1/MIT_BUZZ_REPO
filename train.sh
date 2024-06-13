#!/bin/bash

conda activate python39
module load cuda/11.8
module load nccl/2.18.1-cuda11.8
module load anaconda/2023b
python TCN_Buzz_Claissifier_v2.py
