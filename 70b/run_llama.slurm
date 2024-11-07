#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodes=32
#SBATCH --exclusive
#SBATCH --cpus-per-task 128
#SBATCH --output=slurm-%x-%j.out

srun bash compile_and_run.sh compile
