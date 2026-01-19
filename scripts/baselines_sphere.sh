#!/bin/bash

SEEDS=(2001 2002 2003 2004 2005 2006 2007 2008 2009 2010)
DDIMS=(2 4 8 16 32)

for ddim in "${DDIMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    python -m src.main --config-name "cma_mae.yaml" task.normalized_descriptors=true task.solution_dim=2048 task.descriptor_dim="${ddim}" seed="${seed}"
    python -m src.main --config-name "cma_mega.yaml" task.normalized_descriptors=true task.solution_dim=2048 task.descriptor_dim="${ddim}" seed="${seed}"
    python -m src.main --config-name "cma_maega.yaml" task.normalized_descriptors=true task.solution_dim=2048 task.descriptor_dim="${ddim}" seed="${seed}"
    python -m src.main --config-name "pga_me.yaml" task.normalized_descriptors=true task.solution_dim=2048 task.descriptor_dim="${ddim}" seed="${seed}"
    python -m src.main task.solution_dim=2048 task.descriptor_dim="${ddim}" seed="${seed}"
  done
done

# Everything that is in a directory that starts with a digit in `outputs/`
for dir in outputs/[0-9]*/*/; do
  if [ -d "$dir" ]; then
    python -m src.evaluate "$dir"
  fi
done
