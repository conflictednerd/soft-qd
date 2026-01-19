#!/usr/bin/env bash

# Without grad
# Best value iso 0.05, line 0.5, k = 8
ALL_ISOS=(0.05 0.1 0.2 0.5 1.0)
ALL_LINES=(0.05 0.1 0.2 0.5)
ALL_KS=(8 16)

for iso_v in "${ALL_ISOS[@]}"; do
    for line_v in "${ALL_LINES[@]}"; do
        for k_v in "${ALL_KS[@]}"; do
            python -m src.main --config-name dns.yaml \
                k="${k_v}" line_sigma="${line_v}" iso_sigma="${iso_v}" \
                use_grad=False task.normalized_descriptors=True num_iterations=8000
        done
    done
done

# With grad
ALL_LRS=(0.05 0.1 0.5 1.0)
for lr in "${ALL_LRS[@]}"; do
    python -m src.main --config-name dns.yaml k=8 grad_step_size="${lr}" line_sigma=0.5 iso_sigma=0.05 use_grad=True task.normalized_descriptors=True num_iterations=4000
done