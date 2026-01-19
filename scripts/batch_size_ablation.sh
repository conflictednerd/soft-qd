SEEDS=(8001 8002 8003)
BSZ=(4 8 16 32 64)
# SQUAD
for seed in "${SEEDS[@]}"; do
    for bs in "${BSZ[@]}"; do
        python -m src.main \
            task=image_rendering \
            seed=$seed \
            batch_size=$bs \
            num_iterations=1000
    done
done