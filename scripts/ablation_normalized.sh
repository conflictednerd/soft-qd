SEEDS=(6001 6002 6003)

# SQUAD
for seed in "${SEEDS[@]}"; do
    python -m src.main \
        task=image_rendering \
        seed=$seed \
        task.normalized_descriptors=true \
        num_iterations=1000
done