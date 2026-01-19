SEEDS=(8001 8002 8003)
NNS=(4 8 16 32)
# SQUAD
for seed in "${SEEDS[@]}"; do
    for nn in "${NNS[@]}"; do
        python -m src.main \
            task=image_rendering \
            seed=$seed \
            num_neighbors=$nn \
            num_iterations=1000
    done
done