# =============================
# ===== Running baselines =====
# =============================

SEEDS=(2001 2002 2003 2004 2005)

# DNS
for seed in "${SEEDS[@]}"; do
    python -m src.main \
        --config-name "dns.yaml" \
        task=lsi \
        isoline_batch_size=16 \
        eval_batch_size=16 \
        iso_sigma=0.05 \
        line_sigma=0.5 \
        k=8 \
        use_grad=false \
        task.normalized_descriptors=true \
        seed=$seed \
        num_iterations=3000
    sleep 30
done

# DNS + Grad
for seed in "${SEEDS[@]}"; do
    python -m src.main \
        --config-name "dns.yaml" \
        task=lsi \
        isoline_batch_size=16 \
        eval_batch_size=16 \
        grad_batch_size=8 \
        iso_sigma=0.05 \
        line_sigma=0.5 \
        grad_step_size=0.5\
        k=8 \
        use_grad=true \
        task.normalized_descriptors=true \
        seed=$seed \
        num_iterations=1500
    sleep 30
done

# CMA-MAEGA
for seed in "${SEEDS[@]}"; do
    python -m src.main \
        --config-name "cma_maega.yaml" \
        task=lsi \
        population_size=40000 \
        num_iterations=3000 \
        num_emitters=1 \
        batch_size=16 \
        optim_lr=0.05 \
        sigma0=0.01 \
        archive_lr=0.02 \
        task.normalized_descriptors=true \
        seed=$seed
done

# CMA-MEGA
for seed in "${SEEDS[@]}"; do
    python -m src.main \
        --config-name "cma_mega.yaml" \
        task=lsi \
        population_size=40000 \
        num_iterations=3000 \
        num_emitters=1 \
        batch_size=16 \
        optim_lr=0.05 \
        sigma0=0.01 \
        task.normalized_descriptors=true \
        seed=$seed
done

# CMA-MAE
for seed in "${SEEDS[@]}"; do
    python -m src.main \
        --config-name "cma_mae.yaml" \
        task=lsi \
        population_size=40000 \
        num_iterations=3000 \
        num_emitters=1 \
        batch_size=16 \
        sigma0=0.02 \
        archive_lr=0.1 \
        task.normalized_descriptors=true \
        seed=$seed
done

# PGA-ME
for seed in "${SEEDS[@]}"; do
    python -m src.main \
    --config-name "pga_me.yaml" \
    task=lsi \
    num_iterations=3000 \
    population_size=40000 \
    batch_size=16 \
    iso_sigma=0.01 \
    line_sigma=0.2 \
    grad_step_size=0.05 \
    task.normalized_descriptors=true \
    seed=$seed
done

# SQUAD
for seed in "${SEEDS[@]}"; do
    python -m src.main \
        task=lsi \
        population_size=256 \
        num_iterations=175 \
        batch_size=8 \
        sigma_rule=0.01 \
        optimizer.learning_rate=0.1 \
        seed=$seed
done

# Evaluate

# Everything that is in a directory that starts with a digit in `outputs/`
for dir in outputs/[0-9]*/*/; do
  if [ -d "$dir" ]; then
    python -m src.evaluate "$dir"
  fi
done