# =============================
# ===== Running baselines =====
# =============================

SEEDS=(2001 2002 2003 2004 2005 2006 2007 2008 2009 2010)

# DNS
for seed in "${SEEDS[@]}"; do
    python -m src.main \
        --config-name "dns.yaml" \
        task=image_rendering \
        iso_sigma=0.05 \
        line_sigma=0.5 \
        k=8 \
        use_grad=false \
        task.normalized_descriptors=true \
        seed=$seed \
        num_iterations=8000
done

# DNS + Grad
for seed in "${SEEDS[@]}"; do
    python -m src.main \
        --config-name "dns.yaml" \
        task=image_rendering \
        iso_sigma=0.05 \
        line_sigma=0.5 \
        grad_step_size=0.5\
        k=8 \
        use_grad=true \
        task.normalized_descriptors=true \
        seed=$seed \
        num_iterations=4000
done

# CMA-MAEGA
for seed in "${SEEDS[@]}"; do
    python -m src.main \
        --config-name "cma_maega.yaml" \
        task=image_rendering \
        optim_lr=0.05 \
        sigma0=1.0 \
        archive_lr=0.02 \
        task.normalized_descriptors=true \
        seed=$seed \
        num_iterations=1900
done

# CMA-MEGA
for seed in "${SEEDS[@]}"; do
    python -m src.main \
        --config-name "cma_mega.yaml" \
        task=image_rendering \
        optim_lr=0.05 \
        sigma0=1.0 \
        task.normalized_descriptors=true \
        seed=$seed \
        num_iterations=1900
done

# CMA-MAE
for seed in "${SEEDS[@]}"; do
    python -m src.main \
        --config-name "cma_mae.yaml" \
        task=image_rendering \
        sigma0=1.0 \
        archive_lr=0.02 \
        task.normalized_descriptors=true \
        seed=$seed \
        num_iterations=1900
done

# PGA-ME
for seed in "${SEEDS[@]}"; do
    python -m src.main \
    --config-name "pga_me.yaml" \
    task=image_rendering \
    iso_sigma=0.2 \
    line_sigma=0.2 \
    grad_step_size=0.05 \
    task.normalized_descriptors=true \
    seed=$seed \
    num_iterations=5350
done

# SQUAD
for seed in "${SEEDS[@]}"; do
    python -m src.main \
        task=image_rendering \
        seed=$seed \
        num_iterations=1000
done

# Evaluate

# Everything that is in a directory that starts with a digit in `outputs/`
for dir in outputs/[0-9]*/*/; do
  if [ -d "$dir" ]; then
    python -m src.evaluate "$dir"
  fi
done