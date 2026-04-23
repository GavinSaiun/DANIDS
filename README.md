Stage 1: Dataset preparation
choose 3 compatible datasets
derive common feature set
remove IPs/timestamps
clean NaNs/infs
map labels to benign vs attack
fit source-only scaler
Stage 2: Shift analysis

For every source-target pair:

compare feature distributions
visualise domain separation
compute a shift metric
Stage 3: Baselines

For each pair:

source-only training, zero-shot target evaluation
naive fine-tuning on target
maybe target-only upper bound
Stage 4: Domain adaptation

Apply one or more:

Deep CORAL
MMD
DANN
Stage 5: Continual learning / forgetting

During adaptation:

no replay
random replay
boundary-critical replay
Stage 6: Evaluation

Report:

target accuracy / F1 / recall / AUROC
source retention after adaptation
forgetting score
performance across all 6 transfer directions