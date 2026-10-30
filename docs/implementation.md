# Implementation and reproducibility

## Paper-to-code map

| Paper component | Implementation |
| --- | --- |
| Structured semantic prior and multilevel compilation | `src/sage_sr/priors.py` |
| Conditional GRU policy, FiLM and action bias | `src/sage_sr/policy.py` |
| MCTS, shared queues and four GP mutation operators | `cpp/src/mcts/`, `cpp/src/gp/` |
| Training-only LM constant fitting and reward | `cpp/src/evaluator/`, `cpp/src/eval/` |
| FIFO elite distillation and path inheritance | `src/sage_sr/estimator.py` |
| Ancestry-aware evidence and trust calibration | `src/sage_sr/feedback.py` |
| Trimmed NMSE and task-level success | `src/sage_sr/metrics.py` |

The 64-coordinate condition contains operator and variable preferences,
structural-family weights and confidences. Remaining coordinates encode motif
and interaction statistics and a fixed SHA-256 projection of their text. The
projection is deterministic across processes. Candidate expressions supplied by
the LLM are encoded as hypotheses; their text is never evaluated or directly
inserted into the candidate queue.

## Search semantics

The semantic prior and policy parameters are fixed inside a round. Both expansion
and expression completion use the fused action prior. GP parents are sampled
from the current node's compatible completion queue; edits preserve its fixed
prefix. The four mutation weights apply to node replacement, shrink, subtree
replacement and insertion, in that order. Crossover is a separate operation.

Every candidate fitness request counts against the round's evaluation budget,
including repeated structures. The evaluator caches fitted constants for each
evaluated path. Reading an elite expression or predicting on validation data does
not optimize its constants again. Inherited elites carry their exact coefficients,
training reward and ancestry roots into the next round.

The reward uses **RMSE**, not NRMSE or the test NMSE. Expression length is the
number of prefix-tree nodes, including constant terminals. Invalid predictions
receive zero reward. The default grammar is `+ - * / sin cos exp log R`, where
`R` is a fitted constant. The inherited numerical interpreter uses protected
division, logarithm and exponential evaluation as specified in
`cpp/src/eval/interpreter.cpp`.

The early-stopping error is raw NMSE on both the search-training and internal
validation splits, with constants fixed. Both must be at most `1e-6`. Set
`early_stop_error` to `null` to exhaust the allocated budget. This criterion is
separate from the length-penalized reward. A zero-variance target has error zero
only for an exact prediction, and infinity otherwise.

## Feedback semantics

The demonstration buffer has FIFO capacity 32 and reward-normalized weights,
without a separate age-decay factor. Demonstrations are evaluated under the
current semantic condition and the same native grammar mask as search.
FiLM starts at scale one and shift zero, and is trained by elite distillation.

Structure audit extracts complete prefix subtrees and counts a motif at most
once per ancestry root. Crossovers retain both parents' roots. Ancestry-derived
groups are deduplicated before comparing validation rewards. Agreement is the
sign of the candidate's mean compiled preference among explicitly weighted
tokens; neutral candidates supply no comparison evidence.

Initial explicit-bias trust is `2*clip(confidence,0.1,1)`. Later updates use
`e=clip(10*delta,-1,1)` and `alpha=clip(alpha*exp(e),0,3)`. Two consecutive
recorded updates with `e < -0.3` disable the bias for the remaining run. A missing
comparison group retains trust and resets the consecutive-conflict count.
This coefficient controls explicit token bias only.

## Evaluation semantics

Predictions with relative error at or below its 95th percentile are retained,
including ties. NMSE divides the retained squared error sum by the retained
target deviation sum. Success is a Boolean per task based on the maximum retained
relative error; average success across tasks and multiply by 100 for
`Acc_0.01`. It is not a pointwise accuracy fraction.

Relative error divides by `max(abs(y), float64_machine_epsilon)`, an explicit
guard at zero targets. Any non-finite prediction fails the whole task rather
than being discarded before scoring. Internal validation is a deterministic
15% split, with at least one point on each side for small datasets. Its seed is
saved independently of the search seed.

## Recorded evidence and data access

Saved results include the actual configuration, feature names, fitted expression
and coefficients, per-round priors, calibration, distillation loss, inherited
paths, and evaluation events with GP parents and ancestry roots. Test metrics
are computed only after fit. No test observation enters the semantic context.

Acquire LLM-SRBench from its original distribution:
<https://huggingface.co/datasets/nnheui/llm-srbench>.
Provide benchmark instances as explicit CSV or NPZ files; there is no implicit
download or import-time directory scanning. Preserve the benchmark's train/ID/OOD
splits. Data conversion must retain feature order and use only the training
array as `X/y`.

For a task suite, create a JSON list containing `id`, `domain`, `data` (an NPZ
path relative to the manifest), and optional `context`. Then run:

```bash
python examples/run_benchmark.py manifest.json --config configs/paper.json \
  --seeds 0 1 2 --output results/benchmark
```

The runner derives validation seeds from stable task identifiers, so changing
task order or sharding does not change the internal split. It records per-task
metrics and computes task-level accuracy separately for each domain and seed.

The source release implements the current method specification. Historical
experiment logs and reported result tables are not bundled or rewritten.
Engineering tests validate implementation paths; they do not substitute for
full-budget evaluations with the stated LLM backbones.
