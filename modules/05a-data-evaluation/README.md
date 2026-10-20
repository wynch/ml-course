# Lab 05½ — Data & evaluation: know whether the model improved

> Training gives you a number. Evaluation tells you whether that number means
> anything.

This bridge lab sits between **module 05** (using a real model) and **module 06**
(fine-tuning one). You build a small evaluation harness from scratch before a
trainer or benchmark library hides the important decisions: the split, the
metric, the threshold, the slices, and the possibility that the answer leaked
into the test set.

**🕹 Interactive:** [The honest scorecard](../../explorables/05a-evaluation-lab.html)
— move a decision threshold, inspect the confusion matrix, and introduce data
leakage to see a suspiciously good score.

**✅ Quiz:** [10 questions](../../quizzes/05a.html) — check yourself once the harness runs and the tests pass.

---

## Goals

By the end of this lab, you will:

- distinguish a training objective from an evaluation metric;
- create deterministic, disjoint train/validation/test splits;
- compute a confusion matrix, accuracy, precision, recall, specificity, and F1
  without a metrics library;
- explain why a decision threshold is a product decision, not a universal
  constant;
- measure calibration instead of treating a probability as a promise;
- evaluate slices rather than trusting one aggregate number;
- detect exact duplicate leakage before reporting a result; and
- write a compact evaluation card that makes a result reproducible.

## Objective versus metric

Training minimises a **training objective**: a smooth number the optimiser can
follow, such as cross-entropy on the predicted probabilities. You report an
**evaluation metric**: the number the user cares about, such as F1 at a chosen
threshold. The two often disagree. Cross-entropy rewards every small shift in
probability; F1 changes only when a prediction crosses the threshold. A run can
lower its loss and leave F1 flat, or worse.

## Why this comes before fine-tuning

A fine-tuning run can lower loss while making the behavior you care about worse.
Before changing weights, decide:

1. **What population is the model for?**
2. **What errors have different costs?**
3. **What data is allowed to influence training decisions?**
4. **What result would convince you to ship—or to stop?**

The test set answers the last question only once. Use the training split to fit
parameters, the validation split to make choices, and the test split for the
final estimate.

```mermaid
flowchart LR
    A["raw examples"] --> B["group + deduplicate"]
    B --> C["train<br/>fit weights"]
    B --> D["validation<br/>choose threshold"]
    B --> E["test<br/>final estimate"]
    D --> F["frozen decision rule"]
    F --> E
    E --> G["aggregate + slice metrics<br/>with uncertainty"]
```

## The confusion matrix is the primitive

For a binary classifier, threshold probability `p` into a prediction:

```text
predict positive if p >= threshold
```

Every example then lands in one of four cells:

| | Actually positive | Actually negative |
|---|---:|---:|
| Predicted positive | true positive (TP) | false positive (FP) |
| Predicted negative | false negative (FN) | true negative (TN) |

All familiar metrics are summaries of these four counts:

```text
accuracy  = (TP + TN) / N
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 × precision × recall / (precision + recall)
```

Accuracy is useful when classes and error costs are reasonably balanced. It can
be actively misleading when positives are rare: a classifier that always says
"negative" is 99% accurate on a dataset with 1% positives.

## Thresholds encode costs

`0.5` is a convention, not a law. Lowering the threshold generally catches more
positives (higher recall) and raises more false alarms (lower precision).

- A screening system may prefer recall: missing a positive is costly.
- An interruptive alert may prefer precision: false alarms cause harm.
- A ranking system may not need one global threshold at all.

Choose the threshold on the **validation** set, record the rule, then evaluate
that frozen rule on the **test** set.

## Calibration: does 0.8 mean eight out of ten?

A model can rank examples correctly while being overconfident. Calibration asks:
among predictions near confidence `p`, is the observed frequency also near `p`?

Expected calibration error (ECE) is the average gap between how sure the model
says it is and how often it is right. The lab computes it in three steps:

1. bucket probabilities into equal-width bins;
2. compute mean confidence and empirical positive rate per non-empty bin;
3. take the example-weighted mean absolute gap.

ECE is a diagnostic, not a complete verdict. Its value changes with binning and
sample size, so report those choices.

## Aggregate metrics hide slices

Always inspect slices that could fail differently: device, language, class,
source, time period, or data-collection process. A model can improve overall
while regressing badly on a small but important group.

Do not use slice analysis to hunt for a flattering subgroup. Define important
slices before looking at the final test results.

A slice is also a smaller sample, so its score is noisier. Put an interval on
it before you read a gap. The lab resamples each slice's examples 2,000 times
with replacement (a bootstrap) and keeps the middle 95% of the F1 values
(`bootstrap_ci` in `src/metrics.py`):

| Slice | Examples | Positives | F1 at 0.5 | 95% interval |
|---|---:|---:|---:|---|
| daylight | 120 | 71 | 0.806 | 0.722 – 0.872 |
| low-light | 40 | 11 | 0.727 | 0.428 – 0.909 |

The gap, 0.078, has an interval from −0.111 to +0.347: it includes zero. Forty
low-light examples cannot tell 0.73 from 0.81. The honest report is "no
evidence of a gap yet; collect more low-light data", not "the model fails in
low light". Miller's [*Adding Error Bars to Evals*](https://arxiv.org/abs/2411.00640)
(2024) gives the same advice for language-model benchmarks: treat the questions
as a sample, report a standard error, and compare models on paired differences.

![Left: precision, recall and F1 against the threshold. Right: slice F1 with 95% bootstrap intervals.](figures/evaluation_scorecard.png)

## Leakage: the fastest route to a fake breakthrough

Exact duplicates are the easy case. Near-duplicates, multiple samples from the
same person, and temporal leakage are harder. Split by the unit that could leak:
patient, customer, document, conversation, or time window—not blindly by row.

The included helper checks exact ID overlap. In a real project, also hash
normalized content and look for near-duplicates. Lee et al.,
[*Deduplicating Training Data Makes Language Models Better*](https://arxiv.org/abs/2107.06499)
(2021), find them with MinHash, a cheap estimate of how many word n-grams two
texts share, and report that some standard validation sets overlapped their
training sets.

## Hands-on

Work from the lab's Python project:

```bash
cd modules/05a-data-evaluation/python
uv sync --locked
```

Run the deterministic experiment:

```bash
uv run python evaluation_lab.py
```

It writes `../figures/evaluation_scorecard.png` and `../results/evaluation_lab.json`.
It prints three things: metrics on all 160 rows at three thresholds, to show the
trade-off; the honest protocol, a 112/24/24 split with the threshold chosen on
validation (0.43) and reported once on test (F1 0.692, 95% interval 0.444 –
0.867); and the slice table above. Every number in this README comes from that
JSON file.

Run the tests:

```bash
uv run pytest -q
```

## Exercises

Skeletons live in [`python/exercises/`](python/exercises); verified answers live
in [`python/solutions/`](python/solutions).

1. **Threshold selection** — choose the validation threshold that maximizes F1,
   then report its performance on an untouched test set.
2. **Leakage detector** — find overlapping example IDs and explain why removing
   them can make a good-looking score fall.
3. **Calibration** — implement ECE. Compare two models with the same class
   predictions, first on four examples, where ECE rewards near-certainty, then
   on ten, where it punishes overconfidence. Say why.

## The evaluation card

Every result in later modules should be accompanied by these seven lines:

```text
Task:
Data and split unit:
Model/artifact revision:
Decision rule:
Primary metric and important slices:
Leakage checks:
Random seed and command:
```

This is intentionally smaller than a model card. It is the minimum receipt for
one experiment.

## Checkpoint — you should now be able to…

- [ ] reconstruct every binary metric from TP, FP, FN, and TN;
- [ ] choose a threshold using validation data without touching the test set;
- [ ] explain discrimination versus calibration;
- [ ] identify the real unit that must remain on one side of a split;
- [ ] report both aggregate and important slice metrics, with intervals; and
- [ ] produce an evaluation card another learner can reproduce.

## Sources

- Stanford CS336 — data, evaluation, and language modelling from scratch:
  <https://cs336.stanford.edu/>
- Stanford CRFM HELM — reproducible, multi-metric foundation-model evaluation:
  <https://crfm.stanford.edu/helm/>
- *Hidden Technical Debt in Machine Learning Systems*:
  <https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems>
- NIST AI Risk Management Framework and Generative AI Profile:
  <https://www.nist.gov/itl/ai-risk-management-framework>
