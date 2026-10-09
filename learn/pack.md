# ML compass pack · 2026-10-09

Built by `course/scripts/build_pack.py` from the live files of Vincent's private ML project. Seven parts, each with a one-line header saying what it is and where it comes from.

## Part 1 · Compass rules

_The whole "Compass" section of the project's `AGENTS.md`, unchanged, between a note for a chat with no tools and one line on the missing log._

You are reading this in a claude.ai chat with no tools. You cannot run lint,
`build.py` or git, and you cannot open files or links. Your job is the session
verbs exactly as the rules below define them: `next`, `walk X`, `angles`, `test`
and `intake`. Rule 5's closing steps (lint, `--apply`, the version bump, commit)
are Vincent's, at home; skip them.

The log (`log.jsonl`: his questions, test answers and slips) stays on the Mac and
is not in this pack. You do not need it: the Next box in Part 5 already orders
`test`, and Part 4 gives each entry's status and `since:` date. At the end of
every session, print the new `log.jsonl` rows in one code block, one JSON object
per line, in the exact format below, for Vincent to paste into his Apple Notes
note "ML compass log". Give each row the real date of the session: a test on a
later day may turn an entry `owned`, so the date is the proof. If the session adds
or changes a spine entry or a territory cell, print that as exact YAML in a second
block. Never invent a def and never change a status on your own: only Vincent's
words count, and a def enters only after he confirms it. The Next box is dated;
if he pastes in log rows newer than it, they win.

The row format, with one made-up row. It is an example only, not a real result:

```json
{"date": "2026-01-01", "type": "test", "entry": "example-notion", "result": "fail", "question": "Example question.", "answer": "Example answer, in his words.", "correct": "Example of the right answer.", "slip": "Example: what went wrong, in his words."}
```

A test row may also carry `quiz` (the id of the course quiz he took after a
fail; that row changes no status) and `same_day: true` (a second test of the same
notion on the same day; it changes no status). A question row has `date`, `type`
set to `question`, `q`, `text`, `entry` (a spine key or null) and `status` set to
`open`.

## Compass

The compass is one page, `compass/index.html`, built from three files Vincent can
read in full.

- `spine.yaml`: one entry per notion he has defined, capped at 150. Each has `q`
  (Q1–Q7, or F for foundations) and `angle` (one of seven). An entry needs a def in
  his words. A confirmed def is `shaky`; it becomes `owned` only after a test on a
  later day passes.
- `territory.yaml`: the seven angles; one cell per row × angle with a `view`
  sentence in his words and `candidates` (oracle ids, not yet in the spine). A
  candidate leaves the map the day its entry exists.
- `log.jsonl`: questions and test results. A test row may carry `slip` (what went
  wrong, in his words) and `quiz` (the id of the course quiz he took after a fail).
  `same_day: true` marks a second test of the same notion on the same day; it
  changes no status.
- `oracle/`: a read-only prerequisite graph, used for order, never quoted as fact;
  `oracle/README.md` lists its holes.

Rules:
1. Clarity beats completeness. If a passage is unclear, rewrite it shorter.
2. Define every term at first use. No codenames, experiment ids or paper names in
   the main text; a paper goes in a "See it" pointer.
3. Experiments follow Q7: laptop-scale, one question each, pre-registered on the page
   (expectation, measurement, what would change our mind) before any code; report
   misses too.
4. A session is one verb:
   - `next`: print the Next box and wait.
   - `walk X`: one notion, its angle and one neighbour angle, one door, twelve lines
     at most; at the door he predicts a number, moves one control and says what he
     saw; he restates; it enters shaky.
   - `angles Qn`: one question through the seven angles; he keeps the sentences he
     wants. `angle A`: one angle across the seven questions.
   - `test`: three questions, in the order the Next box prints; keep wrong answers;
     pass → owned, fail → shaky. After a wrong answer he says in his own words what
     went wrong; that goes in `slip`. Then he takes the first unused quiz from the
     notion's `test:` list in his browser and reports the result in a new row, which
     carries `quiz` and changes no status. Where no quiz fits the slip, the agent
     asks one new question on that slip, in another form. One follow-up per fail.
   - `intake <brief>`: a brief → cells, candidates, sentences, for his yes.
   - `rent`: lint and rot report; he decides deletes.
   - `experiment`: parked; rule 3 when it returns.
5. A session ends in this order: one diff line, lint clean, `build.py --apply`, the
   version bump, commit. `--apply` names its backup after the version still in the
   file, so bumping first mislabels it; read the backup path it prints. A sitting
   where `build.py` alone shows no region differs, except the date on the NEXT line,
   skips `--apply` and the bump.

The page: regions between `<!-- gen:x -->` markers come from `build.py --apply`;
never hand-edit them. Prose outside them is hand-written. Check the page with an
HTML parser; agents cannot open it in a browser. Never make a second page or a copy
unless he asks. `compass/proposals/current.html` holds the live proposal; a decided
one takes its date as its name.

Tools, run from this folder (plain `python3` lacks PyYAML):

| Command | Does |
|---|---|
| `uv run --project compass/tools compass/tools/lint.py` | checks spine and territory, and that every door exists |
| `uv run --project compass/tools compass/tools/build.py --next` | prints the Next box |
| `uv run --project compass/tools compass/tools/build.py --apply` | writes the generated regions |
| `uv run --project compass/tools compass/tools/oracle.py` | asks the graph what comes first |
| `uv run --project compass/experiments python compass/experiments/<run>/run.py` | runs one experiment; one shared env |

`compass/tools/out/` holds local page backups and is not tracked.

_Without the log: rule 4's "first unused quiz" cannot be checked here. Ask Vincent
which quizzes of the notion's `test:` list he has taken, and give the first one
he has not._

## Part 2 · spine.yaml

_`compass/spine.yaml`, whole: every notion Vincent has defined, in his words._

```yaml
# needs = direct oracle edges only: A is in B needs when A's kg node is listed in B's kg `from`.
# The structure of the compass: every notion Vincent owns, in page order.
# Cap: 150 entries. Past that, the page is no longer one page.
# Each entry carries `q` (Q1..Q7, or F for foundations; no F entries yet) and
# `angle` (one of the seven in territory.yaml).
# An entry needs a def in Vincent's words. A def he confirms is shaky; it turns
# owned when a test on a later day passes.

parameter:
  term: parameter
  def: "one adjustable number inside the model; modern models have billions."
  q: Q1
  angle: algebra
  needs: []
  see: [course/explorables/01-gradient-descent.html, course/explorables/02-nn-playground.html]
  test: [quiz-01-q6, quiz-02-q6, quiz-04-q4, quiz-05-q3, quiz-06-q2]
  kg: 6390361d
  status: shaky
  since: 2026-08-25

loss:
  term: loss
  def: "the gap between what the model predicts and what it should have predicted."
  q: Q1
  angle: measurement
  needs: []
  see: course/explorables/01-gradient-descent.html
  test: []
  kg: 2f0b4053
  status: shaky
  since: 2026-09-26

gradient:
  term: gradient
  def: "how each of the model's parameters contributes to that prediction gap."
  q: Q1
  angle: calculus
  needs: []
  see: course/explorables/01-gradient-descent.html
  test: [quiz-01-q2, quiz-01-q4, quiz-01-q5]
  read: wiki/source-frontier-learning-edge-of-capability.md
  kg: 1b6a3443
  status: shaky
  since: 2026-09-26

backpropagation:
  term: backpropagation
  def: "the bookkeeping trick that computes every gradient in one backward sweep."
  q: Q1
  angle: calculus
  needs: [loss]
  see: course/explorables/01-gradient-descent.html
  test: [quiz-01-q1, quiz-01-q10]
  read: wiki/source-pc-alm-augmented-lagrangian-predictive-coding.md
  kg: 2bd063e3
  status: owned
  since: 2026-09-26

learning-rate:
  term: learning rate
  def: "the step size of each downhill nudge."
  q: Q1
  angle: dynamics
  needs: []
  see: course/explorables/01-gradient-descent.html
  test: [quiz-01-q7, quiz-01-q9]
  kg: 8212edf3
  status: shaky
  since: 2026-08-25

overfitting:
  term: overfitting
  def: "perfect on training data, poor on new data."
  q: Q2
  angle: measurement
  needs: []
  see: course/explorables/00c-double-descent.html
  test: []
  read: wiki/source-rrsi-regularized-harness-evolution.md
  moved: [exp-03-grokking]
  kg: a1f32f57
  status: shaky
  since: 2026-08-25

double-descent:
  term: double descent
  def: "test error spikes exactly where parameters = data points, then <em>falls again</em> as models grow past it."
  q: Q2
  angle: algebra
  needs: []
  see: course/explorables/00c-double-descent.html
  test: [quiz-00c-q7, quiz-00c-q9]
  kg: null
  status: shaky
  since: 2026-08-25

implicit-bias:
  term: implicit bias
  def: "training's hidden preference for the simplest perfect fit — nobody programs it in."
  q: Q2
  angle: algebra
  needs: []
  see: course/explorables/00c-double-descent.html
  test: [quiz-00c-q10]
  read: wiki/source-abstract-token-curriculum.md
  kg: null
  status: shaky
  since: 2026-08-25

spectral-bias:
  term: spectral bias
  def: "networks learn low frequencies first, fine detail last."
  q: Q2
  angle: dynamics
  needs: []
  see: course/explorables/00c-double-descent.html
  test: []
  moved: [exp-01-spectral-bias]
  kg: null
  status: shaky
  since: 2026-08-25

inverse-problem:
  term: inverse problem
  def: "outputs known, causes unknown, usually more than one answer."
  q: Q2
  angle: algebra
  needs: [loss, implicit-bias]
  test: []
  kg: null
  status: shaky
  since: 2026-09-05

embedding:
  term: embedding
  def: "the learned map from raw input to a point in vector space."
  q: Q3
  angle: algebra
  needs: []
  see: course/modules/05-transformers-library/figures/embedding_pca.png
  test: []
  read: wiki/concept-representation-learning.md
  kg: aa580156
  status: shaky
  since: 2026-08-25

latent-space:
  term: latent space
  def: "the vector space where those points live; \"latent\" = hidden, internal."
  q: Q3
  angle: geometry
  needs: []
  see: course/modules/05-transformers-library/figures/embedding_pca.png
  test: []
  read: wiki/concept-representation-learning.md
  kg: ae8f9598
  status: shaky
  since: 2026-08-25

feature:
  term: feature
  def: "a direction in that space that means something."
  q: Q3
  angle: algebra
  needs: []
  see: course/modules/05-transformers-library/figures/embedding_pca.png
  test: []
  read: wiki/concept-representation-learning.md
  kg: null
  status: shaky
  since: 2026-08-25

collapse:
  term: collapse
  def: "the degenerate win: all inputs map to one point, loss looks great."
  q: Q3
  angle: geometry
  needs: []
  see: ../ml-lab/gridjepa/runs/20260721_134506_phase1/report.md
  test: []
  read: wiki/concept-self-supervised-learning.md
  moved: [gridjepa-e2]
  kg: null
  status: shaky
  since: 2026-08-25

probe:
  term: probe
  def: "a small model trained to read a quantity out of the representation — an instrument, not a component."
  q: Q3
  angle: measurement
  needs: []
  see: ../ml-lab/gridjepa/runs/20260721_134506_phase1/report.md
  test: []
  moved: [gridjepa-e2]
  kg: null
  status: shaky
  since: 2026-08-25

manifold-hypothesis:
  term: manifold hypothesis
  def: "real data lies near a thin surface inside its raw coordinates."
  q: Q3
  angle: geometry
  needs: [embedding]
  test: []
  kg: null   # 8cf1e789 "manifold learning" is the algorithm family, not this claim
  status: shaky
  since: 2026-09-05

attention:
  term: attention
  def: "the soft lookup: queries score keys, values get blended."
  q: Q4
  angle: algebra
  needs: [softmax]
  see: course/explorables/04-attention.html
  test: [quiz-00c-q6, quiz-04-q9]
  read: wiki/concept-transformer-architecture.md
  kg: b0a47943
  status: shaky
  since: 2026-08-25

softmax:
  term: softmax
  def: "turns any list of scores into probabilities that sum to 1; sharpness set by temperature."
  q: Q4
  angle: probability
  needs: []
  see: course/explorables/04-attention.html
  test: [quiz-02-q1, quiz-04-q7, quiz-04-q10]
  moved: [exp-02-temperature]
  kg: 2acd5168
  status: shaky
  since: 2026-08-25

temperature:
  term: temperature
  def: "low = commit to the top choice, if one clearly wins — experiment 2 found a near-tie stays random; high = spread the bets."
  q: Q4
  angle: probability
  needs: []
  see: course/explorables/04-attention.html
  test: [quiz-04-q7]
  read: wiki/concept-calibration.md
  moved: [exp-02-temperature]
  kg: null
  status: shaky
  since: 2026-08-25

composition:
  term: composition
  def: "building new abstractions from previous ones, and reusing them."
  q: Q4
  angle: computation
  needs: []
  test: []
  read: wiki/source-zhang-harness-blog.md
  kg: 8c289766
  status: shaky
  since: 2026-08-25

residual-stream:
  term: residual stream
  def: "the shared bus running through the model; every layer adds its correction to it."
  q: Q4
  angle: algebra
  needs: []
  see: course/explorables/05-transformer-anatomy.html
  test: [quiz-04-q2]
  read: wiki/concept-transformer-architecture.md
  moved: [exp-04-fixed-point]
  kg: 97c5b137
  status: shaky
  since: 2026-08-25

scaling-law:
  term: scaling law
  def: "the measured power-law linking loss to size, data, and compute."
  q: Q4
  angle: measurement
  needs: []
  test: []
  read: wiki/concept-scaling-laws.md
  kg: null
  status: shaky
  since: 2026-08-25

fixed-point:
  term: fixed point
  def: "a state the next step leaves unchanged."
  q: Q4
  angle: dynamics
  needs: [residual-stream]
  see: compass/experiments/04-fixed-point/RESULTS.md
  test: []
  read: wiki/source-loopcd-decoding-looped-transformers.md
  moved: [exp-04-fixed-point]
  kg: 0fe102e7
  status: shaky
  since: 2026-09-05

state-space-model:
  term: state-space model
  def: "a network that carries a fixed-size running state instead of re-reading the whole past."
  q: Q5
  angle: dynamics
  needs: []
  see: ../ml-lab/gridjepa/README.md
  test: []
  kg: 5b840157
  status: shaky
  since: 2026-08-25

jepa:
  term: JEPA
  def: "joint-embedding predictive architecture: predict the next <em>representation</em>, never the raw input."
  q: Q5
  angle: geometry
  needs: []
  see: [../ml-lab/gridjepa/README.md, ../ml-lab/levjepa-mobile/REPORT.md]
  test: []
  read: wiki/concept-jepa.md
  kg: 2a26e277
  status: shaky
  since: 2026-08-25

mixture-of-experts:
  term: mixture-of-experts
  def: "many sub-networks; a router picks a few per input."
  q: Q5
  angle: computation
  needs: []
  see: ../ml-lab/gridjepa/README.md
  test: []
  read: wiki/concept-mixture-of-experts.md
  kg: 49b907ab
  status: shaky
  since: 2026-08-25

inductive-bias:
  term: inductive bias
  def: "the assumptions an architecture bakes in before seeing any data."
  q: Q5
  angle: probability
  needs: []
  see: ../ml-lab/gridjepa/README.md
  test: [quiz-08-q1]
  read: wiki/source-local-support-learning.md
  kg: null
  status: shaky
  since: 2026-08-25

diffusion:
  term: diffusion
  def: "generate by learning to undo noise, step by step, coarse to fine."
  q: Q5
  angle: probability
  needs: []
  see: ../ml-lab/gridjepa/README.md
  test: [quiz-09-q1, quiz-09-q2, quiz-09-q3, quiz-09-q4]
  read: wiki/concept-diffusion-language-models.md
  moved: [exp-01-spectral-bias]
  kg: f526cc7e
  status: shaky
  since: 2026-08-25

interpretability:
  term: interpretability
  def: "reading a model's internals to find the mechanism, not just the behavior."
  q: Q6
  angle: measurement
  needs: []
  see: course/explorables/05-transformer-anatomy.html
  test: [quiz-05-q2]
  read: wiki/concept-mechanistic-interpretability.md
  moved: [exp-03-grokking]
  kg: null
  status: shaky
  since: 2026-08-25

calibration:
  term: calibration
  def: "agreement between stated confidence and actual frequency of being right."
  q: Q6
  angle: probability
  needs: []
  see: course/modules/05a-data-evaluation/README.md#calibration-does-08-mean-eight-out-of-ten
  test: [quiz-05a-q5, quiz-05a-q8]
  kg: e152337e
  status: shaky
  since: 2026-08-25

hallucination:
  term: hallucination
  def: "a confident claim with nothing behind it."
  q: Q6
  angle: probability
  needs: []
  test: []
  moved: [crystallizer-v0]
  kg: null
  status: shaky
  since: 2026-08-25

self-certification:
  term: self-certification
  def: "a system grading itself on a quantity it controls."
  q: Q6
  angle: measurement
  needs: []
  see: ../ml-lab/gridjepa/runs/20260721_134506_phase1/report.md
  test: []
  moved: [gridjepa-e2]
  kg: null
  status: shaky
  since: 2026-08-25

pre-registration:
  term: pre-registration
  def: "writing down the expectation and the measurement before running anything."
  q: Q7
  angle: measurement
  needs: []
  test: []
  kg: null
  status: shaky
  since: 2026-08-25

falsifiable:
  term: falsifiable
  def: "the claim names what would prove it wrong."
  q: Q7
  angle: measurement
  needs: []
  test: []
  kg: null
  status: shaky
  since: 2026-08-25

baseline:
  term: baseline
  def: "the dumb alternative your result must beat to mean anything."
  q: Q7
  angle: measurement
  needs: []
  test: []
  kg: null
  status: shaky
  since: 2026-08-25
```

## Part 3 · territory.yaml

_`compass/territory.yaml`, whole: the seven angles and the map cells._

```yaml
# The map. Seven questions down (rows), seven angles across (columns).
# A question is what you want to understand; an angle is the toolkit you look with.
# Row F holds the mathematics under every column.
#
# A cell holds two things:
#   view        one sentence in your own words, 20 words at most. Empty until you write it.
#   candidates  kebab id -> oracle node id, or ~ when the graph has no node for it.
#               These are on the map, not yours yet.
#
# A candidate leaves the map the day its spine entry exists. Nothing else deletes it.
# Cells with no view and no candidates are left out; they read as blank on the page.

angles:
  algebra:     vectors, matrices, what a linear map does
  calculus:    slopes, the chain rule, what a small step changes
  probability: chance, belief, information
  geometry:    distance, shape, where the data lies
  dynamics:    energy, iteration, what settles
  computation: what runs, what it costs
  measurement: what we can check, how we fool ourselves

rows: [F, Q1, Q2, Q3, Q4, Q5, Q6, Q7]

# Display names, for every candidate the kebab id spells wrong: acronyms,
# capitals, hyphens and apostrophes the id cannot carry.
labels:
  svd: SVD
  kl-divergence: KL divergence
  pca: PCA
  big-o-cost: big-O cost
  sgd-noise: SGD noise
  bayes-rule: "Bayes' rule"
  gaussian: Gaussian
  hebbian-rule: Hebbian rule
  hessian: Hessian
  taylor-expansion: Taylor expansion
  cross-entropy: cross-entropy
  cross-validation: cross-validation
  held-out-set: held-out set
  k-nearest-neighbours: k-nearest neighbours
  mean-squared-error: mean squared error

cells:

  F:
    algebra:
      view: ""
      candidates:
        vector: b0d51c58
        dot-product: e01902fe
        basis: a7a3c8f6
        linear-map: c2e6dacb
        matrix: 6e00cd56
        matrix-product: 47018079
        eigenvector: c0ba48a5
        svd: b9afcb2d
        rank: 5a8460f6
        covariance: d797a4fb
    calculus:
      view: ""
      candidates:
        derivative: 1c7ed7db
        partial-derivative: f6452833
        chain-rule: 16f3f581
        taylor-expansion: 527e4637
        hessian: acdb1b86
        convexity: ab79b47e
        limit: 55ea09e5
        integral: 1a20f6d5
    probability:
      view: ""
      candidates:
        probability: 3955f8fd
        conditional-probability: 6e67f84d
        bayes-rule: 00af80da
        random-variable: 484bb48c
        distribution: ba3bcd04
        expectation: ff4fd532
        variance: 7c0766a4
        law-of-large-numbers: ~
        gaussian: 924895d1
        likelihood: 0f66d507
        entropy: 67671a2f
        cross-entropy: c796b9d1
        kl-divergence: 74ec9dbb
    geometry:
      view: ""
      candidates:
        distance: 7408d918
        norm: 733859c1
        hyperplane: 7ffa0136
        margin: 34e39f66
        kernel: 6923dd1b
        nearest-neighbour: d4cae8c5
        curse-of-dimensionality: 8c754a45
        manifold: ~          # "manifold learning" is a family of algorithms, not the object
    dynamics:
      view: ""
      candidates:
        iteration: aa82c330
        energy-function: 839bed28
        attractor: ~
        hebbian-rule: 4467f861
        phase-transition: ~
    computation:
      view: ""
      candidates:
        algorithm: b1eb2ec8
        big-o-cost: 1a95f2bc
        computational-graph: 207d4c0d
        floating-point: ~
        parallelism: d05e059e
    measurement:
      view: ""
      candidates:
        held-out-set: ~
        cross-validation: 14fdc6d2
        metric: 2cde3824
        mean-squared-error: d8a681bc
        hypothesis-test: acae5c7b
        base-rate: ~
        confidence-interval: f3279006

  Q1:
    algebra:
      view: ""
      candidates:
        neural-network: 4460d5ad
        perceptron: d6cc6bc3
        linear-regression: 9f618331
    calculus:
      view: ""
      candidates:
        gradient-descent: 4c36741a
        activation: 82e0526f
    probability:
      view: ""
      candidates:
        sgd-noise: f928c592
    dynamics:
      view: ""
      candidates:
        random-init: 05fd548c

  Q2:
    calculus:
      view: ""
      candidates:
        regularization: 2696fbf2
    probability:
      view: ""
      candidates:
        bias-variance: ~

  Q3:
    algebra:
      view: ""
      candidates:
        pca: 444b1ac2
    calculus:
      view: ""
      candidates:
        universal-approximation: d1a6fb74

  Q5:
    algebra:
      view: ""
      candidates:
        convolution: 9ca01c94
    geometry:
      view: ""
      candidates:
        k-nearest-neighbours: c1ec92e2
        support-vector-machine: 9ea1cd15
    dynamics:
      view: ""
      candidates:
        hopfield-network: 038e7826

  Q7:
    measurement:
      view: ""
      candidates:
        hyperparameter: a8131908

# Reading order: Why Machines Learn (Ananthaswamy, 2024). One chapter per intake session.
book:
  - {ch: 1,  idea: "the perceptron finds a separating line",        cells: [Q1/algebra, F/geometry]}
  - {ch: 2,  idea: "data as vectors, dot product, the weight vector", cells: [F/algebra]}
  - {ch: 3,  idea: "the bottom of the bowl: gradient descent, least squares", cells: [F/calculus, Q1/calculus, F/measurement]}
  - {ch: 4,  idea: "Bayes, distributions, expectation, variance",   cells: [F/probability]}
  - {ch: 5,  idea: "distance, nearest neighbours, the curse of dimensionality", cells: [F/geometry, Q5/geometry, Q2/measurement]}
  - {ch: 6,  idea: "eigenvectors, covariance, PCA",                 cells: [F/algebra, Q3/algebra]}
  - {ch: 7,  idea: "margins, kernels, the kernel trick, SVM",       cells: [F/geometry, Q5/geometry]}
  - {ch: 8,  idea: "energy, Hopfield memory, Hebb's rule",          cells: [F/dynamics, Q5/dynamics, Q4/dynamics]}
  - {ch: 9,  idea: "XOR, hidden layers, universal approximation",   cells: [Q3/calculus, Q1/calculus]}
  - {ch: 10, idea: "backpropagation, the chain rule, random init",  cells: [F/calculus, Q1/calculus, Q1/dynamics]}
  - {ch: 11, idea: "convolution, pooling, receptive fields",        cells: [Q5/algebra]}
  - {ch: 12, idea: "bias-variance, double descent, regularization", cells: [Q2/probability, Q2/algebra, Q2/calculus]}
```

## Part 4 · Status

_From `compass/spine.yaml`: each entry's status and `since:` date, and the count per status. Nothing from `log.jsonl`, which stays on the Mac._

35 entries: owned 1, shaky 34.

| Term | Status | Since |
|---|---|---|
| parameter | shaky | 2026-08-25 |
| loss | shaky | 2026-09-26 |
| gradient | shaky | 2026-09-26 |
| backpropagation | owned | 2026-09-26 |
| learning rate | shaky | 2026-08-25 |
| overfitting | shaky | 2026-08-25 |
| double descent | shaky | 2026-08-25 |
| implicit bias | shaky | 2026-08-25 |
| spectral bias | shaky | 2026-08-25 |
| inverse problem | shaky | 2026-09-05 |
| embedding | shaky | 2026-08-25 |
| latent space | shaky | 2026-08-25 |
| feature | shaky | 2026-08-25 |
| collapse | shaky | 2026-08-25 |
| probe | shaky | 2026-08-25 |
| manifold hypothesis | shaky | 2026-09-05 |
| attention | shaky | 2026-08-25 |
| softmax | shaky | 2026-08-25 |
| temperature | shaky | 2026-08-25 |
| composition | shaky | 2026-08-25 |
| residual stream | shaky | 2026-08-25 |
| scaling law | shaky | 2026-08-25 |
| fixed point | shaky | 2026-09-05 |
| state-space model | shaky | 2026-08-25 |
| JEPA | shaky | 2026-08-25 |
| mixture-of-experts | shaky | 2026-08-25 |
| inductive bias | shaky | 2026-08-25 |
| diffusion | shaky | 2026-08-25 |
| interpretability | shaky | 2026-08-25 |
| calibration | shaky | 2026-08-25 |
| hallucination | shaky | 2026-08-25 |
| self-certification | shaky | 2026-08-25 |
| pre-registration | shaky | 2026-08-25 |
| falsifiable | shaky | 2026-08-25 |
| baseline | shaky | 2026-08-25 |

## Part 5 · Next box

_Printed by `build.py --next` on 2026-10-09._

```text
NEXT · 2026-10-09
learn  algorithm (F·computation) · parallelism (F·computation) · limit (F·calculus)
test   parameter (2026-08-25) · learning rate (2026-08-25) · overfitting (2026-08-25)
fill   Q7·measurement · Q2·algebra · Q3·geometry
open   16 open · 2026-09-05 Q1 What landscape does my data build?
```

## Part 6 · Course quizzes

_From `course/quizzes/`. A spine `test:` id `quiz-<quiz>-q<n>` means question n of that quiz. URL: `https://wynch.github.io/ml-course/quizzes/<file>`; the page shows all ten questions, so name the question number and its topic._

### quiz-00a · The perceptron & least squares

https://wynch.github.io/ml-course/quizzes/00a.html

- `quiz-00a-q1` the bias as a feature
- `quiz-00a-q2` mistakes versus the bound
- `quiz-00a-q3` why a tie counts as a mistake
- `quiz-00a-q4` the shape of Novikoff
- `quiz-00a-q5` how the margin scales the work
- `quiz-00a-q6` the XOR obstruction
- `quiz-00a-q7` the residual is orthogonal
- `quiz-00a-q8` evaluating the bound
- `quiz-00a-q9` the gradient-descent stability edge
- `quiz-00a-q10` least squares as projection

### quiz-00b · Probability, neighbours & eigenvectors

https://wynch.github.io/ml-course/quizzes/00b.html

- `quiz-00b-q1` Base rates and the posterior
- `quiz-00b-q2` How compressible FashionMNIST is
- `quiz-00b-q3` Where naive Bayes wins and loses
- `quiz-00b-q4` Where the naive assumption lives
- `quiz-00b-q5` The sign guard in power iteration
- `quiz-00b-q6` What Cover-Hart actually says
- `quiz-00b-q7` What controls power-iteration speed
- `quiz-00b-q8` What k = 0 reconstructs
- `quiz-00b-q9` Reading the PCA results
- `quiz-00b-q10` Train error at k = 1

### quiz-00c · Kernels, memory & the modern bridge

https://wynch.github.io/ml-course/quizzes/00c.html

- `quiz-00c-q1` Why only support vectors matter
- `quiz-00c-q2` One solver, three kernels
- `quiz-00c-q3` Measuring a margin without w
- `quiz-00c-q4` Why Hopfield energy can only fall
- `quiz-00c-q5` Measured Hopfield capacity
- `quiz-00c-q6` The attention-Hopfield bridge
- `quiz-00c-q7` Where the spike lands
- `quiz-00c-q8` Two lifts, two kernels
- `quiz-00c-q9` Reading the double-descent curve honestly
- `quiz-00c-q10` Minimum norm is a statement about coordinates

### quiz-01 · Autograd from scratch

https://wynch.github.io/ml-course/quizzes/01.html

- `quiz-01-q1` reverse topological order
- `quiz-01-q2` gradient accumulation (+=)
- `quiz-01-q3` the product rule in __mul__
- `quiz-01-q4` the ReLU sub-gradient at 0
- `quiz-01-q5` how close the gradient check gets
- `quiz-01-q6` parameter count of the moons MLP
- `quiz-01-q7` learning rates past the stability edge
- `quiz-01-q8` what momentum actually changes
- `quiz-01-q9` the decaying learning-rate schedule
- `quiz-01-q10` what the engine does and does not store

### quiz-02 · Neural networks & the training loop

https://wynch.github.io/ml-course/quizzes/02.html

- `quiz-02-q1` why softmax + cross-entropy is fused
- `quiz-02-q2` shapes through Linear.backward
- `quiz-02-q3` why there is no activation on the logits
- `quiz-02-q4` FashionMNIST test accuracy
- `quiz-02-q5` matmul speedups: cache order vs BLAS
- `quiz-02-q6` parameter count of the 2-16-16-2 net
- `quiz-02-q7` dying ReLU at tiny width
- `quiz-02-q8` momentum multiplies the effective step
- `quiz-02-q9` He initialization
- `quiz-02-q10` what the loop and the optimizers really do

### quiz-03 · Tokenization: from bytes to tokens

https://wynch.github.io/ml-course/quizzes/03.html

- `quiz-03-q1` Byte-level base vocabulary
- `quiz-03-q2` The deterministic tie-break
- `quiz-03-q3` Why the Zig lane is faster
- `quiz-03-q4` The Python/Zig scoreboard
- `quiz-03-q5` Vocabulary size after training
- `quiz-03-q6` In-domain vs out-of-domain compression
- `quiz-03-q7` What merging actually shrinks
- `quiz-03-q8` Merge order is the model
- `quiz-03-q9` Non-overlapping merge semantics
- `quiz-03-q10` Encoding replays merges in order

### quiz-04 · Attention & the transformer

https://wynch.github.io/ml-course/quizzes/04.html

- `quiz-04-q1` Why the √d_head
- `quiz-04-q2` Pre-norm residual blocks
- `quiz-04-q3` Weight tying and the export
- `quiz-04-q4` Parameter count
- `quiz-04-q5` Final train/val loss
- `quiz-04-q6` What the model predicts after a speaker tag
- `quiz-04-q7` Temperature and the softmax
- `quiz-04-q8` The future-leak probe
- `quiz-04-q9` Attention tensor shapes
- `quiz-04-q10` Masking with −inf, before the softmax

### quiz-05 · The transformers library

https://wynch.github.io/ml-course/quizzes/05.html

- `quiz-05-q1` GQA and the KV cache
- `quiz-05-q2` The logit lens projection
- `quiz-05-q3` The parameter budget
- `quiz-05-q4` The decision layer
- `quiz-05-q5` MPS vs CPU at batch 1
- `quiz-05-q6` top-k vs top-p
- `quiz-05-q7` Hand-rolled greedy decoding
- `quiz-05-q8` How the budget scales with model size
- `quiz-05-q9` Reading a chat template
- `quiz-05-q10` What layer 0 of the lens shows

### quiz-05a · Data & evaluation

https://wynch.github.io/ml-course/quizzes/05a.html

- `quiz-05a-q1` Why the threshold is chosen on validation
- `quiz-05a-q2` The >= in confusion_counts
- `quiz-05a-q3` What leakage does to the scorecard
- `quiz-05a-q4` What an aggregate score hides
- `quiz-05a-q5` Computing ECE by hand
- `quiz-05a-q6` The tie-breaking contract
- `quiz-05a-q7` Raising the threshold
- `quiz-05a-q8` What ECE actually rewards
- `quiz-05a-q9` What split_indices guarantees
- `quiz-05a-q10` Choosing the unit that must not leak

### quiz-06 · Fine-tuning: make the model yours

https://wynch.github.io/ml-course/quizzes/06.html

- `quiz-06-q1` Why B starts at zero
- `quiz-06-q2` The trainable-parameter budget
- `quiz-06-q3` The real loss curve
- `quiz-06-q4` Reading the LoRA config
- `quiz-06-q5` Trainer logging cadence
- `quiz-06-q6` The SFT objective
- `quiz-06-q7` The low intrinsic rank of ΔW
- `quiz-06-q8` When factorisation stops paying
- `quiz-06-q9` Reading before/after honestly
- `quiz-06-q10` MLX vs PyTorch/MPS

### quiz-07 · Inference internals

https://wynch.github.io/ml-course/quizzes/07.html

- `quiz-07-q1` Decode & the vanishing causal mask
- `quiz-07-q2` The tied LM head
- `quiz-07-q3` What the KV cache actually holds
- `quiz-07-q4` What int8 actually buys you
- `quiz-07-q5` Parity against PyTorch
- `quiz-07-q6` The measured KV-cache speedup
- `quiz-07-q7` Linear vs quadratic decode, counted
- `quiz-07-q8` How the speedup scales with context
- `quiz-07-q9` LayerNorm, and the two details that break parity
- `quiz-07-q10` top-p (nucleus) sampling

### quiz-08 · Vision: convolutions to ViTs

https://wynch.github.io/ml-course/quizzes/08.html

- `quiz-08-q1` The convolutional inductive bias
- `quiz-08-q2` Why ViTs need position embeddings
- `quiz-08-q3` ViT ↔ text transformer, row by row
- `quiz-08-q4` Fine-tune vs zero-shot on beans
- `quiz-08-q5` The patch sequence, counted
- `quiz-08-q6` The patch-size trade-off
- `quiz-08-q7` What a kernel
- `quiz-08-q8` patchify shapes
- `quiz-08-q9` same vs valid padding
- `quiz-08-q10` How CLIP zero-shot actually scores

### quiz-09 · Diffusion: learning to denoise

https://wynch.github.io/ml-course/quizzes/09.html

- `quiz-09-q1` the forward process & its closed form
- `quiz-09-q2` the ε-prediction objective
- `quiz-09-q3` the learned denoising field / score
- `quiz-09-q4` what diffusion actually learns
- `quiz-09-q5` cosine vs linear noise schedule
- `quiz-09-q6` size of the 2D denoiser
- `quiz-09-q7` schedule choice at sampling time
- `quiz-09-q8` ancestral sampling is stochastic
- `quiz-09-q9` the reverse (ancestral) sampling step
- `quiz-09-q10` UNet skip connections and shapes

### quiz-10 · Agents: models that act

https://wynch.github.io/ml-course/quizzes/10.html

- `quiz-10-q1` the ReAct loop and how it terminates
- `quiz-10-q2` why generation stops at “Observation:”
- `quiz-10-q3` code-as-action vs text tool-calls
- `quiz-10-q4` small-model failure modes
- `quiz-10-q5` reading the eval suite results
- `quiz-10-q6` the suite
- `quiz-10-q7` the looping failure mode
- `quiz-10-q8` how code-as-action collapses turns
- `quiz-10-q9` action parsing and the cost of format drift
- `quiz-10-q10` safe tool design

## Part 7 · URL map

_Where things are on the public site, so you can give Vincent links. Doors outside `course/` are private and have no URL._

| What | URL |
|---|---|
| Learn page | https://wynch.github.io/ml-course/learn/ |
| Course home | https://wynch.github.io/ml-course/ |
| Reader (all lessons) | https://wynch.github.io/ml-course/reader/ |
| Reader, one module | https://wynch.github.io/ml-course/reader/#/<module-slug>, e.g. https://wynch.github.io/ml-course/reader/#/01-autograd |
| Explorables | https://wynch.github.io/ml-course/explorables/index.html |
| One explorable | https://wynch.github.io/ml-course/explorables/<file>; spine door `course/explorables/X` maps to it |
| Quizzes | https://wynch.github.io/ml-course/quizzes/index.html |
| Course map | https://wynch.github.io/ml-course/map.html |
| Compass page (snapshot) | https://wynch.github.io/ml-course/compass/ |
| Wiki index, with search | https://wynch.github.io/ml-course/wiki/ |
| One wiki page | https://wynch.github.io/ml-course/wiki/<name>.html; spine `read: wiki/<name>.md` maps to it |
| This pack | https://wynch.github.io/ml-course/learn/pack.md |


Modules: `00a-perceptron` The perceptron & least squares, `00b-bayes-knn-pca` Probability, neighbours & eigenvectors, `00c-kernels-hopfield` Kernels, memory & the modern bridge, `01-autograd` Autograd from scratch, `02-neural-networks` Neural networks & the training loop, `03-tokenization` Tokenization: from bytes to tokens, `04-attention-transformer` Attention & the transformer, `05-transformers-library` The transformers library, `05a-data-evaluation` Data & evaluation, `06-fine-tuning` Fine-tuning: make the model yours, `07-inference-internals` Inference internals, `08-vision` Vision: convolutions to ViTs, `09-diffusion` Diffusion: learning to denoise, `10-agents` Agents: models that act.


Doors per spine entry:

| Entry | Status | Door | Wiki |
|---|---|---|---|
| parameter | shaky | https://wynch.github.io/ml-course/explorables/01-gradient-descent.html<br>https://wynch.github.io/ml-course/explorables/02-nn-playground.html | none |
| loss | shaky | https://wynch.github.io/ml-course/explorables/01-gradient-descent.html | none |
| gradient | shaky | https://wynch.github.io/ml-course/explorables/01-gradient-descent.html | https://wynch.github.io/ml-course/wiki/source-frontier-learning-edge-of-capability.html |
| backpropagation | owned | https://wynch.github.io/ml-course/explorables/01-gradient-descent.html | https://wynch.github.io/ml-course/wiki/source-pc-alm-augmented-lagrangian-predictive-coding.html |
| learning rate | shaky | https://wynch.github.io/ml-course/explorables/01-gradient-descent.html | none |
| overfitting | shaky | https://wynch.github.io/ml-course/explorables/00c-double-descent.html | https://wynch.github.io/ml-course/wiki/source-rrsi-regularized-harness-evolution.html |
| double descent | shaky | https://wynch.github.io/ml-course/explorables/00c-double-descent.html | none |
| implicit bias | shaky | https://wynch.github.io/ml-course/explorables/00c-double-descent.html | https://wynch.github.io/ml-course/wiki/source-abstract-token-curriculum.html |
| spectral bias | shaky | https://wynch.github.io/ml-course/explorables/00c-double-descent.html | none |
| inverse problem | shaky | none | none |
| embedding | shaky | https://wynch.github.io/ml-course/modules/05-transformers-library/figures/embedding_pca.png | https://wynch.github.io/ml-course/wiki/concept-representation-learning.html |
| latent space | shaky | https://wynch.github.io/ml-course/modules/05-transformers-library/figures/embedding_pca.png | https://wynch.github.io/ml-course/wiki/concept-representation-learning.html |
| feature | shaky | https://wynch.github.io/ml-course/modules/05-transformers-library/figures/embedding_pca.png | https://wynch.github.io/ml-course/wiki/concept-representation-learning.html |
| collapse | shaky | private: `../ml-lab/gridjepa/runs/20260721_134506_phase1/report.md` | https://wynch.github.io/ml-course/wiki/concept-self-supervised-learning.html |
| probe | shaky | private: `../ml-lab/gridjepa/runs/20260721_134506_phase1/report.md` | none |
| manifold hypothesis | shaky | none | none |
| attention | shaky | https://wynch.github.io/ml-course/explorables/04-attention.html | https://wynch.github.io/ml-course/wiki/concept-transformer-architecture.html |
| softmax | shaky | https://wynch.github.io/ml-course/explorables/04-attention.html | none |
| temperature | shaky | https://wynch.github.io/ml-course/explorables/04-attention.html | https://wynch.github.io/ml-course/wiki/concept-calibration.html |
| composition | shaky | none | https://wynch.github.io/ml-course/wiki/source-zhang-harness-blog.html |
| residual stream | shaky | https://wynch.github.io/ml-course/explorables/05-transformer-anatomy.html | https://wynch.github.io/ml-course/wiki/concept-transformer-architecture.html |
| scaling law | shaky | none | https://wynch.github.io/ml-course/wiki/concept-scaling-laws.html |
| fixed point | shaky | private: `compass/experiments/04-fixed-point/RESULTS.md` | https://wynch.github.io/ml-course/wiki/source-loopcd-decoding-looped-transformers.html |
| state-space model | shaky | private: `../ml-lab/gridjepa/README.md` | none |
| JEPA | shaky | private: `../ml-lab/gridjepa/README.md`<br>private: `../ml-lab/levjepa-mobile/REPORT.md` | https://wynch.github.io/ml-course/wiki/concept-jepa.html |
| mixture-of-experts | shaky | private: `../ml-lab/gridjepa/README.md` | https://wynch.github.io/ml-course/wiki/concept-mixture-of-experts.html |
| inductive bias | shaky | private: `../ml-lab/gridjepa/README.md` | https://wynch.github.io/ml-course/wiki/source-local-support-learning.html |
| diffusion | shaky | private: `../ml-lab/gridjepa/README.md` | https://wynch.github.io/ml-course/wiki/concept-diffusion-language-models.html |
| interpretability | shaky | https://wynch.github.io/ml-course/explorables/05-transformer-anatomy.html | https://wynch.github.io/ml-course/wiki/concept-mechanistic-interpretability.html |
| calibration | shaky | https://wynch.github.io/ml-course/reader/#/05a-data-evaluation (section `#calibration-does-08-mean-eight-out-of-ten`) | none |
| hallucination | shaky | none | none |
| self-certification | shaky | private: `../ml-lab/gridjepa/runs/20260721_134506_phase1/report.md` | none |
| pre-registration | shaky | none | none |
| falsifiable | shaky | none | none |
| baseline | shaky | none | none |
