#!/usr/bin/env python3
"""Write ../source/palace.json from the tables below, then validate it.

These tables are the source of the palace: rename a room or change an object here,
then run this script and tools/build_page.py.

    python3 course/palace/tools/build_palace.py          # write source/palace.json
    python3 course/palace/tools/build_palace.py --check  # write nothing; exit 1 on an error or a stale file

Grid: metres, x east, y north, origin at the south-west corner of the house.
Levels: -2 foundations, -1 cellar, 0 ground (and garden, yard), 1, 2; the tower
stacks its own levels 0..4 on its own footprint. One storey is 4 m.
"""
import json, re, sys
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "source" / "palace.json"
COURSE = HERE.parents[1]

EXPL = {p.stem.split("-")[0]: p.name for p in (COURSE / "explorables").glob("*.html")
        if p.stem not in ("index", "05b-architecture-explorer")}
SLUG = {p.name.split("-")[0]: p.name for p in (COURSE / "modules").iterdir() if p.is_dir()}

def links(m):
    return {"module": SLUG[m], "explorable": EXPL[m], "quiz": m}

FLOORS = [
    {"id": "garden", "level": 0, "name": "The old garden",
     "reason": "Before the house: the classic methods. It lies beyond the output balcony, so the balcony looks back at where the walk began; the first learning machine stands farthest out."},
    {"id": "cellar", "level": -1, "name": "The cellar",
     "reason": "Weights being learned. Entered from the garden under the balcony, where the loss is computed; walked west, the way the gradient flows, so it ends under the gate, the last thing the gradient reaches."},
    {"id": "ground", "level": 0, "name": "The ground floor",
     "reason": "Weights used: one forward pass. The south row reads text in, west to east; the north row is the way back, where each written token returns to the gate."},
    {"id": "foundations", "level": -2, "name": "The footings",
     "reason": "How big to build: loss against size, data and compute. Below the cellar because every floor stands on it."},
    {"id": "first", "level": 1, "name": "The first floor",
     "reason": "Weights changed after training: fine-tuning, what that forgets, choices, and making the model smaller."},
    {"id": "second", "level": 2, "name": "The second floor",
     "reason": "Weights frozen: only the words going in or coming out change. Examples, steps, instructions, fetched pages, tools."},
    {"id": "yard", "level": 0, "name": "The yard",
     "reason": "Outside the model: a score from the world decides, a reward or a program that checks."},
    {"id": "tower", "level": 0, "name": "The tower",
     "reason": "Weights read, never changed: looking inside. The higher you climb, the harder the question; the top asks whether any of it is real."},
]

# Halls are not lessons: corridors, stairs and paths the walk may cross more than once.
HALLS = [
    ("garden_path", "garden", 0, [68, -4, 106, 2], "Garden path; its west end drops by steps into the cellar."),
    ("loss_pit", "cellar", -1, [60, 0, 68, 24], "Foot of the garden steps, under the balcony: where the loss is computed and the backward pass begins."),
    ("gradient_corridor", "cellar", -1, [12, 10, 60, 14], "The gradient channel: gold water running west, under the residual stream."),
    ("core", "ground", 0, [0, 10, 12, 24], "Stair core beside the gate, from the footings to the second floor."),
    ("residual_corridor", "ground", 0, [12, 10, 60, 14], "The residual stream: a channel of blue water from the gate to the balcony; every ground-floor room pours into it."),
    ("balcony", "ground", 0, [60, 0, 68, 24], "The output balcony, looking east over the old garden."),
    ("gallery1", "first", 1, [12, 10, 60, 14], "First-floor gallery."),
    ("gallery2", "second", 2, [12, 10, 60, 14], "Second-floor gallery."),
    ("outside_stair", "second", 2, [56, -4, 68, 0], "Outside stair from the workshop of hands down to the yard path."),
    ("yard_path", "yard", 0, [-12, -4, 56, 0], "Yard path along the south wall, walked west."),
    ("tower_stair", "tower", 0, [-12, -16, -10, -4], "Spiral stair inside the tower."),
]

# Rooms: module, name, floor, level, rect, colour family.
ROOMS = [
    ("00a", "The string-line bed", "garden", 0, [94, 2, 106, 20], "gold"),
    ("00b", "The four-tree orchard", "garden", 0, [82, 2, 94, 20], "gold"),
    ("00c", "The hollow pond", "garden", 0, [70, 2, 82, 20], "gold"),
    ("01", "The echo room", "cellar", -1, [50, 0, 60, 10], "gold"),
    ("02", "The mill", "cellar", -1, [44, 14, 60, 24], "gold"),
    ("05a", "The vault", "cellar", -1, [40, 0, 50, 10], "gold"),
    ("14", "The sunken slope", "cellar", -1, [26, 14, 44, 24], "gold"),
    ("16", "The blindfold room", "cellar", -1, [20, 0, 40, 10], "gold"),
    ("30", "The one-way room", "cellar", -1, [12, 14, 26, 24], "gold"),
    ("03", "The gate", "ground", 0, [0, 0, 12, 10], "ink"),
    ("04", "The atrium", "ground", 0, [12, 0, 30, 10], "ink"),
    ("08", "The picture door", "ground", 0, [30, 0, 42, 10], "ink"),
    ("05", "The balcony room", "ground", 0, [42, 0, 60, 10], "ink"),
    ("07", "The scriptorium", "ground", 0, [50, 14, 60, 24], "blue"),
    ("18", "The satchel room", "ground", 0, [41, 14, 50, 24], "blue"),
    ("27", "The slate room", "ground", 0, [32, 14, 41, 24], "blue"),
    ("09", "The fog room", "ground", 0, [22, 14, 32, 24], "blue"),
    ("33", "The blank-line room", "ground", 0, [12, 14, 22, 24], "blue"),
    ("19", "The footings", "foundations", -2, [12, 0, 36, 24], "gold"),
    ("06", "The tailor's room", "first", 1, [12, 0, 22, 10], "blue"),
    ("29", "The fading shelf", "first", 1, [22, 0, 31, 10], "blue"),
    ("17", "The tasting room", "first", 1, [31, 0, 41, 10], "blue"),
    ("23", "The press", "first", 1, [41, 0, 51, 10], "blue"),
    ("28", "The three-coin room", "first", 1, [51, 0, 60, 10], "blue"),
    ("20", "The example desk", "second", 2, [12, 0, 22, 10], "faint"),
    ("22", "The scratch room", "second", 2, [22, 0, 31, 10], "faint"),
    ("34", "The sign-painter's room", "second", 2, [31, 0, 41, 10], "faint"),
    ("21", "The library", "second", 2, [41, 0, 51, 10], "faint"),
    ("10", "The workshop of hands", "second", 2, [51, 0, 60, 10], "faint"),
    ("24", "The lever shed", "yard", 0, [46, -16, 56, -4], "ink"),
    ("26", "The marking hut", "yard", 0, [34, -16, 46, -4], "ink"),
    ("35", "The greenhouse", "yard", 0, [22, -16, 34, -4], "ink"),
    ("25", "The sand table", "yard", 0, [10, -16, 22, -4], "ink"),
    ("31", "The camera obscura", "yard", 0, [0, -16, 10, -4], "ink"),
    ("15", "The map room", "tower", 0, [-10, -16, 0, -4], "faint"),
    ("11", "The listening room", "tower", 1, [-10, -16, 0, -4], "faint"),
    ("12", "The dictionary room", "tower", 2, [-10, -16, 0, -4], "faint"),
    ("13", "The lens room", "tower", 3, [-10, -16, 0, -4], "faint"),
    ("32", "The lookout", "tower", 4, [-10, -16, 0, -4], "faint"),
]

DOORS = [
    ("00a", "garden_path"), ("00b", "garden_path"), ("00c", "garden_path"),
    ("00a", "00b"), ("00b", "00c"),
    ("garden_path", "loss_pit"),
    ("loss_pit", "gradient_corridor"), ("loss_pit", "01"),
    *[(r, "gradient_corridor") for r in ("01", "02", "05a", "14", "16", "30")],
    ("01", "05a"), ("05a", "16"), ("02", "14"), ("14", "30"),
    ("gradient_corridor", "core"), ("30", "core"),
    ("core", "03"), ("core", "residual_corridor"), ("core", "33"), ("core", "19"),
    ("core", "gallery1"), ("core", "gallery2"),
    *[(r, "residual_corridor") for r in ("04", "08", "05", "07", "18", "27", "09", "33")],
    ("03", "04"), ("04", "08"), ("08", "05"),
    ("residual_corridor", "balcony"), ("05", "balcony"), ("balcony", "07"),
    ("07", "18"), ("18", "27"), ("27", "09"), ("09", "33"),
    ("08", "yard_path"),
    *[(r, "gallery1") for r in ("06", "29", "17", "23", "28")],
    ("06", "29"), ("29", "17"), ("17", "23"), ("23", "28"),
    *[(r, "gallery2") for r in ("20", "22", "34", "21", "10")],
    ("20", "22"), ("22", "34"), ("34", "21"), ("21", "10"),
    ("10", "outside_stair"), ("outside_stair", "yard_path"),
    *[(r, "yard_path") for r in ("24", "26", "35", "25", "31")],
    ("31", "15"), ("yard_path", "tower_stair"),
    *[(r, "tower_stair") for r in ("15", "11", "12", "13", "32")],
]

ROUTE = [
    ("00a", "Start at the far end of the old garden, at a line strung between two stakes: the first machine that learned fixed a line each time it was wrong."),
    ("00b", "A line needs training; next door, four tricks need little or none: count with Bayes, ask the neighbours, find the main axis, sum up with a few centres."),
    ("00c", "The string line stopped at the first cut that worked. The pond finds the widest one, lifts points so a flat cut works, keeps memories as valleys and shows the curve that dips twice. Its outflow is attention: follow it to the house, and go down the garden steps first."),
    ("01", "Garden steps drop under the balcony, where the loss is computed. Learning starts there: the error is sent back through the graph, one local slope at a time."),
    ("02", "Slopes alone train nothing: the mill turns them into a loop of batches, steps and measures, and shows the loss on unseen data turning up."),
    ("05a", "Unseen data has to stay unseen: beside the mill sits the vault, where splits, metrics, thresholds and leaks are kept honest."),
    ("14", "With a fair score, watch how the optimiser moves: the floor sinks into a slope where step size, momentum and late understanding play out."),
    ("16", "All of that used labels. In the blindfold room the data supplies its own: hide part, predict it, and stop the cheat where everything maps to one point."),
    ("30", "Last room on the gradient's path: what if no gradient comes back at all? Learning with forward passes and local goals."),
    ("03", "The stair rises beside the gate, the last place the gradient reaches. Now walk forwards: text arrives and is chopped into tokens."),
    ("04", "Tokens become vectors and pour into the residual stream; in the atrium each one asks the others what to read."),
    ("08", "Pictures skip the gate's merges: cut into patches at the south door, they walk into the same block, and nobody wears blinkers."),
    ("05", "At the end of the row stands a real model, many atria deep; its last vector becomes odds over every token, read on the balcony."),
    ("07", "From the balcony, turn back along the north row: the scriptorium writes one token, keeps its keys and values, and sends it back to the gate."),
    ("18", "The scriptorium's shelves grow with every token; the satchel room carries a fixed-size state instead."),
    ("27", "A fixed state must be written well: the slate room writes each value by correcting what the slate already returns."),
    ("09", "Every room so far wrote one token after another. The fog room makes a whole picture at once, by removing noise step by step."),
    ("33", "Bring that idea back to text: start from a line of blanks and fill the surest first. The stair core is next door."),
    ("19", "You have seen the whole model. Before climbing, go down past the cellar to the footings: how big to build it for a budget."),
    ("06", "Up to the first floor, where weights change after training: the tailor fits a trained model with thin extra matrices."),
    ("29", "Every fitting costs something: on the fading shelf the old skill wears away while the new one is learned."),
    ("17", "Instead of showing good replies, let people choose between two: the tasting room learns from choices."),
    ("23", "A tuned model is still big: the press makes it smaller by teaching, cutting and rounding."),
    ("28", "Rounding after training costs; the three-coin room trains with three values from the start, and rounds the cache too."),
    ("20", "Up again: from here on no weight moves. At the example desk a frozen model learns a rule from examples in its prompt."),
    ("22", "Examples help; spending more tokens helps too: the scratch room thinks out loud and takes a vote."),
    ("34", "If words change the answer, tune the words: the sign-painter repaints the instruction and keeps what scores."),
    ("21", "Some answers are not in the weights at all: the library fetches the right page and pastes it in."),
    ("10", "Give the model tools and a loop: the workshop of hands acts, reads the result and acts again. Its door opens on the outside stair."),
    ("24", "Down in the yard the world gives scores, not labels: pull levers, keep what pays, and do not move too fast."),
    ("26", "Put a language model in the yard and let a program mark its answers: reward from a checker."),
    ("35", "Keep the checker, freeze the model: the greenhouse breeds programs and keeps only what the checker passes."),
    ("25", "Instead of trying moves in the world, try them in a model of it: the sand table plans before acting."),
    ("31", "Build that model from pictures alone: the camera obscura plans inside what it learned from frames. Its back wall is the tower."),
    ("15", "Climb to look inside. At the tower's foot, the map room shows the shape a trained network gives its input."),
    ("11", "One floor up, listen to single parts of a running model: hooks, probes, swaps and neurons."),
    ("12", "Single neurons mix many things; the dictionary room rewrites each state as a few directions from a long list."),
    ("13", "Point the same tools at vision models and a JEPA, and ask what one unit or one probe can prove."),
    ("32", "At the top, test every finding against a random copy, a second seed and a known truth. From the lookout the whole palace lies below: walk it again in your head."),
]

def F(name):  # an existing Moves formation
    return {"moves_formation": name}

def NEW(spec):
    return {"new": spec}

# Objects per room: (idea, image, formation, move ids). Ideas follow each README's Goals and section heads.
OBJ = {
"00a": [
 ("Rosenblatt's rule: on each mistake, add the point to the weights or subtract it, by its label; a right answer changes nothing.",
  "A garden rake on a string line lunges at every plant on the wrong side and drags the string toward it, then rests.", F("clouds('learn')"), ["a5"]),
 ("If some line separates the data with margin γ, the mistakes stop within (R/γ)².",
  "A tally post with only so many notches: the wider the gap between the two dashed strings, the fewer the notches, and the rake must stop when they run out.", F("clouds('margin')"), ["a6"]),
 ("No line splits XOR; one hidden layer that bends the plane lets a line work again.",
  "Four beds in a checker of blue and gold; the string spins forever until the bed folds like a cloth and the string lies flat between.", F("xor('spin') then xor('folded')"), ["a7", "a8", "a9", "a10"]),
 ("Least squares is a projection: the fit lies in the column space and the residual stands at a right angle to it; by gradient descent it explodes past a learning rate you can predict.",
  "A sundial: the shadow is the fit, the upright pin the residual. A ball hopping toward the shadow's tip lands on it; make each hop too long and it bounds off the dial.",
  NEW("Points on a tilted plane in 3-D; one target point above drops a plumb line to its foot, with a right-angle tick; then a ball steps toward the foot and overshoots when the rate passes the bound."), []),
 ("Ridge: add a penalty on the size of the weights, (XᵀX + λI)w = Xᵀy, and two near-copy features stop cancelling in huge opposite weights.",
  "Two stakes pulling the string in opposite directions with all their might, until a spring between them makes both relax.", F("polyfit('penalty')"), ["n5"]),
],
"00b": [
 ("Bayes as odds: posterior odds are prior odds times the likelihood ratio, so a 96%-sensitive test can leave you 95% likely to be healthy.",
  "A hive with one queen among thousands of bees; a sieve catches the queen and a crowd of workers, and the workers win.", F("rare()"), ["n16"]),
 ("Naive Bayes assumes the features are independent given the class: closed form, cheap, and wrong in a known way.",
  "Garden gnomes who each vote on the flower alone, never looking at one another.",
  NEW("Two tilted clouds; each gets an upright, axis-aligned ellipse that ignores the tilt; the straight cut they give cannot follow the tilt of the clouds."), []),
 ("k-NN labels a point by its nearest neighbours; 1-NN error lies between the Bayes error and twice it (Cover-Hart).",
  "A newcomer on a bench turns to the five nearest trees and takes their vote; two fences beside the bench mark the Bayes error and twice it.", F("knn()"), ["a11"]),
 ("PCA by power iteration: multiply any arrow by the covariance again and again and it turns to the main axis; deflate to find the next.",
  "A weather vane that, each time the wind blows, swings closer to the orchard's long row.", F("pca()"), ["a12"]),
 ("k-means sums a cloud up with a few centres: assign each point, move each centre to its points' mean, repeat; a moving average over small batches can shake a centre out of a bad spot, if the batches are noisy enough.",
  "Three scarecrows that each walk to the middle of the birds they scare, until none moves.",
  NEW("Three blobs of points; three gold centres step toward their cluster means each round, with the dividing lines redrawn; one run uses a jittering moving average."), []),
],
"00c": [
 ("The hard-margin SVM cuts the widest gap; only the support vectors get a vote, read off α.",
  "A plank laid across the pond as wide as it will go, held up by three stones that touch its edges; the rest of the stones could vanish.", F("clouds('margin')"), []),
 ("The kernel trick: lift the points so a flat cut works, and compute the inner product without ever doing the lift.",
  "Two rings of lily pads; a swell lifts the outer ring high and the inner one hardly at all, and a flat sheet of glass slides between them.",
  NEW("Concentric rings in a plane; each point rises in z by x²+y² so the outer ring lifts above the inner; a horizontal plane of faint points cuts between the rings."), []),
 ("A Hopfield network stores patterns as valleys of an energy; updates only go downhill and repair a corrupted input; it holds about 0.138·N patterns.",
  "A stone basin carved with hollows; a chipped marble dropped in rolls into the nearest hollow and comes out whole.", F("basins()"), []),
 ("The modern Hopfield update Xᵀsoftmax(βXξ) lines up term by term with attention, softmax(QKᵀ/√d)V.",
  "A torch from the atrium stands by the basin: shine it on the hollows and the best match flashes back, the same trick as the mirrors indoors.", F("rays('blend')"), ["a97"]),
 ("Double descent: test error peaks where parameters equal data points, p = n, then falls again as the fitter picks the smallest-norm fit; here it never gets back below the first dip.",
  "A footbridge dips, climbs to a sharp hump at the pond's neck, then comes down again, but lands higher than its first dip.", F("dd('reveal')"), ["n1", "n6", "n7", "n8", "n9"]),
],
"01": [
 ("Every operation you apply records a node; the forward pass builds a graph.",
  "Every word you say in the echo room is carved as a stone on the floor, joined by string to the words it came from.", F("graph('fwd')"), ["a18"]),
 ("Reverse mode is message passing: each node multiplies the slope it receives by its own local slope and passes it back.",
  "Shout at the last stone and the echo runs back along the strings, louder or softer at each stone.", F("graph('back')"), ["a19"]),
 ("Run backward in reverse topological order, so each node has all its gradient before it passes any on.",
  "Numbered stepping stones; the echo may only leave a stone once every string into it has rung.", F("graph('all')"), ["a20"]),
 ("Gradient checking: compare each slope with a finite difference to catch a wrong derivative.",
  "A pair of scales by the door weighs every slope against a nudge of the input; a wrong one tips over.",
  NEW("Pairs of points, one per parameter: the analytic slope and the finite-difference slope sit together; one pair splits apart and turns gold."), []),
 ("With this engine a small MLP walks downhill on its loss to 100% on two moons' 100 training points.",
  "A ball rolls down the floor of the echo room into a bowl, step by step against the slope.", F("bowl('roll')"), ["a13", "a14"]),
],
"02": [
 ("A network is weighted sums, each followed by a bend; derive each layer's forward and backward by hand.",
  "The mill's stacked grindstones, each one tilted, each one followed by a hinge.", F("layers()"), ["a21"]),
 ("Fused softmax and cross-entropy: the gradient on the scores is simply p minus the one-hot answer.",
  "A row of sacks on a scale; the right sack is pulled down by exactly how short it fell.",
  NEW("Bars of a softmax with the right class marked gold; arrows on each bar show p − y, down on the true class, up on the rest."), []),
 ("The training loop: split, batch, forward, backward, step, measure, every epoch.",
  "A waterwheel with six buckets, each painted with one step, turning round and round.",
  NEW("Six stations on a ring; points ride the ring in batches, pausing at each station; a counter of epochs ticks on each turn."), []),
 ("It overfits: training loss keeps falling while held-out loss turns up; stop at its lowest point.",
  "Two water marks on the mill wall: one keeps sinking, the other bottoms out and starts to rise.", F("losses('reveal') then losses('stop')"), ["n2", "n3", "n4"]),
],
"05a": [
 ("Make deterministic, disjoint train, validation and test splits, by whatever unit could leak; open the test set once.",
  "Three locked chests; the test chest has one key, which breaks in the lock after one use.", F("split() then vault()"), ["n10", "n11", "n18"]),
 ("Leakage is the fastest route to a fake win: duplicates across splits inflate the score.",
  "A drip from the training chest into the test chest, and a trophy that grows as the puddle grows.", F("leak()"), ["n12"]),
 ("The confusion matrix is the primitive: accuracy, precision, recall, specificity and F1 all come from its four counts.",
  "Four drawers in the vault, labelled by the two ways to be right and the two ways to be wrong.", F("confusion()"), ["n13"]),
 ("A threshold is a product decision: moving it trades one kind of error for the other.",
  "A sliding bar across the vault door; push it and coins fall from one drawer into the other.", F("thresh()"), ["n14"]),
 ("Calibration: does 0.8 sure mean right eight times in ten? Measure it, do not assume it.",
  "A shelf of jars labelled 10% to 90%, each filled to how often it was right; the honest jars match their labels.", F("calib()"), ["n15"]),
 ("One aggregate can hide a slice that got worse; compare slices, with error bars.",
  "A good average on the front of the vault door, and behind it one small drawer marked low-light that is half empty.", F("boot()"), ["n17"]),
],
"14": [
 ("SGD, momentum and Adam as update rules: momentum keeps speed through flats, Adam scales each step per weight.",
  "Three balls race down the sunken floor: one plods, one swings with momentum, one adjusts its stride on every tile.", F("bowl('momentum')"), ["a16"]),
 ("The learning rate decides a lot, and the best one depends on the optimiser.",
  "A ball thrown too hard bounces from wall to wall of the slope and never settles.", F("bowl('bounce')"), ["a15"]),
 ("Linear warmup then cosine decay: start gently, then let the rate fall smoothly to zero.",
  "A dimmer switch on the wall that slowly brightens, then fades along a smooth curve.",
  NEW("A learning-rate curve that rises in a straight line then falls as half a cosine; a gold point rides it while the ball's steps lengthen and shorten."), []),
 ("Slice the loss along filter-normalised directions; the top Hessian eigenvalue measures how sharp the bottom is.",
  "A surveyor cuts a slice through the floor and holds it up: a sharp V or a wide U.", F("basins()"), ["a17"]),
 ("Grokking: the network fits its training set long before it finds the rule; weight decay made the late jump happen in this run.",
  "A clock on the far wall stays wrong for hours, then suddenly shows the right time; a small weight on its pendulum is what does it.", F("curves(GROK) then decay()"), ["a59", "a60", "a61"]),
],
"16": [
 ("Self-supervised learning: the data supplies its own targets; hide part and predict it back.",
  "Blindfolded statues whose hands try to trace the half of the picture that is covered.", F("spiral('mask')"), ["a41", "a42"]),
 ("SimCLR: two crops of one image should land close together, and crops of others far away.",
  "Two halves of one torn photo pulled together across the room by a spring.", F("views()"), ["a47"]),
 ("A JEPA predicts the latent of the hidden block, not its pixels.",
  "A statue points at a sealed box and names what it holds, without ever drawing it.",
  NEW("A context cloud and a separate target cloud; a predictor arrow lands on the target cloud's centre in latent space, never on its pixels."), []),
 ("Collapse: map every input to one point and the JEPA loss is perfect and worthless.",
  "All the statues turn and point at the same nail in the wall.", F("collapse()"), ["a48"]),
 ("Block collapse with stop-gradient, an EMA target, a variance term or SIGReg; measure it by spread and effective rank.",
  "A sheepdog that runs round the points and keeps them spread in a round flock.", F("spread()"), ["a49"]),
],
"30": [
 ("Backprop keeps every layer's activations for the backward pass, so its memory grows with depth and batch.",
  "A porter who must carry a copy of every room he passed through, his pile growing at each door.",
  NEW("A stack of layers; each leaves a pile of points behind as the forward pass goes up; a gold bar of memory grows with depth; checkpointing keeps only some piles and recomputes the rest."), []),
 ("MeZO: nudge every weight along one random direction both ways; two forward passes give the slope along it.",
  "A blindfolded walker taps the floor one step left and one step right and leans toward the lower tap.", F("mezo('probe') then mezo('step')"), ["a64", "a65"]),
 ("Forward-forward and Mono-Forward: each layer learns from a goal of its own, with no error sent down.",
  "Each layer of the room has its own little target pinned on its wall and its own dart.",
  NEW("Separate layer rings, each with its own gold target; points in each ring move toward their target with no arrows between rings."), []),
 ("Predictive coding: layers trade prediction errors; the error fades with depth.",
  "Whispers passed down a line of people get quieter at every ear.",
  NEW("Layer columns with an error bar beside each; the bars shrink from output to input over the inference steps."), []),
 ("Read a memory claim with its meter: the memory a rule saves is not the memory its optimiser costs.",
  "Two gauges by the door, one for tensors alive, one for the whole process, often disagreeing.",
  NEW("Two vertical bar meters per rule side by side, tensors alive against process memory."), []),
],
"03": [
 ("Text must become numbers: start from single bytes or characters.",
  "At the gate, a sentence is pulled apart into loose letters that rattle into a tray.", F("tokens(0,0)"), ["a24"]),
 ("BPE merges the commonest adjacent pair into one symbol, again and again.",
  "A gatekeeper who glues the two letters he sees side by side most often into one tile.", F("tokens(0,1)"), ["a25"]),
 ("The list of merges is the whole tokenizer; frequent pieces become tokens.",
  "The gatekeeper's ledger of glued pairs, which is all he needs to chop any new text.", F("tokens(1,3)"), ["a26"]),
 ("Byte-level BPE round-trips exactly: decode(encode(text)) gives back the text.",
  "Tiles run through the gate and back out, and the sentence that leaves is letter for letter the one that came in.",
  NEW("Token chips merge into longer chips, then unmerge back to the same characters in the same places."), []),
 ("Tokenizers are unfair across languages: fertility, the tokens spent per word, differs.",
  "A toll booth that takes one coin a word from some travellers and two from others, for the same message.",
  NEW("Vertical bars of tokens per word for several languages, from the module's fertility results."), []),
],
"04": [
 ("Each token sends out a query; every token offers a key; the better a key lines up, the larger its weight.",
  "In the atrium each guest shines a torch; the others hold up mirrors, and the mirrors that face the torch flash brightest.", F("rays('query') then rays('rotate')"), ["a32", "a33"]),
 ("Attention is a soft dictionary lookup: the answer is the values weighted by those matches; the scores are divided by √d_head first, so the softmax does not saturate.",
  "The guest collects a cup from each mirror-holder and pours them together, more from the brightest.", F("rays('blend')"), ["a34"]),
 ("The causal mask: each token reads only what came before it.",
  "Every seat in the atrium has blinkers so no guest can look at anyone who came in later.", F("matrix('full') then matrix('causal')"), ["a35", "a36"]),
 ("Multi-head attention: each head learns its own habit, such as one step back.",
  "Several torches per guest, each with a coloured lens, each looking for a different kind of neighbour.", F("matrix('head')"), ["a37"]),
 ("The block: attention, then a small MLP, each adding into the residual stream with pre-norm, repeated.",
  "The atrium repeats along the corridor like locks on a canal; at each, the guests and then a clerk tip a cupful into the blue channel, and it flows on.", F("stack()"), ["a38"]),
],
"08": [
 ("A convolution slides a small kernel of weights over the image: locality and weight sharing.",
  "A stamp the size of a postage stamp walks across a mural, inking a weighted sum at each spot.",
  NEW("A 20 × 21 point grid (the grid formation) with a 3 × 3 gold window sliding across it; the output grid fills point by point."), []),
 ("A small CNN invents its own edge and blob detectors with no one telling it to.",
  "The stamps on the door's shelf turn out, when you look, to be little edges and dots no one carved.",
  NEW("Sixteen small 3 × 3 point patches that sharpen from noise into edge and blob patterns as training runs."), []),
 ("A ViT cuts the image into patches and reads them as tokens with the same block, without the causal mask.",
  "The picture is cut into square tiles, and the tiles walk in single file through the door into the atrium, where no one wears blinkers.", F("patches()"), ["a40"]),
 ("Attention maps show where the model looks: last-layer [CLS] attention and attention rollout.",
  "A lantern behind the picture lights up the parts the guests are staring at.",
  NEW("The image grid with each point's size and colour set by its [CLS] attention weight."), []),
 ("CLIP matches images to text prompts with no task training; pretraining scale trades against task fine-tuning.",
  "Captions hang on hooks by the door; each picture walks to the caption nearest in meaning.",
  NEW("One image point and several caption points on a sphere; the nearest caption lights gold."), []),
],
"05": [
 ("Account for where every parameter lives in a real model: embeddings, attention and MLP, layer by layer.",
  "Thirty-two atria in a row; in each the torch-bearers fill one bench and the clerks of the small network fill three.",
  NEW("420 points shared out among boxes for the embedding table, then per-layer attention and MLP boxes in proportion to their parameter counts."), []),
 ("Tokens become points in a learned space; tokens used alike sit close: July beside June and August, before any layer runs.",
  "On the gallery floor, word-tiles cluster: months together, countries together, code together.", F("embed('gather') then embed('cluster')"), ["a27", "a28"]),
 ("The model scores every token it knows; a softmax turns the scores into odds.",
  "From the balcony a row of flags rises over the garden, one per token, as tall as its odds.", F("bars('prob')"), ["a30"]),
 ("Temperature, top-k, top-p and min-p reshape the odds before you sample.",
  "A wind on the balcony: cold, and only the tallest flag stands; warm, and the flags draw level.", F("bars('temp')"), ["a31"]),
 ("The logit lens reads the residual stream at every layer through the output head; the answer settles at a decision layer.",
  "Windows along the gallery each show a flag; from one window on, they all show the same winner.",
  NEW("A column of layer slices; at each slice a small bar chart of next-token odds; from one layer up the top bar stays gold."), []),
 ("A chat template wraps each turn in special tokens the model was trained on.",
  "Each speaker's words go in an envelope with a coloured seal at each end; the model learned to read from sealed letters.", F("tokline()"), []),
],
"07": [
 ("Load a transformer from raw bytes: a manifest says where each tensor sits in one flat blob.",
  "A scribe unrolls one long scroll of numbers and snips it into labelled pieces.",
  NEW("A long ribbon of points that splits into named blocks: embeddings, then per-layer QKV, MLP and norms."), []),
 ("The KV cache keeps each token's key and value, so decoding stays linear instead of quadratic.",
  "Shelves along the scriptorium wall where each written token leaves its key and value, never written twice.", F("matrix('cache')"), ["a39"]),
 ("Prefill pushes the prompt through to fill the cache; decode adds one token at a time, and there the mask has nothing to hide.",
  "The scribe first copies the page he was given, to stock his shelves, then writes on one letter at a time.",
  NEW("An attention matrix whose lower triangle fills in one flash (prefill), then gains one row per step (decode) with no masked cells in the new row."), []),
 ("Sample by hand: temperature, top-k and top-p from scratch.",
  "A bag of letter tiles in proportion to the odds; the scribe trims the bag before drawing.", F("bars('temp')"), []),
 ("Quantise weights to int8 and weigh size against quality; GGUF's Q8_0 is the same idea, Q4_K a harsher one.",
  "Pens that write with only 255 shades of ink, and a ledger of what was lost.", F("quant('hist')"), []),
 ("A parity test: the hand-written engine must match PyTorch to the tolerance.",
  "Two scribes write the same page; laid one on the other, every letter lines up.",
  NEW("Two point clouds that overlay exactly; any point that differs beyond the tolerance turns gold."), []),
],
"18": [
 ("A GRU's update gate mixes the old state with a new candidate, token by token.",
  "A satchel with a flap that opens a little or a lot, mixing old contents with the new.",
  NEW("One fixed cloud of points (the state); at each token a gate value blends it toward a new candidate cloud."), []),
 ("A recurrent model carries one fixed-size state instead of keeping every past token.",
  "Next door the shelves grow; here a single satchel of fixed size is all anyone carries.", F("state()"), ["a66"]),
 ("A diagonal linear recurrence is a convolution, so it trains in parallel; check it against one token at a time.",
  "A row of fading footprints that can be stamped all at once or one step at a time, with the same result.",
  NEW("A sequence of points whose state is a decayed sum of the past; drawn once as a sliding kernel and once step by step, overlapping."), []),
 ("Selective scan: when the decay depends on the input, the convolution is gone and a scan is needed.",
  "Each new footprint now decides how fast the old ones fade, so no single stamp fits the row and you must walk it step by step.",
  NEW("The same sequence with a per-token decay that varies; the kernel picture breaks into a chain of steps."), []),
 ("Count and measure what each model keeps between tokens: the GPT's cache grows, the recurrence stays flat.",
  "A rising pile of books next to a satchel that never gets heavier.",
  NEW("Two curves of memory per generated token against context length: a rising line and a flat one."), []),
],
"27": [
 ("The state is a matrix S that stores values under keys: S k returns the value.",
  "A slate grid on the wall; touch a key at the head of a column and its value lights up down that column.",
  NEW("A square grid of points (as in wgrid); a key column and the returned value row light blue."), []),
 ("Linear attention adds each pair into S; decay fades old pairs; both leave interference as S fills.",
  "Chalk written over chalk until the slate is grey; a damp cloth fades the oldest writing.",
  NEW("Pairs written into the grid as outer products; the grid gets noisier; a decay factor dims older writes."), []),
 ("The delta rule writes by correcting: read what S returns for k and write only the gap to v; that is one step of gradient descent.",
  "Before writing, the clerk reads what the slate says under that key and chalks in only what is missing or wrong.", F("delta()"), ["a67"]),
 ("The gated delta rule adds decay to the delta rule; compute it a chunk at a time with one triangular solve.",
  "A clerk who chalks in only the difference and also wipes the whole slate a little before each write.",
  NEW("The delta rule's grid with a global fade per step; the sequence split into chunks, each solved in one go."), []),
 ("Fill one 32 × 32 state past its size: plain adding keeps a little of every pair; the rules that correct or fade keep the newest and lose the oldest.",
  "One slate with more and more names chalked on; four clerks read them back, and the ones who rub out as they write lose the oldest names first.",
  NEW("Recall curves against the number of stored pairs for each write rule, from the module's capacity results."), []),
],
"09": [
 ("The forward process adds Gaussian noise on a schedule, and has a closed form that jumps to any step.",
  "A fog machine in the corner turns a spiral of lanterns into haze, one puff at a time.", F("diffuse('forward')"), ["a43"]),
 ("Train a network to predict the noise that was added; one squared error is the whole loss.",
  "A guide in the fog points at each lantern and says how far the fog has pushed it.", F("diffuse('noise')"), ["a44"]),
 ("Run it backwards: start from pure noise and subtract the predicted noise, step by step.",
  "The fog machine runs backwards, drawing the haze in puff by puff, until the spiral of lanterns stands clear.", F("diffuse('reverse')"), ["a45"]),
 ("Swap the scheduler, DDPM to DDIM, and trade steps for quality.",
  "The same clearing done in ten long strides instead of hundreds of small ones.",
  NEW("Two reverse paths for the same points: many short steps against few long ones, ending close together."), []),
 ("Latent diffusion, text conditioning and classifier-free guidance turn a denoiser into Stable Diffusion.",
  "Whisper 'cat' to the guide: he clears the fog once with the word and once without, then steps further the way the word pulled.",
  NEW("Reverse paths bent toward one labelled target cloud, more strongly as a guidance scale rises."), []),
],
"33": [
 ("Flow matching regresses a straight-line velocity from noise to data; one sampling step lands on the average.",
  "Strings stretched straight from each noise point to a spiral point; cut to a single step, every point lands in the middle.",
  NEW("Spiral data and noise points joined by straight lines; points move along them; a one-step sample collapses to the centre."), []),
 ("A masked diffusion model: drop the causal mask, hide a random share of characters, learn to guess them back.",
  "A line of blank tiles on a long table, filling in the surest letters first, in any order.", F("fill()"), ["a46"]),
 ("Sample a line in any number of steps; the reveal rule decides which blanks fill first.",
  "A dial on the table from 1 to 128; turned low, many letters appear at once and the line gets worse.",
  NEW("The fill formation run at 1, 8 and 128 steps side by side."), []),
 ("Block diffusion writes block by block left to right, denoising inside each block, with a KV cache over finished blocks.",
  "The table is cut into short boards; each board fills by unmasking, then slides onto the scriptorium's shelf.",
  NEW("Blocks of tiles left to right; within the current block tiles unmask; finished blocks drop into a cache row."), []),
 ("A data-repeat curve: on text seen 200 times the diffusion curve crosses under its overfitting twin, but the twin stopped at its best epoch still leads; on fresh data the twin leads throughout.",
  "Two runners on a looping track: the first peaks at lap 64 and tires; the second starts slowly and passes him at lap 200, but never beats his best lap.",
  NEW("Two loss curves against epochs of repeated data; they cross."), []),
],
"19": [
 ("Train one recipe at six sizes, from 51k to 3.2M parameters, and watch each held-out loss settle.",
  "Six footings poured side by side, each wider than the last, each with its own water mark.", F("power('curves')"), ["a68"]),
 ("Loss falls as a power of size: a straight line on log axes; check the fitter on known data first.",
  "A plumb line stretched across the footings that touches every water mark.", F("power('line')"), ["a69"]),
 ("Count N by hand and compute C = 6ND for training.",
  "A brick counter: six times the bricks times the loads.",
  NEW("A block of points N wide and D deep, stacked six layers high, its count shown as C."), []),
 ("Each compute budget has a best size; a toy grid can fake that optimum.",
  "A rope draped over the footings: the lowest point under each budget picks the size.", F("power('envelope')"), ["a70"]),
 ("Kaplan and Chinchilla disagreed on how to split compute between size and data, and why.",
  "Two architects arguing over one plan, one adding floors, the other adding bricks to each floor.",
  NEW("Two lines of optimal N against C with different slopes, crossing the module's toy points."), []),
],
"06": [
 ("Supervised fine-tuning: next-token loss on curated replies, read with care.",
  "The tailor's tape measures only the sleeves, the replies, not the whole coat.", F("vbars(LOSSB)"), ["n23"]),
 ("Full fine-tuning moves every weight.",
  "A whole coat unpicked and resewn, every stitch moved.", F("wgrid()"), ["n19"]),
 ("Most of the change lives in a few directions: the update has low intrinsic rank.",
  "Lay the changes side by side and they all point along a few seams.", F("lowrank()"), ["n20"]),
 ("LoRA trains two thin matrices beside a frozen one, under 3% of the weights.",
  "Two thin strips of cloth pinned beside the coat; the coat itself stays untouched.", F("lora('train')"), ["n21"]),
 ("LoRA's B starts at zero, so training starts from the unchanged model.",
  "The new strips start the same colour as the coat, so on day one nobody sees a change.", F("lora('zero')"), ["n22"]),
],
"29": [
 ("Forgetting is the rise of a held-out loss on the old task while learning a new one.",
  "Old books on a shelf whose ink fades as new books are pressed in beside them.", F("forget('full')"), ["a62"]),
 ("Measure how far the model moved as KL from the start model, and say on which inputs.",
  "A tape measure from each book's old spot to where it now stands.",
  NEW("Points drift from their start positions; a faint line joins each to its start; the mean length is shown."), []),
 ("Replay and self-distillation keep most of the old skill; LoRA learns less and forgets less.",
  "Every few new books, someone reads aloud from an old one and its ink darkens again.", F("forget('replay')"), ["a63"]),
 ("Compare methods at a matched new-task score, not only at the end; one number predicts forgetting only sometimes.",
  "Racers lined up at the same finishing post, so you compare what they dropped on the way.",
  NEW("A scatter of runs: forgetting against new-task gain, with a vertical line at a matched gain."), []),
 ("Loss of plasticity: plain backprop can lose the ability to learn; continual backprop resets dead units.",
  "Some shelves go dusty and stop taking books; a caretaker swaps a few out for new ones.",
  NEW("A grid of units that dim and go dark over tasks; a few are relit at each step."), []),
],
"17": [
 ("Bradley-Terry: scores from pairwise choices; the chance one wins is a sigmoid of the score gap.",
  "Two glasses, a taster points at one; each glass's score shifts by the surprise.", F("prefs()"), ["n26"]),
 ("Train a linear reward head on a frozen model and test it on held-out pairs.",
  "A scorecard clipped to a closed cask: the cask is never opened, only the card is learned.",
  NEW("A frozen cloud of reply points with a single learned direction; held-out pairs scored along it."), []),
 ("DPO comes from Bradley-Terry plus a KL penalty: raise the chosen reply, lower the rejected, stay near the reference.",
  "Two glasses on a balance, tied by a short leash to the old balance.",
  NEW("Pairs of points; chosen moves up, rejected down, both tethered to a faint reference copy by springs."), []),
 ("Length can pass for quality: check for it in the data, the reward head and DPO.",
  "The taster keeps choosing the taller glass, whatever is in it.", F("vbars(LENB)"), ["n27"]),
],
"23": [
 ("Distillation: a half-size student trained on the teacher's outputs.",
  "A big press hands its print to a small press, which learns to copy it.", F("distill('teach')"), ["n24"]),
 ("Soft targets say more than hard labels; the temperature sets how much.",
  "The big press shows the faint second guesses too, not just the boldest letter.", F("distill('soft')"), ["n25"]),
 ("Magnitude pruning: remove the smallest weights; a random cut of the same size breaks the model.",
  "Shears that clip the thinnest twigs first; clipping at random kills the tree.", F("quant('prune')"), ["a83"]),
 ("Quantise by hand: int8 per row, 4 bits per group of 64; count the bytes.",
  "The press squeezes a fat book into a thin one: 723.6 MB down to 203.6 MB.", F("quant('hist') then quant('q16')"), ["a80", "a81"]),
],
"28": [
 ("BitLinear: weights in {−1, 0, 1} during training, 8-bit inputs, a straight-through gradient.",
  "A room where every coin is minus one, zero or one, and the shopkeeper learns to price with them.", F("quant('q3')"), ["a82"]),
 ("Compare models at the same file size, not the same parameter count, with every arm tuned.",
  "Three purses that weigh the same; one holds many three-value coins, two hold fewer, finer ones.",
  NEW("Vertical bars of held-out loss for ternary and float models rounded to 4 and 2 bits, all at one file size."), []),
 ("A bf16 cast at export can change a ternary model's codes.",
  "A coin passed through the mint's press comes out a slightly different coin.",
  NEW("Ternary points on three levels; after a cast a few slip off their level and turn gold."), []),
 ("Quantise the KV cache per token or per channel; a Hadamard rotation spreads the outliers.",
  "A grate in the floor looks down on the scriptorium's shelves; the tall books are cut down and the shelf is spun so no one stands out.",
  NEW("A cache matrix with a few tall outlier columns; a rotation spreads them evenly before rounding to 8, 4 and 2 bits."), []),
],
"20": [
 ("In-context learning: show a frozen model a few examples of a new rule and it continues the pattern.",
  "A desk with a few worked examples pinned up; the clerk copies the rule onto the next card.", F("icl(1,4)"), ["a71"]),
 ("On in-context regression its guesses follow least squares on the examples in front of it.",
  "The clerk's pencil line lies exactly on a ruler laid through the pins.", F("icl(4,12)"), ["a72"]),
 ("More examples, lower error, as if a small learner ran inside one forward pass.",
  "Each pin added makes the line steadier.", F("icl(12,24)"), ["a73"]),
 ("Score a few-shot prompt on a real model from two next-token logits, with an interval.",
  "Two lamps, 'positive' and 'negative', brighten as examples go up on the board.",
  NEW("Two bars per review for the two label logits; accuracy with an interval against the number of shots."), []),
],
"22": [
 ("Chain of thought: let the model write its steps first; on 200 maths problems a small model went from 15% to 61.5%.",
  "A wall of slate where the model writes long sums before the answer.", F("chains('think')"), ["a74"]),
 ("Self-consistency: sample many chains and keep the most common answer; any-of-n right is the ceiling.",
  "Eight scribes write on eight slates; the answer most of them reached is kept.", F("chains('vote')"), ["a75"]),
 ("Best-of-n with the model as its own grader: its grades are worth less than they seem.",
  "A scribe marks his own slates and always gives himself top marks.",
  NEW("Eight chains with a self-given score above each; the chosen one is not always right."), []),
 ("Compare methods at equal tokens, not equal n.",
  "A coin box under each method; the winner is read at the same height of coins.",
  NEW("Accuracy against tokens spent for each method, with a vertical line at a fixed token budget."), []),
],
"34": [
 ("A prompt as a parameter: a frozen model, a fixed scorer, a training and a held-out split, one budget; APE and OPRO propose and keep.",
  "A sign-painter repaints the instruction board, scores each sign on sixty visitors, and keeps the best.", F("prompts('search')"), ["a76"]),
 ("The best of many prompts on a few questions is partly luck: only fresh questions tell.",
  "The winning sign is carried out to eight hundred strangers and scores lower there.", F("prompts('heldout')"), ["a77"]),
 ("A text 'gradient': a critique in words that says how to change the prompt; it shares a direction with a gradient but no size.",
  "A critic's note pinned to the sign: 'more about places', with no number on it.",
  NEW("A cloud of prompt points; a written-critique arrow of fixed length points toward better scores, with no magnitude."), []),
 ("A GEPA-style loop reflects on failures and mutates the best prompts, on the same budget.",
  "The painter reads the visitors' complaints and paints a new sign from the best two old ones.", F("evo('run')"), []),
],
"21": [
 ("Embed a text as a unit vector with a model trained for it and search by cosine; a language model's mean-pooled states search poorly.",
  "Books shelved by meaning, each with a compass needle; the needles of an untrained catalogue point anywhere.",
  NEW("Paragraph points on a sphere for two embedders: one clustered by topic, one smeared."), []),
 ("BM25 over an inverted index: idf, term-frequency saturation and length do the work.",
  "A card catalogue: each word has a drawer listing the books it appears in, rare words weighing most.", F("passages('rank')"), ["a78"]),
 ("Measure a retriever with recall@k against a known answer: here word matching put the right paragraph first 75.5% of the time, a neural search 62.5%.",
  "Two librarians race to the right shelf; the one who reads the cards wins.", F("passages('pull')"), ["a79"]),
 ("A RAG loop: retrieve, paste, answer, score with exact match and F1; split the error between retriever and reader.",
  "The librarian hands a page to the reader; a referee checks whether the page or the reader was wrong.",
  NEW("A two-stage funnel: retrieved passages flow into a reader point; failures coloured by stage."), []),
 ("An agent can keep memory in the prompt, in a store or in its weights, and each costs something.",
  "Three places to leave a note: on your hand, in a drawer, or tattooed.", F("passages('rank')"), []),
],
"10": [
 ("A model only writes text; an agent is a model, tools and a loop.",
  "A workshop of hands on the wall, each holding a tool, and a model that can only speak.", F("tokline() then agent('ring')"), ["n28", "n29"]),
 ("The ReAct loop: thought, action, observation, until a final answer ends it.",
  "A ticker tape that prints thought, action, observation, round and round, until it prints the answer and stops.", F("trace() then agent('exit')"), ["n30", "n31"]),
 ("Code as action: the model writes code that calls tools, instead of JSON tool calls.",
  "The hands take a written recipe rather than one shouted command.", F("agent('code')"), ["n32"]),
 ("Agents are stochastic: run the suite many times and read success rates, not one demo.",
  "Ten identical workbenches running the same job; the tally board shows how many finished.", F("vbars(AGB)"), ["n33"]),
 ("MCP: one protocol plugs many tools into the model.",
  "A single wall socket with every tool's plug fitting it.", F("hub()"), ["n34"]),
 ("Small-model failures, format drift, loops and over-acting, and tool output as untrusted input: check every step.",
  "A hand that keeps reaching for the same tool, and a note slipped in with a tool's result that tries to give orders.", F("agent('check') then agent('stuck')"), ["n35", "n36"]),
],
"24": [
 ("Bandits: no labels, only rewards; ε-greedy, UCB1 and Thompson sampling trade exploring for exploiting; regret measures the cost.",
  "A shed of four levers that pay out at unknown rates; a tally of pulls and wins chalked under each.", F("bandit(0,24) then bandit(24,420)"), ["a84", "a85"]),
 ("Tabular Q-learning: learn the value of each move in each cell; check it against breadth-first search.",
  "A grid on the shed floor where numbers spread out from the goal tile, cell by cell.",
  NEW("A 10 × 10 grid of points with walls; each cell's brightness is its learned value, spreading back from the goal."), []),
 ("REINFORCE: raise each action's log-probability in step with the return that followed.",
  "A cart balancing a pole; every move that kept it up gets a chalk mark that makes it likelier.", F("policy()"), ["a86"]),
 ("PPO clips each update so the new policy stays near the old; unclipped, training rose then collapsed.",
  "A governor on the cart's engine that stops any lurch.", F("curves(CLIP)"), ["a87"]),
],
"26": [
 ("A checker reward: sample a group of replies, mark each right or wrong, push up those above the group mean.",
  "A teacher's hut where eight answers come in at once and the right ones are stamped.", F("checker('grade')"), ["a90"]),
 ("GRPO, Dr. GRPO and RLOO differ in how they form the advantage; GRPO's division by length matters.",
  "Three stamps of different shapes for the same right answers.",
  NEW("One group of eight replies; three rows of advantage bars for the three rules, from the same rewards."), []),
 ("Read pass@k with the unbiased estimator: a crossover with the base model would mean training narrowed it.",
  "Two ladders of how often any of k tries is right, before and after; watch where they cross.",
  NEW("pass@k curves for base and trained model against k on log axes."), []),
 ("A random-reward control: with coin-flip grades accuracy fell from 59.1% to 4.6%.",
  "A teacher who stamps at random, and a class that gets steadily worse.", F("checker('noise')"), ["a91"]),
 ("KL to the base model measures how far training moved the policy.",
  "A rope from the hut back to the schoolhouse, paid out as the class changes.",
  NEW("Policy points drifting from a faint base copy, joined by lines whose mean length is the KL."), []),
],
"35": [
 ("A FunSearch-style loop: a language model proposes changes to code, a checker scores them, the best become parents.",
  "Pots of seedlings where a gardener grafts new shoots onto the strongest plants.", F("evo('run')"), ["a92"]),
 ("The checker: an online bin packer scored against its L1 lower bound, run in a sandbox the code cannot touch.",
  "A glass cage around the measuring table, so no plant can reach the ruler.",
  NEW("Items dropping into bins; a floor line at the L1 bound; a glass box drawn round the scorer."), []),
 ("Islands, clusters and parents drawn by score; measure the gain over random search and sampling alone at equal budget.",
  "Separate benches of pots, each bench breeding its own line.",
  NEW("Several island clouds; parents chosen by score; one baseline cloud for random search and one for sampling with no selection."), []),
 ("Without the verifier, the search learns to inflate its own scores and loses a better rule it had found.",
  "A plant that grows a fake ruler beside itself.", F("evo('cheat')"), ["a93"]),
 ("Read an evolve-set gain against a held-out gain; they differ.",
  "The prize plant wilts a little when moved to a new garden.", F("prompts('heldout')"), []),
],
"25": [
 ("Write a grid world, its rules and a breadth-first search for distances.",
  "A sand table with an 8 × 8 grid of walls and one flag.", F("grid8('world')"), ["a88"]),
 ("Two models of it: one predicts the next cell, a JEPA predicts the next latent.",
  "Two crystal balls on the table, one showing the next square, one a blur that is meant to mean it.",
  NEW("The grid world beside a cloud of latent points; an action arrow moves the agent cell and the latent point together."), []),
 ("Plan by random shooting: imagine many action sequences, score them, take the first move of the best, replan.",
  "Dozens of ghost pawns race out across the sand; the real pawn takes one step after the best ghost.", F("grid8('plan')"), ["a89"]),
 ("A low JEPA loss does not mean a useful latent: check against a floor, a ceiling and a random-weight control.",
  "One crystal ball shows the same blur whatever you do, and still claims a perfect score.", F("collapse()"), []),
],
"31": [
 ("Render a small world as frames and collect trajectories from a noisy expert.",
  "A dark hut where a pinhole throws the moving picture of two rooms onto the wall.",
  NEW("A 64 × 64 point grid showing two rooms and an agent dot; frames step as the agent moves."), []),
 ("Train a JEPA world model end to end from pixels with a prediction loss and SIGReg.",
  "The picture on the wall is squeezed into a small round cloud of marbles that never bunch up.", F("spread()"), []),
 ("Tell a collapsed latent from a useful one with probes, collapse meters and a distance curve.",
  "A gauge on the wall: if every marble rolls to one spot, the needle drops to zero.", F("collapse()"), []),
 ("Plan in the latent with CEM: imagined paths narrow round by round.",
  "A sheaf of chalk paths on the wall that tightens each round toward the goal.",
  NEW("A cloud of imagined latent paths from the agent; each round keeps the best and resamples, the sheaf narrowing to the goal."), []),
 ("A latent that holds the agent can still fail a planner; PLDM's extra terms on module 25's grid show why.",
  "The camera image is sharp, yet the pawn walks into a wall.", F("grid8('plan')"), []),
],
"15": [
 ("A ReLU network cuts its input space into pieces; inside each piece it is linear.",
  "A map on the table cut into flat tiles by straight folds.", F("relu(1,4) then relu(4,14)"), ["a22", "a23"]),
 ("Count the regions a line crosses exactly; the count grows with width.",
  "A string pulled across the map, a bead at every fold it crosses.",
  NEW("A segment between two inputs crossing region boundaries; beads mark each crossing; the count rises as the net widens."), []),
 ("Intrinsic dimension: TwoNN and MLE estimate how much room the data takes, checked on sets of known dimension.",
  "A sheet of paper rolled into a scroll: in a room of three dimensions it still has two.",
  NEW("A swiss roll of 420 points in 3-D, unrolling to a flat sheet; two estimates printed beside it."), []),
 ("PCA needs far more components than the intrinsic dimension on curved data.",
  "A ruler laid along the scroll, needing many straight pieces to follow the curve.", F("pca()"), []),
 ("Linear CKA compares two networks' representations; a high score alone does not show they learned the same thing.",
  "Two maps that look alike, and a third, drawn by a random hand, that looks alike too.", F("panels('tidy')"), ["a57"]),
],
"11": [
 ("A forward hook reads a module's output, and another can edit it.",
  "A stethoscope pressed to one wire of the running model, and a clamp beside it.",
  NEW("The layers formation with one node ringed; its value shown, then clamped and the downstream points shift."), []),
 ("A probe per layer: train a line to read one fact from the residual stream; deeper layers often make it plain; it needs a random-weight control.",
  "A plumb line dropped at each floor of a shaft: at some floor it falls cleanly between two piles.", F("clouds('probe')"), ["a53"]),
 ("Activation patching: copy one activation from a clean run into a corrupt one; if the answer returns, you found where it lives.",
  "Swap one wire between two identical radios; one radio starts playing the right station.", F("matrix('patch')"), ["a54"]),
 ("Read an MLP neuron by its top contexts on both its positive and its negative pole.",
  "A two-headed bell that rings one tune when pushed and another when pulled.",
  NEW("A single neuron point with two fans of context points, one per sign of its activation."), []),
 ("Wiring: split a neuron's inputs exactly into the writes that feed it, and find later neurons that read it.",
  "Threads from earlier rooms tied to the bell, and threads from the bell to rooms above.", F("graph('back')"), []),
],
"12": [
 ("Toy superposition: a small model can pack more features than it has dimensions when they are sparse.",
  "Five weathervanes crammed onto one roof built for two.", F("superpos('rays') then superpos('interfere')"), ["a50", "a51"]),
 ("A TopK sparse autoencoder rewrites each activation as a few directions from a long dictionary.",
  "A thick dictionary on a lectern; each sentence of the model is spelled with three or four of its words.", F("dict()"), ["a52"]),
 ("Measure an SAE: variance explained, L0, dead latents, and the loss when its output replaces the layer.",
  "Four dials on the lectern; one counts pages no one ever opens.",
  NEW("Four small meters in a row, one per measure, with the dead-latent share as faint points that never light."), []),
 ("Steering: add a direction during generation and replies lean toward it; a sentiment score alone cannot judge it.",
  "A finger pushing the weathervane toward one word, and the model's talk drifting that way.", F("steer()"), ["a55"]),
 ("Neurons against random directions, with an untrained copy and a GELU null.",
  "The same dictionary written by a monkey looks almost as tidy.", F("panels('reveal')"), []),
],
"13": [
 ("One InceptionV1 channel tested on images you draw: a claim about a unit needs a comparison set and its rank among neighbours.",
  "A lens that lights on V shapes, set among twenty other lenses that light on them too.",
  NEW("A row of 21 channel points with bars of response to V images; the claimed channel ranked among its neighbours."), []),
 ("Feature visualisation paints the image a channel wants; say what it does and does not show.",
  "A painter who paints only what makes one lens glow.",
  NEW("Points start as noise and are pushed to raise one channel's response, forming a repeating pattern."), []),
 ("Probe a vision transformer layer by layer against a random-weight control.",
  "Plumb lines down the lens tower, and a second tower of random glass beside it.", F("clouds('probe')"), []),
 ("Read a neuron by its top images, and know three neurons prove nothing.",
  "Three portraits on the wall, and a sign underneath: 'three'.", F("panels('reveal')"), []),
 ("Probe a JEPA encoder for what it keeps: position, neighbours, path length; read a negative R².",
  "A telescope turned to the yard's pavilions, scoring how well the tile world can be read back.",
  NEW("Probe scores as bars for three targets, trained against random encoders, with one below zero."), []),
],
"32": [
 ("Build a model whose units are known, then grade the tools against that truth.",
  "A clock built with its gears behind glass, so you know every wheel.", F("truth()"), ["a58"]),
 ("Score a dictionary against the truth by best cosine, MMCS.",
  "Each found gear is held up to the real gears and scored by the closest fit.",
  NEW("Found directions and true directions as rays; each found ray is matched to its nearest true ray with the cosine shown."), []),
 ("The random-weight control: some scores barely change on a model with random weights.",
  "A second clock of random parts on the lookout ledge, ticking almost as neatly.", F("panels('reveal')"), ["a56"]),
 ("Match two seeds' dictionaries one to one; seed agreement says what is stable.",
  "Two telescopes pointed at the same sky from two towers, ticking off the stars both see.",
  NEW("Two dictionaries' directions paired by a one-to-one matching; unmatched ones fade."), []),
 ("An attribution graph, activation times gradient in a frozen model, must be tested by ablation.",
  "A map of which gears drive the hands; remove a gear and see whether the hands stop.", F("graph('back')"), []),
],
}

def build():
    rooms = []
    for (m, name, floor, level, rect, colour) in ROOMS:
        objs = []
        for k, (idea, image, form, moves) in enumerate(OBJ[m], 1):
            objs.append({"id": f"{m}-{k}", "idea": idea, "image": image, "formation": form,
                         "moves": moves, "links": links(m)})
        doors = sorted({b for a, b in DOORS if a == m} | {a for a, b in DOORS if b == m})
        rooms.append({"id": m, "module": SLUG[m], "name": name, "floor": floor, "level": level,
                      "rect": rect, "size": [rect[2] - rect[0], rect[3] - rect[1]],
                      "colour": colour, "doors": doors, "objects": objs, "links": links(m)})
    halls = []
    for (hid, floor, level, rect, note) in HALLS:
        doors = sorted({b for a, b in DOORS if a == hid} | {a for a, b in DOORS if b == hid})
        halls.append({"id": hid, "floor": floor, "level": level, "rect": rect, "note": note, "doors": doors})
    route, prev = [], None
    for n, (rid, why) in enumerate(ROUTE, 1):
        via = path_via(prev, rid) if prev else []
        route.append({"n": n, "room": rid, "via": via, "why": why})
        prev = rid
    return {"title": "The model house", "units": "metres; x east, y north; level = storey, 4 m each",
            "rule": "Ask what happens to the weights, and you know where you are: before the house (garden), being learned (below ground), used once (ground floor), changed after training (first floor), frozen while the words change (second floor), scored by the world (yard), read only (tower).",
            "floors": FLOORS, "halls": halls, "rooms": rooms, "route": route,
            "landmarks": LANDMARKS,
            "prologue": {"where": "garden_path, east end", "moves": ["a1", "a2", "a3", "a4"],
                         "what": "Before the first bower, the garden gate: 420 points in a loose field (a1), then a grid of numbers (a2), then two kinds of points and a first wrong line (a3, a4). It opens the walk the way the Moves page opens."},
            "epilogue": {"where": "32, the lookout", "moves": ["a94", "a95", "a96", "a97", "a98", "a99", "a100"],
                         "what": "At the top, the Moves page's last chapter: five ideas that kept coming back, each pointing at rooms below: hide and predict (16, 33), one local rule (01), lookup as a dot product (04, 21, 27), propose, score, keep (22, 34, 35), no number without its control (05a, 32)."}}

LANDMARKS = [
    {"id": "residual_stream", "where": "residual_corridor", "what": "A channel of blue water down the ground-floor corridor from the gate to the balcony. Every ground-floor room has a door on it and pours in; from any room you can see it."},
    {"id": "gradient_channel", "where": "gradient_corridor", "what": "Directly below, in the cellar, a channel of gold water runs the other way, west. Grates in the ground-floor corridor show it."},
    {"id": "balcony_view", "where": "balcony", "what": "The output balcony looks east over the old garden: you can see the string-line bed where the walk began, farthest away because it is oldest."},
    {"id": "garden_steps", "where": "garden_path", "what": "Steps from the garden go down under the balcony into the cellar: the loss is computed at the output, so learning starts beneath it."},
    {"id": "core_by_gate", "where": "core", "what": "The stair core stands beside the gate. The cellar stair comes up there because the gradient reaches the embedding table last; every upper floor is reached from where text enters."},
    {"id": "picture_door", "where": "08", "what": "The picture door opens south onto the yard path: images come from the world and skip the gate's merges: they are cut into patches instead."},
    {"id": "light_well", "where": "16", "what": "A light well in the yard path looks down into the blindfold room (16): the JEPA objective below, the JEPA world models (25, 31) across the path."},
    {"id": "glass_over_atrium", "where": "06", "what": "The tailor's room floor is glass over the atrium: thin matrices are added above the frozen block."},
    {"id": "grate_over_shelves", "where": "28", "what": "The north row is one storey high with a skylight roof; from the first-floor gallery outside the three-coin room you look down through the skylight onto the scriptorium's KV shelves (07), which 28 also rounds."},
    {"id": "outside_stair", "where": "outside_stair", "what": "The only way from the house's top floor to the yard is the outside stair from the workshop of hands: agents are where the model starts to act."},
    {"id": "tower", "where": "32", "what": "The tower stands at the south-west corner and is the tallest thing. From the lookout you see the whole roof, the yard and, far east, the garden: the place to rehearse the walk."},
]

def neighbours(node):
    return {b for a, b in DOORS if a == node} | {a for a, b in DOORS if b == node}

HALL_IDS = {h[0] for h in HALLS}

def path_via(a, b):
    """Shortest door path from room a to room b that crosses halls only."""
    q, seen = deque([(a, [])]), {a}
    while q:
        node, path = q.popleft()
        for nb in sorted(neighbours(node)):
            if nb == b:
                return path
            if nb in HALL_IDS and nb not in seen:
                seen.add(nb)
                q.append((nb, path + [nb]))
    raise ValueError(f"no door path from {a} to {b}")

def validate(p):
    errs = []
    ids = [r["id"] for r in p["rooms"]]
    modules = sorted(p_.name for p_ in (COURSE / "modules").iterdir() if p_.is_dir())
    if sorted(r["module"] for r in p["rooms"]) != modules:
        errs.append("rooms do not cover each module exactly once")
    if len(set(ids)) != len(ids):
        errs.append("duplicate room ids")
    nodes = {r["id"]: r for r in p["rooms"]} | {h["id"]: h for h in p["halls"]}
    for a, b in DOORS:
        if a not in nodes or b not in nodes:
            errs.append(f"door to unknown node {a}-{b}")
    for n in nodes.values():
        for d in n["doors"]:
            if n["id"] not in nodes[d]["doors"]:
                errs.append(f"door {n['id']}->{d} not symmetric")
    seq = [s["room"] for s in p["route"]]
    if sorted(seq) != sorted(ids) or len(seq) != len(set(seq)):
        errs.append("route does not visit every room exactly once")
    for s0, s1 in zip(p["route"], p["route"][1:]):
        chain = [s0["room"], *s1["via"], s1["room"]]
        for x, y in zip(chain, chain[1:]):
            if y not in nodes[x]["doors"]:
                errs.append(f"route step {s1['n']}: no door {x}-{y}")
        if any(v not in HALL_IDS for v in s1["via"]):
            errs.append(f"route step {s1['n']} crosses a room")
    moves = json.loads((HERE / "moves_index.json").read_text())["moves"]
    for r in p["rooms"]:
        if not 3 <= len(r["objects"]) <= 7:
            errs.append(f"{r['id']} has {len(r['objects'])} objects")
        for o in r["objects"]:
            for mv in o["moves"]:
                if mv not in moves:
                    errs.append(f"{o['id']}: unknown move {mv}")
                elif moves[mv].get("module") not in (None, r["module"]):
                    errs.append(f"{o['id']}: move {mv} links {moves[mv]['module']}, not {r['module']}")
            L = o["links"]
            for f in (COURSE / "explorables" / L["explorable"], COURSE / "quizzes" / f"{L['quiz']}.html", COURSE / "modules" / L["module"] / "README.md"):
                if not f.exists():
                    errs.append(f"{o['id']}: missing {f}")
    used = [mv for r in p["rooms"] for o in r["objects"] for mv in o["moves"]]
    dup = sorted({m for m in used if used.count(m) > 1})
    return errs, dup, sorted(set(moves) - set(used))

if __name__ == "__main__":
    p = build()
    errs, dup, unused = validate(p)
    text = json.dumps(p, indent=1, ensure_ascii=False) + "\n"
    check = "--check" in sys.argv[1:]
    if check:
        if not OUT.exists() or OUT.read_text() != text:
            errs.append(f"{OUT.name} is stale: run tools/build_palace.py")
    else:
        OUT.write_text(text)
    print("rooms", len(p["rooms"]), "objects", sum(len(r["objects"]) for r in p["rooms"]),
          "route", len(p["route"]))
    print("moves used twice:", dup)
    print("moves not placed:", unused)
    print("errors:", errs or "none")
    sys.exit(1 if errs else 0)
