# Data for section 4

| File | Size | What |
|---|---|---|
| `gridjepa_h1_encoder.bin` | 1.4 MB | the encoder's weights only: 12 tensors, 360,224 numbers, a PyTorch state dict for `python/src/grid_encoder.py` |
| `grid_states.npz` | 0.2 MB | 10,000 frames of the tile world: `windows` (15 × 15 tile ids), `scalars` (5), `pos` (agent x, y), `goal_dist` (BFS steps to the goal, `inf` if it cannot be reached), `ep` (episode id) |

## Where they come from

Both come from **gridjepa, the author's earlier experiment**: a JEPA world model trained from scratch
on a small tile world, on a laptop CPU, to learn how such models behave. The encoder is its "h1"
model: horizon-1 prediction, an EMA target encoder with stop-gradient against collapse, an arena of
12 × 12 cells, a 15 × 15 window, a latent of 128 numbers, 10,000 training steps. Only the encoder
is copied here; its predictor and target copy are not.

The frames were collected with gridjepa's own code, on arenas the model never trained on (its
held-out collection seed): 12,000 steps of random walks (60% of episodes keep their last move with probability 0.6), then 10,000 frames drawn at
random. They come from 97 episodes, one arena each. `goal_dist` is gridjepa's breadth-first search
on the live arena, where walls, blocks, doors and hazards block a move.

The weights hold nothing but numbers learned from that tile world. The encoder class in
`python/src/grid_encoder.py` is a copy of gridjepa's, layer for layer, so a seed gives the same
random weights in both.
