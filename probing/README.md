## Probing Experiments

**Experiment 1:** [Hidden state] -> actions
Train a linear regression probe for every layer’s hidden state for every layer on every timestep. Evaluate with R2/ MSE on held-out trajectories.
Baselines:
Normal: Original data
Randomized pairs: randomly shuffle hidden states and action sequences on a trajectory basis
Noise baseline: [Hidden state] -> gaussian noise with same dim as actions.


**Experiment 2:** [Vision encoder outputs] -> actions
Train a linear regression probe for the patch features (vision encoder outputs) on every timestep. Evaluate with R2/ MSE on held-out trajectories.
Baselines:
Normal: Original data
Randomized pairs: randomly shuffle vision encoder outputs and action sequences on a trajectory basis
Noise baseline: [Vision encoder outputs] -> gaussian noise with same dim as actions.


**Experiment 3:** [Hidden state] -> visual concepts (continuous positions/binary states) / visual-language concepts (continuous positions/binary states)
Train a linear regression probe for every layer’s hidden state for every layer on every timestep. Evaluate with R2/ MSE on held-out trajectories.
Visual / visual-language concepts are derived from simulator states. This part is to be implemented later. You should be able to just access a file and get them. The category (continuous/binary) will also be provided.
Baselines:
Normal: Original data
Randomized pairs: randomly shuffle hidden states and concepts on a trajectory basis
Noise baseline: [Hidden state] -> gaussian noise (continuous positions) / random binary values (binary states).


**Experiment 4:** [Vision encoder outputs] -> visual concepts (continuous positions/binary states) / visual-language concepts (continuous positions/binary states)
Train a linear regression probe for the patch features (vision encoder outputs) on every timestep. Evaluate with R2/ MSE on held-out trajectories.
Visual / visual-language concepts are derived from simulator states. This part is to be implemented later. You should be able to just access a file and get them. The category (continuous/binary) will also be provided.
Baselines:
Normal: Original data
Randomized pairs: randomly shuffle vision encoder outputs and concepts on a trajectory basis
Noise baseline: [Vision encoder outputs] -> gaussian noise (continuous positions) / random binary values (binary states).
