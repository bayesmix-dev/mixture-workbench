# Model families

A model family owns its states, targets and specialized algorithms. These objects stay together because an algorithm's validity depends on the variables retained, the variables integrated out and the associated caches.

| Family | Main representations |
|---|---|
| [species_sampling](species_sampling/README.md) | Collapsed DP/PYP partitions, occupied atoms for Neal 2/8, ordered GEM atoms for DP slice sampling |
| [mfm](mfm/README.md) | Finite-mixture partitions, total component count and sampled hyperparameters |
| [point_process](point_process/README.md) | Spectral DPP states and general DPP/Strauss location states |

Reusable observation/base calculations belong in [components](../components/README.md), prior laws in [priors](../priors/README.md), and execution resources in [execution](../execution/README.md). A helper that depends on one family's target or cache belongs with that family even when its name resembles a broadly applicable algorithm.

Read [docs/targets](../../docs/targets/README.md) and the relevant [transition recipe](../../recipes/add-move.md) before transferring a move across representations. A new hierarchy should begin with an explicit state and target in its own family directory, reusing compatible calculations where available.
