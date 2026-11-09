# Diagnostics and partition summaries

This directory contains the original [partition summaries](summaries.jl), including entropy, coclustering and ARI calculations. Preserve chain and iteration identities when exporting summaries, and treat component labels as local to each draw. The cached [PartitionARIContrast](../models/point_process/algorithms/PartitionARIContrast.jl) used by bias tempering stays with that algorithm because its refresh rules are part of the transition.

Diagnostic evaluation must leave production state and its random streams unchanged. If a summary needs scratch space or simulation, give it a private copy and an explicit independent RNG. A reference contrast used by a transition must still follow that transition's documented refresh and cache rules.

Target evaluations used in sampler decisions belong with the relevant [model](../models/README.md); a target value printed as a summary retains the same model-specific definition. Read the [target notes](../../docs/targets/README.md) before comparing scores or component counts across representations. See [diagnose-mixing](../../recipes/diagnose-mixing.md) for an investigation workflow and [validation](../../docs/validation.md) for the scope of existing checks.
