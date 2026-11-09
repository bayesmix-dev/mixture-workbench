# Prior laws and constructions

This directory contains prior calculations that can be identified separately from a sampler's working state.

- [partition](partition/README.md): Dirichlet-process and Pitman–Yor laws, EPPFs and seating weights.
- [finite_mixture/counts.jl](finite_mixture/counts.jl): the finite-label falling-factorial helper included by `MFMTarget`.
- [point_process](point_process/README.md): finite spectral constructions and location geometry for repulsive mixtures.

Model-dependent allocation factors and hyperparameter updates can remain with their [model family](../models/README.md) when they require that family's state. In particular, the MFM target and its component-count and weight-shape updates remain under `models/mfm/`.

A prior implementation supplies its law and support. The corresponding model owns counts, caches, retained variables and valid transitions. Read [add-model](../../recipes/add-model.md) and [target specifications](../../docs/targets/README.md) before reusing an algorithm for a different prior.
