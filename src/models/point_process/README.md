# Repulsive point-process mixtures

This family contains the original Gaussian spectral DPP representations and the general location-state sampler for DPP and Strauss priors. State layouts, covariance caches and target calculations belong here because location updates depend on their geometry.

The baseline spectral sampler is in [algorithms/spectral.jl](algorithms/spectral.jl). Its model, state, target and initialization are in [model.jl](model.jl), [states/spectral.jl](states/spectral.jl), [targets.jl](targets.jl) and [initialization.jl](initialization.jl). [PointProcessSampler.jl](PointProcessSampler.jl) defines the general occupied/unused location state and its prior adapters. Shared spectral and metric constructions are under [priors/point_process](../../priors/point_process/README.md). The `algorithms/` directory also holds the specialized reallocation, split–merge, reference-refresh and tempering modules, including the cached `PartitionARIContrast` workspace used by bias tempering.

Choose the state representation before choosing a move. A finite spectral subset, an augmented conditional configuration and a general point-process state retain different variables. The DPP and Strauss location laws also couple component locations, so an observation likelihood and independent base draw do not describe their full target.

Read [docs/targets](../../../docs/targets/README.md), [split–merge](../../../docs/algorithms/split-merge.md) and [tempering](../../../docs/algorithms/tempering.md) alongside the source. Tests under [test/](../../../test/) identify the representation and transition they check. Calibration procedures belong in [calibration](../../calibration/README.md); execution workers belong in [execution](../../execution/README.md).
