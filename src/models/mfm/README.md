# Mixture of finite mixtures

[MFMBaseline.jl](MFMBaseline.jl) owns the original Gaussian MFM state and baseline updates. [MFMTarget.jl](MFMTarget.jl) evaluates the collapsed canonical-partition target with its total component count and hyperparameters. Gaussian NIW evidence is included from [components](../../components/README.md) into the existing `MFMTarget` module.

The `algorithms/` directory contains MFM hyperparameter, block, split–merge, reference and tempering updates. Their state assumptions and target calculations remain specific to this family. A similarly named DPP move lives with the [point-process family](../point_process/README.md).

The implemented prior uses a shifted-Poisson total component count and sampled weight-shape and count-rate parameters. Its retained variables, reference measure and data-dependent Gaussian initialization are documented in [docs/targets](../../../docs/targets/README.md). Inspect those details before adapting the allocation law or reusing a transition.

Start validation from the `mfm_*.jl` files under [test/](../../../test/), then run the portable suite described in [docs/validation.md](../../../docs/validation.md). CLI assembly remains in [cli](../../cli/README.md) and run orchestration in [execution](../../execution/README.md).
