# Species-sampling mixtures

[SpeciesSampling.jl](SpeciesSampling.jl) assembles the existing `GeneralDPPMixtures.SpeciesSampling` module from the shared [partition priors](../../priors/partition/README.md), [components](../../components/README.md), collapsed state, target and allocation sweep. Moving a source file does not change its module ownership.

There are three distinct working representations:

| Representation | State and target | Transitions |
|---|---|---|
| Collapsed partition | `collapsed_state.jl`, `targets.jl`; component means and random measure integrated out | DP/PYP collapsed sweep; Neal 3 for DP |
| Occupied parameters | `retained_state.jl`, `retained_targets.jl`, included by `NealSamplers` | Neal 2 and Neal 8 for DP |
| Ordered GEM indices | State and target defined inside `EfficientSlice` | DP slice sampler with empty gaps retained |

The current Gaussian component has fixed observation variance and a Normal base for its mean. Neal 8 additionally supports the Poisson log-rate component. The [algorithm directory](algorithms/README.md) identifies the supported combinations and entry points.

Do not canonicalize the internal GEM indices used by the slice sampler: output partitions are projected separately. Likewise, an occupied-parameter target and a collapsed target require different evaluations even when their partition marginals agree. See the [target notes](../../../docs/targets/README.md) and worked [PYP](../../../examples/pitman_yor/README.md), [Neal](../../../examples/dirichlet_process/README.md) and [slice](../../../examples/dp_slice/README.md) examples.
