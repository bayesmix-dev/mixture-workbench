# Species-sampling algorithms

| Source | Supported implementation | Owning module |
|---|---|---|
| [collapsed.jl](collapsed.jl) | Collapsed Gaussian allocation sweep with DP or PYP prior | `SpeciesSampling` |
| [NealSamplers.jl](NealSamplers.jl) | Neal 2/3/8 for DP; Gaussian components for all three and Poisson log-rate components for Neal 8 | `NealSamplers` |
| [EfficientSlice.jl](EfficientSlice.jl) | DP slice sampling with ordered GEM indices and Gaussian location components | `EfficientSlice` |

Neal 3 delegates to the shared collapsed DP sweep. Neal 2/8 retain occupied component parameters and include their state and target files from the parent directory. The slice implementation owns a separate ordered state whose empty gaps matter to its updates.

Add a transition beside the representation it updates, specify its target and required component operations, and add independent checks in [test/](../../../../test/). The current command-line combinations are explicit in [SpeciesSamplingCLI.jl](../../../cli/SpeciesSamplingCLI.jl); unsupported combinations fail before sampling. See the [family README](../README.md) for the state map and examples.
