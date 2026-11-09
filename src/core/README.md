# Shared utilities

These files are included directly into `GeneralDPPMixtures` and preserve its existing helper names.

| Source | Responsibility |
|---|---|
| [random.jl](random.jl) | Explicit seed derivation |
| [numerics.jl](numerics.jl) | Stable numerical and weighted-draw helpers |
| [streams.jl](streams.jl) | Random-stream ownership |
| [partitions.jl](partitions.jl) | Canonical output labels |

Put a helper here when its operation is independent of a particular model state and target. Component sufficient statistics belong in [components](../components/README.md); state geometry and transition caches belong with their [model family](../models/README.md).

Changes to random draws or stream assignment can alter reproducibility even when a function signature stays the same. Keep diagnostic streams independent of production streams and validate the callers affected by a shared numerical change. Canonical output labels must not replace the ordered internal GEM indices of the slice sampler.
