# Point-process constructions

This directory holds the finite spectral construction and the Mahalanobis geometry used by the repulsive Gaussian mixtures. The retained spectrum is part of the implemented model; transitions must respect its finite rank.

| Source | Owning module |
|---|---|
| [spectrum.jl](spectrum.jl) | `GeneralDPPMixtures`: finite Fourier spectrum |
| [configurations.jl](configurations.jl) | `PointProcessSampler`: DPP/Strauss prior types and parameters |
| [MahalanobisRepulsion.jl](MahalanobisRepulsion.jl) | `MahalanobisRepulsion`: metric-aware spectral construction |
| [MahalanobisStrauss.jl](MahalanobisStrauss.jl) | `MahalanobisStrauss`: metric-aware Strauss prior |

The [point-process model family](../../models/point_process/README.md) owns allocation state, occupied and unused locations, covariance caches and transition-specific geometry workspaces. Some prior evaluation code remains there because it depends directly on those representations, including the DPP/Strauss prior adapters in `PointProcessSampler`.

Use [calibration](../../calibration/README.md) for procedures that select prior or geometry parameters. Adding a new construction requires an explicit support and density calculation under the reference measure used by its model state; it does not imply that the existing location moves apply unchanged.
