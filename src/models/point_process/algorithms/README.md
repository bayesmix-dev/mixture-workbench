# Point-process transitions

These algorithms operate on the retained representations described in the [family guide](../README.md). Choose the state and target before selecting a transition.

| Operation | Sources |
|---|---|
| Spectral conditional and marginal sweeps | [spectral.jl](spectral.jl) |
| Reallocation | [DPPReallocation](DPPReallocation.jl), [DPPMultiReallocation](DPPMultiReallocation.jl), [DPPCollapsedReallocation](DPPCollapsedReallocation.jl) |
| Sequential, global and triple split–merge | [DPPSequentialSplitMerge](DPPSequentialSplitMerge.jl), [DPPGlobalSplitMerge](DPPGlobalSplitMerge.jl), [DPPTripleSplitMerge](DPPTripleSplitMerge.jl) |
| Likelihood and bias tempering | [DPPLikelihoodTempering](DPPLikelihoodTempering.jl), [DPPBiasTempering](DPPBiasTempering.jl), [PartitionARIContrast](PartitionARIContrast.jl) |
| Reference, prior-refresh and fixed-count helpers | [DPPEnsembleReferences](DPPEnsembleReferences.jl), [DPPPriorRefresh](DPPPriorRefresh.jl), [DPPFixedCount](DPPFixedCount.jl) |

The original split–merge module remains under [vendor/projection_DPP](../../../../vendor/projection_DPP/src/DPPSplitMerge.jl). The general DPP/Strauss sweep stays with its state in [PointProcessSampler](../PointProcessSampler.jl). In particular, a DPP-specific move is not enabled for Strauss by placing both in this family.

`PartitionARIContrast` is a cached workspace used by bias tempering, so it remains beside the algorithms that consume it. Ordinary partition summaries belong in [diagnostics](../../../diagnostics/README.md).

The [benchmark assembly](../../../execution/Benchmark.jl) loads the original module dependencies in order. The [CLI](../../../cli/WorkbenchCLI.jl) exposes a narrower set of supported combinations. Read the corresponding `dpp_*.jl` tests under [test/](../../../../test/) and the [move recipe](../../../../recipes/add-move.md) before changing proposal laws or geometry updates.
