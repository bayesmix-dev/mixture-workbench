# MFM transitions

These modules update the Gaussian MFM state defined by [MFMBaseline](../MFMBaseline.jl) and use the [MFM target](../MFMTarget.jl). Their calculations depend on this representation and its component-count and weight-shape parameters.

| Operation | Sources |
|---|---|
| Hyperparameter updates | [MFMHyperparameters](MFMHyperparameters.jl) |
| Block and sequential split–merge | [MFMBlockMoves](MFMBlockMoves.jl), [MFMSequentialSplitMerge](MFMSequentialSplitMerge.jl) |
| Concentration and triple-component moves | [MFMConcentrationSplitMerge](MFMConcentrationSplitMerge.jl), [MFMTripleSplitMerge](MFMTripleSplitMerge.jl) |
| Reference proposals | [MFMReferenceIndependence](MFMReferenceIndependence.jl), [MFMReferenceInvolution](MFMReferenceInvolution.jl) |
| Likelihood and bias tempering | [MFMLikelihoodTempering](MFMLikelihoodTempering.jl), [MFMBiasTempering](MFMBiasTempering.jl) |

The baseline sweep remains with its state in `MFMBaseline`. Existing proposal implementations share some numerical and reference-workspace utilities with point-process modules; those imports retain their original namespaces. Sharing a helper does not identify equal targets or acceptance ratios.

Check the owning loader and tests before using a module. The [portable CLI](../../../cli/WorkbenchCLI.jl) selects specific implemented paths; a source file is not automatically a configurable method. Relevant checks are the `mfm_*.jl` files under [test/](../../../../test/). See the [move recipe](../../../../recipes/add-move.md) when adding a transition.
