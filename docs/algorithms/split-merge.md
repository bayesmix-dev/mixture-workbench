# Split–merge and reallocation moves

The repository contains several model-specific proposals. Select one by its state and target requirements, rather than by the name “split–merge”.

| Implementation | Main requirements | Checks |
|---|---|---|
| [src/MFMSequentialSplitMerge.jl](../../src/MFMSequentialSplitMerge.jl) | MFM NIW evidence and predictive calculations; ordered observation anchors and a replayable allocation order | [test/mfm_sequential_splitmerge.jl](../../test/mfm_sequential_splitmerge.jl), [test/mfm_sequential_balance.jl](../../test/mfm_sequential_balance.jl) |
| [src/MFMBlockMoves.jl](../../src/MFMBlockMoves.jl) | MFM state, canonical partition target and explicit proposal bookkeeping | [test/mfm_block_moves.jl](../../test/mfm_block_moves.jl) |
| [src/DPPSequentialSplitMerge.jl](../../src/DPPSequentialSplitMerge.jl) | Marginal DPP state, fixed spectral subset, proposal location law and full geometry correction | [test/dpp_sequential_splitmerge.jl](../../test/dpp_sequential_splitmerge.jl), [test/dpp_sequential_balance.jl](../../test/dpp_sequential_balance.jl) |
| [vendor/projection_DPP/src/DPPSplitMerge.jl](../../vendor/projection_DPP/src/DPPSplitMerge.jl) | Gaussian cluster calculations, DPP location proposal and explicit forward/reverse densities | [test/runtests.jl](../../test/runtests.jl) |
| [src/DPPCollapsedReallocation.jl](../../src/DPPCollapsedReallocation.jl) | Fixed-count DPP reallocation and its integrated proposal calculations | [test/dpp_collapsed_reallocation.jl](../../test/dpp_collapsed_reallocation.jl), [test/dpp_collapsed_balance.jl](../../test/dpp_collapsed_balance.jl), [test/dpp_powered_collapsed_balance.jl](../../test/dpp_powered_collapsed_balance.jl) |

Read each module's `propose!`, `commit!` and `raw_ratio` together. A proposal records its candidate and probability terms before committing changes. This separation permits inspection and independent checks of rejected proposals and acceptance ratios.

## Sequential proposals

For the MFM sequential proposal, two ordered anchors start nonempty daughter groups. Each remaining observation is added once in a sampled order. A merge is deterministic conditional on the selected union; its reverse probability is evaluated by replaying the sequential split at the original oriented partition. Count factors depend on whether the proposal also changes `M`.

The DPP sequential proposal uses an auxiliary cluster integral to construct allocations and a location law to propose component locations. The target still includes joint DPP geometry and fixed-location covariance-collapsed likelihoods. An auxiliary integral used for allocation scores does not replace those target terms. Draws outside the location domain and splits beyond the fixed rank are rejected according to the implemented proposal law.

## Extending a proposal

Document anchor and union selection, scan order, auxiliary variables, proposed parameters, support and count changes. Evaluate the reverse law at the candidate with the same auxiliary conventions. Retain terms that cancel in the derivation until their cancellation has been checked under the actual selection law.

If a new likelihood removes a required integral or proposal density, implement a replacement proposal with an evaluable probability. An approximate score may guide a proposal when its actual forward and reverse probabilities are included; it is not an approximation to insert silently into the target.

Follow [the transition recipe](../../recipes/add-move.md). Passing balance and state tests addresses the cases checked. Acceptance and mixing require separate experiments.
