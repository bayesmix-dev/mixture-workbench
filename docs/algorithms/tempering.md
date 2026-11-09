# Likelihood tempering

The implemented likelihood paths keep the model prior fixed and power a likelihood after specified variables have been integrated out. If `P(s)` is the retained-state prior factor and `L(s)` is that collapsed likelihood, the path is

```text
log pi_beta(s) = log P(s) + beta * log L(s) + constant(beta).
```

Only the `beta = 1` slot has the original posterior target. Each independent diagnostic chain needs its own ladder and random streams.

| Model | Definition of `L` | Source | Checks |
|---|---|---|---|
| MFM | Product of NIW cluster evidences, integrating component means and covariances | [src/MFMLikelihoodTempering.jl](../../src/MFMLikelihoodTempering.jl), [src/MFMTarget.jl](../../src/MFMTarget.jl) | [test/mfm_likelihood_tempering.jl](../../test/mfm_likelihood_tempering.jl) |
| Marginal DPP | Product of covariance-collapsed cluster likelihoods conditional on retained locations | [src/DPPLikelihoodTempering.jl](../../src/DPPLikelihoodTempering.jl), [src/primitives.jl](../../src/primitives.jl) | [test/dpp_likelihood_tempering.jl](../../test/dpp_likelihood_tempering.jl), [test/dpp_powered_collapsed_balance.jl](../../test/dpp_powered_collapsed_balance.jl) |

For compatible states and fixed temperatures, the symmetric swap log ratio is `(beta_i - beta_j) * (log L(s_j) - log L(s_i))`. Compatibility includes using the same underlying model and prior. Replica provenance, state and random streams must follow the implementation's swap convention.

## Integration and powering

Powering a collapsed likelihood and integrating a powered likelihood generally give different functions:

```text
(integral f(y | theta) G(dtheta))^beta
integral f(y | theta)^beta G(dtheta)
```

The distinction also occurs inside proposal construction. The powered DPP allocation proposal uses `A_beta`, which integrates the fixed-location covariance-collapsed likelihood raised to `beta` over candidate centres. It is not `A^beta`. This extra integration constructs a proposal and does not remove the retained locations from the DPP target. See [src/DPPCollapsedReallocation.jl](../../src/DPPCollapsedReallocation.jl) and [src/DPPSequentialSplitMerge.jl](../../src/DPPSequentialSplitMerge.jl).

Some powered auxiliary integrals require a properness condition on `beta`. Respect the source gates and documented fallback proposals; do not extrapolate a cold formula to arbitrary temperatures. The public CLI accepts only implemented method/tempering combinations and validates its ladder before sampling.

## Assessing a ladder

Check the cold-chain quantities alongside adjacent swap attempts/acceptances, physical-replica movement and round trips when those are recorded. Adjacent swap acceptance alone does not establish transport through the full ladder. Compare independent cold chains through partition and predictive summaries as well as component counts.

The short CLI examples exercise execution. They do not choose a generally suitable temperature ladder. See [the mixing recipe](../../recipes/diagnose-mixing.md).
