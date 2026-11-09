# Validation

Assess calculations, transitions and exploration separately. A wrapper can execute successfully while a transition is wrong; a transition can preserve its intended target while exploring it poorly.

| Question | Evidence | Example files |
|---|---|---|
| Is state internally consistent? | Rebuilt counts, sufficient statistics and caches; unchanged state after rejection | [test/mfm_block_moves.jl](../test/mfm_block_moves.jl), [test/runtests.jl](../test/runtests.jl) |
| Are target calculations correct in checked cases? | Independent density calculations, quadrature and small reference problems | [test/mfm_target.jl](../test/mfm_target.jl), [test/runtests.jl](../test/runtests.jl) |
| Are proposal probabilities and corrections consistent? | Forced endpoint replay, independent raw ratios and finite-case balance checks | [test/mfm_sequential_balance.jl](../test/mfm_sequential_balance.jl), [test/dpp_sequential_balance.jl](../test/dpp_sequential_balance.jl), [test/dpp_collapsed_balance.jl](../test/dpp_collapsed_balance.jl) |
| Is the powered target implemented as specified? | Cold-path agreement, powered target differences and swap ratios | [test/mfm_likelihood_tempering.jl](../test/mfm_likelihood_tempering.jl), [test/dpp_likelihood_tempering.jl](../test/dpp_likelihood_tempering.jl), [test/dpp_powered_collapsed_balance.jl](../test/dpp_powered_collapsed_balance.jl) |
| Can users execute and read the run? | CLI and wrapper tests on synthetic data; malformed-input checks | [test/test_wrappers.R](../test/test_wrappers.R), Python wrapper tests |
| Does this run explore adequately? | Independent runs, trace plots, partition/predictive summaries, Monte Carlo diagnostics and runtime | Experiment-specific analysis |

## Run checks

```sh
julia --project=. test/runtests.jl
python3 -m unittest discover -s test -p 'test_*.py'
Rscript test/test_wrappers.R
```

Use the checked-in environment and report the commit and commands. Record passed, failed and unassessed checks. A failed stochastic check deserves investigation under its recorded seed and tolerance, not repeated execution until it passes.

## Independent references

Avoid reproducing the implementation in the test. Derive a small density from its model specification, enumerate a finite case or use numerical integration where possible. For caches, reconstruct from observation membership. For an acceptance calculation, reconstruct the target and proposal contributions independently of the production accumulator.

## Mixing evidence

Start independent chains from dispersed states. Examine label-invariant quantities: occupied counts, entropy, pairwise co-clustering, partition contrasts and predictive summaries where implemented. Agreement on occupied count alone can conceal different partitions. Assess Monte Carlo error and computational cost for quantities relevant to the analysis.

For tempering, record the temperature ladder and distinguish posterior draws from heated states. Independent ladders supply independent cold chains. Observe transport across the ladder in addition to local swap rates.

Short examples establish that code paths run. Balance tests address their checked states and transition laws. Neither supplies a general mixing guarantee.
