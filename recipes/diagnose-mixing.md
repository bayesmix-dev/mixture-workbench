# Diagnose mixing

**Outcome:** an evidence-based assessment of a particular configuration and an experiment addressing the observed obstruction.

1. Record the commit, data, model settings, initialization, seeds, transitions, run lengths and any temperature ladder. Separate chains from replicas within a ladder.
2. Confirm input validation and state/target checks before interpreting the traces. Read `docs/validation.md`.
3. Run independent chains from dispersed states. Compare occupied counts, entropy, partition summaries and predictions where implemented. Use label-invariant quantities; equal count histograms can conceal different partitions.
4. Inspect move attempts and acceptances, accepted-move displacement and time spent in each update. Separate changing the count from rearranging allocations at fixed count.
5. For tempering, examine adjacent swap counts and replica transport across the ladder. Compare independent cold chains; use only the target-temperature slot for posterior summaries.
6. Form a specific explanation supported by the diagnostics, then change one relevant ingredient: proposal, initialization, ladder or computational budget. Keep the model prior fixed when comparing samplers for one posterior.
7. Report changes in exploration, Monte Carlo error and runtime. Treat prior sensitivity as a separate experiment.

Keep compact configurations and results needed to reproduce the finding. Describe what was assessed and what remains unassessed; do not label a run converged solely because a scalar trace appears stable.
