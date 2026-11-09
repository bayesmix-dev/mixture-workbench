# Lessons carried into the workbench

The workbench grew from investigations of Gaussian MFM and repulsive-mixture samplers. The source and tests preserve the resulting implementations. These observations guide extension tasks; they are not universal rankings of models or proposals.

## A valid move may contribute little

Split–merge proposals can have negligible acceptance in a particular posterior even when their correction is implemented correctly. Inspect selection, allocation, parameter and geometry contributions separately, then measure movement after accepted proposals. Reallocation at fixed occupied count can address a different obstruction from changing the count.

## Counts can conceal partition disagreement

Two chains may agree on occupied counts while assigning observations differently. Compare partition and predictive summaries alongside scalar counts. Keep model/prior sensitivity separate from disagreement caused by incomplete exploration.

## A ladder requires its own diagnostics

Many replicas multiply computation. A temperature path must permit movement from hot regions back to the cold posterior; local swap acceptance alone does not establish this. Record runtime and assess independent cold chains before interpreting a tempered run.

## Conditional and marginal calculations have different jobs

Analytic covariance integration can coexist with a coupled repulsive prior on locations. A convenient integral used to build a proposal is not automatically the target density. In powered moves, distinguish an integral raised to a power from an integral of a powered likelihood.

## Independent checks are reusable research tools

Retain target reconstruction, forced reverse replay and cache checks when a sampler evolves. They let an agent identify which assumption a new likelihood or prior invalidates before running an expensive experiment.

Use the [validation guide](validation.md) and [mixing recipe](../recipes/diagnose-mixing.md) to turn these observations into concrete checks.
