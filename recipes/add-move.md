# Add or change a transition

**Outcome:** a transition whose target, proposal law and state effects can be inspected and checked independently.

1. Read the target notes, the nearest implementation and its tests. For split–merge moves, start at `docs/algorithms/split-merge.md`; for a powered path, also read `docs/algorithms/tempering.md`.
2. State the variables updated and the variables conditioned on. Define the full proposal: branch/anchor/union selection, scan order, auxiliary randomness, parameter draws, support and any dimension/count change.
3. Derive forward and reverse probabilities and the acceptance ratio. Include selection probabilities, reference-measure factors and Jacobians when applicable. An approximation used for a proposal must have its actual probability evaluated.
4. Separate proposal generation from committing a candidate. Keep the original state valid on rejection and document which random streams the attempt advances.
5. Add forced proposal/reverse replay where useful. Compare raw target/proposal differences with the production ratio, rebuild caches independently, and test boundary self-loops and support failures.
6. Check finite-case balance or an independent small posterior where feasible. For tempered moves, test the cold path and an admissible hot case.
7. Run an experiment measuring movement and runtime. Acceptance rate is one diagnostic; assess the quantities the move was meant to improve.

Deliver the derivation, source, tests, example and validation commands. State unresolved checks explicitly.
