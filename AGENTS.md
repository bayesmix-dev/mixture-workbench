# Agent entry point

This is a Julia research workbench with file-based R and Python command-line wrappers. Work on complete source implementations. Configuration files select implemented behaviour; do not infer a sampler from an unfamiliar model name or silently substitute a supported model.

## Navigate by task

| Task | Read before editing |
|---|---|
| Add an observation distribution | `recipes/add-likelihood.md` |
| Change a component base measure | `recipes/add-base-measure.md` |
| Add a hierarchy or allocation prior | `recipes/add-model.md` |
| Add or modify a transition | `recipes/add-move.md` |
| Investigate mixing | `recipes/diagnose-mixing.md` |
| Change inputs, outputs or wrappers | `docs/cli.md` |

Then read `docs/targets/README.md`, the relevant algorithm document and its source and tests. Use `docs/validation.md` to choose checks appropriate to the change.

## Commands

Run from the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl
python3 -m unittest discover -s test -p 'test_*.py'
Rscript test/test_wrappers.R
```

For a CLI smoke run, use the command in `README.md`. Keep runtime results under `results/` or another ignored output directory.

## Invariants

- State the variables retained in the target and the variables integrated out. Keep the full target separate from proposal-only calculations.
- Maintain allocation counts, active slots, sufficient statistics, predictive caches and geometry together. A rejected proposal must leave the sampler state intact; proposal RNG advancement follows the documented transition.
- Evaluate forward and reverse proposal probabilities under the same selection law, including anchors, order, auxiliaries, support constraints and count changes.
- A fixed DPP spectrum has finite rank. A split beyond that rank is a self-loop; do not increase the rank or add numerical jitter to obtain acceptance.
- The current likelihood-tempering paths power an already collapsed likelihood. An integrated powered proposal density is a separate object; preserve the distinction documented in `docs/algorithms/tempering.md`.
- Keep random streams independent where the implementation requires them. Diagnostics must not mutate production state or consume its random streams.
- Validate configuration keys and values. Unsupported combinations must fail with a useful error before sampling.
- Preserve chain and iteration identifiers. Treat component labels as local to a draw and use label-invariant summaries for comparisons.

## Deliver an extension

Include the model/state specification, affected source, a runnable synthetic example, independent checks, and the commands and results of validation. Record substantive agent prompts and the resulting design decisions when an extension is presented as an agent workflow. Remove credentials, private data and infrastructure details before publication.

Report calculation checks and mixing evidence separately. Mark checks as passed, failed or unassessed; do not turn a smoke run into a claim that a chain has converged. Keep the supported CLI settings aligned with the implementation and documentation.
