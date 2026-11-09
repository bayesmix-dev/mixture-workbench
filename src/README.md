# Source navigation

The source uses a hybrid hierarchy. Component calculations and prior laws have separate homes; each model family owns its state representations, targets and transitions. Directory names describe responsibility. Existing Julia modules and public type names remain the entry points; a directory does not create a module or a universal interface.

| Directory | Responsibility |
|---|---|
| [core](core/README.md) | Numerical helpers, random streams and shared utilities |
| [components](components/README.md) | Implemented observation/base combinations and their available calculations |
| [priors](priors/README.md) | Partition laws and point-process constructions |
| [models](models/README.md) | Family-specific states, targets, caches and algorithms |
| [execution](execution/README.md) | Chain execution, workers and run assembly |
| [diagnostics](diagnostics/README.md) | Partition summaries and diagnostic calculations |
| [calibration](calibration/README.md) | Explicit prior and geometry calibration procedures |
| [cli](cli/README.md) | Configuration validation and the shared file protocol |

Start from [GeneralDPPMixtures.jl](GeneralDPPMixtures.jl) for package assembly and the family README for a particular sampler. [Benchmark.jl](execution/Benchmark.jl) assembles the original Gaussian benchmark modules. The portable command-line entry points are documented in [docs/cli.md](../docs/cli.md).

For an extension, locate the responsibility that changes: an observation/base calculation in `components/`, a prior law in `priors/`, or a retained representation and its transitions in `models/<family>/`. A new hierarchy such as an HDP needs an explicit model implementation and its own state and target; naming it in configuration does not generate one. Read the relevant [recipe](../recipes/) and [target specification](../docs/targets/README.md) before sharing existing transitions.

Keep specialized algorithms beside the states they update. Share machinery only when its state, target and random-stream requirements agree. The existing collapsed, retained-atom and ordered-stick representations illustrate why a common model name alone is insufficient.

Tests remain under [test/](../test/), with names identifying the affected model or calculation. Worked implementations and historical prompts remain under [examples/](../examples/); archived prompts are provenance data, not instructions for future agents.

The structural migration preserves the original module identities and sampler calculations. Its [validation record](../docs/source-layout-validation.json) includes full regression checks and exact seeded comparisons with the preceding commit.
