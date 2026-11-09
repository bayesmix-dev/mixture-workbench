# Command-line assembly and file protocol

[IOProtocol.jl](IOProtocol.jl) supplies the shared numeric CSV parser, argument parsing, strict-key helper, metadata writer and recursive source digest. It depends on SHA/TOML and does not assemble benchmark or sampler modules.

[WorkbenchCLI.jl](WorkbenchCLI.jl) assembles the original Gaussian entry point and preserves its helper bindings for existing example scripts. [SpeciesSamplingCLI.jl](SpeciesSamplingCLI.jl) assembles the concrete PYP, Neal and DP slice entry points using the shared protocol independently of the benchmark driver.

Each executable implements the same three arguments: `--config FILE.toml --data FILE.csv --output DIRECTORY`. Configuration chooses implemented behavior and is validated before output is created. A new model or method requires Julia source, configuration checks and output definitions; an unknown name produces an error.

The R/Python wrappers launch these executables and read their files. They do not compile a model specification. Keep their contract aligned with [docs/cli.md](../../docs/cli.md), and test changes with the wrapper and integration checks under [test/](../../test/).

Source digests include Julia files recursively under `src/`, the pinned dependency files and the vendored split–merge source. Completed metadata must be published after result streams are flushed so wrappers can distinguish complete output from an interrupted run.
