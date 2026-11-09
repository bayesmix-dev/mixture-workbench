# Add an observation likelihood

**Outcome:** a complete, runnable implementation for the new observation distribution, with explicit requirements and independent checks. A log-density function alone is not a sampler.

1. Read `docs/targets/README.md` and choose a nearby complete implementation. The current Gaussian sufficient statistics and evidence are in `src/primitives.jl`; the MFM predictive calculations are in `src/MFMBaseline.jl` and `src/MFMTarget.jl`.
2. Specify the observation space, component parameterisation and likelihood. Provide simulation and an independent reference log density, which may be written in R or Python.
3. List the calculations used by each selected transition: observation density, prior draws, cluster evidence, predictive density, parameter proposal and proposal density. Mark which remain analytic and which need replacement.
4. Implement model-specific state and updates where the Gaussian assumptions no longer hold. Do not reuse Gaussian counts/means/scatter or Student laws merely because their interfaces fit. Keep parameter-update tuning explicit.
5. Revisit every split–merge and tempering path selected for the example. Derive new forward/reverse probabilities if parameter proposals or collapsed calculations change. Unsupported paths must fail before sampling.
6. Add a small synthetic example, independent density checks, state/rejection checks and a small posterior reference where feasible. Run the relevant existing checks and assess mixing separately.
7. Expose the implementation through the CLI only after the complete path works. Document settings, outputs, supported moves and checks.

Record the substantive prompts and corrections if this is an agent-assisted showcase. Finish with the model specification, source files, reproduction commands, validation outcomes and unresolved checks.
