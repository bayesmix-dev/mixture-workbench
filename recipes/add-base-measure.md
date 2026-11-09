# Add or change a base measure

**Outcome:** a component-prior implementation whose target contributions and sampler requirements are explicit.

1. Identify the component parameters and whether their prior factorises across components. A joint DPP or Strauss location prior is not an independent base measure.
2. Read `docs/targets/README.md`. For the MFM NIW base measure, inspect `src/MFMBaseline.jl` and `src/MFMTarget.jl`; for coupled locations, inspect `src/PointProcessSampler.jl` and `src/samplers.jl`.
3. Specify density, support, normalising terms and prior simulation. State which parameters remain in the chain and which are integrated out.
4. Re-derive affected evidence, predictive and conditional distributions. If conjugacy is lost, implement an explicit parameter transition or proposal and update the state representation.
5. Audit moves using prior draws or analytic integrals, including birth, split–merge, reallocation and hot-chain updates. Preserve forward/reverse support and include the actual proposal law in the acceptance ratio.
6. Add independent density and prior-simulation checks, a small posterior reference where possible, and state/rejection checks. Demonstrate a synthetic run and assess its mixing.
7. Document supported combinations and reject unsupported settings explicitly.

Deliver the model/state specification, source, example and validation outcomes. Keep correctness evidence and empirical exploration separate.

A complete source-level example is the [Poisson-lognormal finite mixture](../examples/poisson_lognormal/README.md), including its task prompts and validation record.
