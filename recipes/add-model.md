# Add a different model

**Outcome:** a complete model implementation with its own state, target and transitions, following the repository's execution and output conventions.

1. Write the generative specification and identify the observational units, groups, shared parameters and model-specific latent variables. For an HDP, grouped data and shared atoms change the state; renaming the MFM allocation prior is insufficient.
2. Identify the target's reference measure, retained variables, integrated variables, support and normalising terms needed by transitions. Read `docs/targets/README.md` to identify calculations that can actually be reused.
3. Implement the state and initialization. Reuse numerical utilities when their assumptions match; keep model-specific invariants explicit.
4. Implement a baseline sweep before adding block moves or tempering. Specify the conditional or Metropolis–Hastings law of each update.
5. Add independent target calculations and tiny posterior references where feasible. Check state rebuilding and initialization from dispersed configurations.
6. Add a synthetic example. Define output identifiers for observations, groups and draws; preserve the shared `chain` and `iteration` convention. Store model-specific quantities separately when needed.
7. Add split–merge or tempering only after deriving them for this target. State their capabilities in the CLI and fail on unsupported combinations.
8. Run the relevant checks and report exploration evidence for the example.

If development uses an agent, preserve the substantive prompts and human corrections as an extension record. Deliver source, target notes, example, reproduction commands and passed/failed/unassessed checks.
