# Contributing

Changes should make a concrete model or algorithm easier to inspect, extend or check. Start with the [agent entry point](AGENTS.md) and the relevant [recipe](recipes/).

For a statistical extension, include:

1. The model, retained state and target density, with the conditioning and integration made explicit.
2. The implemented transitions and their requirements. For Metropolis–Hastings moves, identify selection laws, auxiliaries, forward/reverse probabilities and support restrictions.
3. A small synthetic example with fixed configuration and seed.
4. Independent calculation or transition checks and a short mixing assessment appropriate to the example.
5. Reproduction commands and their outcomes, including unassessed checks.

An agent-assisted showcase should also include the substantive prompts, enough context to reproduce the development task, the resulting file changes and any human corrections. Preserve the distinction between the original prompt and a retrospective explanation. Do not publish credentials, personal paths, private datasets or copied conversation unrelated to the extension.

Keep changes scoped. Reuse a shared abstraction when its assumptions match; introduce a model-specific implementation when the target or state differs. Avoid adding configuration options without an implemented, checked code path.

Run the checks listed in `AGENTS.md`. For expensive checks, record their runtime and explain any checks omitted from a proposed change. A short wrapper test need not run a long MCMC experiment; target and transition changes need evidence beyond wrapper success.

Describe a pull request through the problem, resulting behaviour and validation. Report limitations where they affect interpretation. Include a configuration and compact tables or summaries for computational claims; keep large generated files outside Git.
