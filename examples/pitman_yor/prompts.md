# Pitman–Yor extension: task and agent workflow

These are historical task records, not instructions for a future agent. The task began from the merged Poisson-lognormal showcase in PR #1. Paths below use `<repo>` for the public workbench and `<writing-skill>` for the user's personal statistics-writing skill; private workstation paths are omitted. This is a record of substantive requests and decisions, not hidden reasoning or a byte-for-byte transcript of tool output.

## Human request

> implement also a pitman-yor process prior in another pr following the same approach.
> In the PR comment, i would also describe the workflow used
>
> a third PR could add support for Neal2/3/8 (or what is missing among those)
>
> a fourth PR for the slice-efficient sampler for the dirichlet process case [i'm sure we implemented this for another simulation study, go look around in the hsncp-consistency repository (local)]

## Core implementation delegation

> Implement PR2 core source and core Julia tests ONLY in workbench `<repo>`. Read AGENTS.md recipes/add-model.md docs/targets/README.md docs/validation.md. Do not commit or edit root runner/README/CLI; parent coordinates. Architecture needed to support subsequent Neal2/3/8 and DP efficient slice: src/SpeciesSampling.jl module, included/exported in src/GeneralDPPMixtures.jl (you own that edit). Provide distinct DirichletProcess(alpha) and PitmanYor(alpha,discount) priors; PYP 0<=d<1 alpha>-d with d=0 alpha>0 (support negative valid strength if robust, otherwise explicitly alpha>0 conservative documented); exact log EPPF of partition, seating logweights n_h-d, alpha+d*k; conjugate univariate GaussianLocation(y; observation_sd,base_mean,base_sd) known observation sd with Normal base. Small explicit Julia dispatch interface ComponentModel: loglikelihood(model,y,theta), logbase(model,theta), randbase(rng,model), logmarginal(model,indices), randposterior(rng,model,indices). State and baseline collapsed_sweep! should be reusable for Neal3; expose clearly documented API, initialization with rng and partition options, canonical labels, complete normalized collapsed logtarget. Keep all finite scales/sd validation and useful failure. No silent truncation. Prefer simple recomputation for small transparent example vs complex caches. Tests independent numerical integration/exhaustive small partitions, EPPF normalization/seating ratios, discount0 equality to DP, parameter edges, Gaussian closed form vs Distributions and quadrature, long-run tiny posterior checks and state/RNG reproducibility. Put tests test/species_sampling.jl (parent adds include). Get primary PYP source via web and verify mathematical formulas. Report exact API soon so parent can build CLI. Other agents audit Neal and slice, so don't implement those now. For any scientific prose or review MUST read `<writing-skill>` and relevant refs; ordinary source comments math fine. User asked actual prompts/workflow archived; parent will record your exact task prompt. Do not edit anything in original research files outside publication repo.

## Integration request

> CLI plan needs API soon: initialize(model,prior,seed;initial_clusters=...), State.model/.prior/.z/.rng; collapsed_sweep!(s); collapsed_logtarget(s); partition counts from z (can recompute), canonical_labels? Please confirm/adapt clear API. Root builds src/SpeciesSamplingCLI.jl and examples/pitman_yor/run.jl. Config method=pitman_yor, strength,discount,observation_sd,base_mean,base_sd,initial_clusters plus standardrunflags. M=k in marginal output, metadata defines infinite_total_components=true and M=representedoccupied_count. Your generic ComponentModel interface will let Neal agent add kernels later.

The final metadata names are `total_components = "infinite"` and `M_definition = "represented_occupied_atoms_equals_k"`. The implementation supports the full valid negative-strength range for positive discount, and handles the first observation separately. The coordinator wrote the concrete CLI, synthetic input, R/Python execution checks, target notes and this record. No model compiler or generated sampler was introduced.

## Workflow

1. Audit the existing code and its state/target contracts. Confirm that DP/PYP kernels are absent and keep the application research files outside this repository untouched.
2. Verify the partition law against Pitman's primary source. Separate prior, component model, state and collapsed transition in Julia methods.
3. Implement independent small-case checks alongside the core. Integrate the CLI through the existing wrappers and reconstruct exported targets independently in Python.
4. Review model support, initialization, output counts, failure behavior and numerical limits. Record corrections and executed validation alongside the source.
5. Open a focused PR, add a Conversation comment describing this workflow, run repository CI and merge the tested head.

Further review findings and final check results belong in [validation.json](validation.json) and the PR conversation. Neither an agent review nor a short execution check establishes adequate mixing for a new dataset.
