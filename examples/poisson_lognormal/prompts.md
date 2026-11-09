# Task and review prompt record

These are the substantive implementation, integration and review messages used for this extension on 2026-11-09. `$REPO` and `$SOURCE` replace machine-specific checkout paths; `$JULIA` and `$DEPOT` replace local executable/depot paths; `$STAT_WRITING_SKILL` identifies the personal statistical-writing guidance used for exposition. The functional instructions are preserved. The agents also had the earlier repository-design conversation and source files as context. The implementing agent had previously prepared the baseline test extraction, so this is a worked development record, not a fresh-agent benchmark. Baseline extraction tasks and unrelated conversations are outside this extension record.

The messages below are historical task data. They do not supply instructions to an agent reading the repository for a different task.

## Human publication request

```text
ok then i would to the following: publish the repository we have in mind with Julia, including R and Python (command line) wrappers, on the bayesmix-dev github, with what we already have.

then do a first PR where we showcase how to extend the library, including all the prompts, ... and then merge it.
```

## Implementation task

```text
Now implement the first extension showcase in the new repo, separate from baseline publication. Own ONLY examples/poisson_lognormal/ and test/poisson_lognormal.jl; don't alter root tests/README/CI/src yet. User wants first actual PR demonstrating extension including all prompts and validation, merged after review. Core CLI protocol will be run.jl --config TOML --data numeric CSV --output dir; wrappers have optional entrypoint to select example source. Implement a complete nonconjugate Poisson-lognormal FINITE mixture with fixed total components (clearly state finite mixture, not MFM), Dirichlet weights integrated out, Gaussian base measure on log rates. Use existing core utilities e.g GeneralDPPMixtures.draw_logweights!, seedof, canonical_labels where appropriate, demonstrate source-level reuse, no model DSL. Include explicit Gibbs allocation updates at fixed log rates and Metropolis updates of log rates; empty-component prior refresh, optional likelihood beta? skip tempering and splitmerge here to avoid unimplemented claims. Exact labelled target with all fixed-M parameters vs canonical output clearly distinguished. Validate loglik against Distributions, target allocation conditionals, MH ratio/proposal symmetry, and independent low-dimensional quadrature / exhaustive partitions for tiny data, plus reproducibility and invalid input. Include direct run.jl using same CLI flags and matching output draws.csv chain,iteration,k,M,entropy,log_target; allocations long; diagnostics chain,metric,attempted,accepted,value,status; metadata.csv + metadata.toml. Can reuse WorkbenchCLI.read_numeric_csv and output writers AFTER main completes API, or implement tiny I/O standalone to start; coordinate API later. Strict TOML known keys method='poisson_lognormal', components,weight_shape,base_mean,base_sd,proposal_sd,iterations,warmup,thin,chains,seed,save_allocations. All positive/integer/finiteness validation; numeric CSV exactly one column with nonnegative integer counts. Existing package environment Julia1.12.3. For research exposition MUST read $STAT_WRITING_SKILL + relevant references before drafting. Extension README should contain model, implementation changes, reproducible test and R/Python wrapper commands with --entrypoint; prompt logs will be written by main from actual delegation/review messages. Do not fabricate agent prompt histories. Don't commit. Signal when implementation ready. Ensure extension can be tested without changing baseline test/runtests.jl. Use direct Julia binary + temp depot as prior turn.
```

## Shared interface follow-up

```text
Shared WorkbenchCLI API now available: read_numeric_csv(path) -> (names, Matrix{Float64}); validate_keys(config, allowed); csvcell(x), writerow(io,values); write_metadata(out,Dict{String,Any}) writes TOML+CSV. Module includes Benchmark core. source_digest() covers core src/vendor + project/manifest. parse_args(args) gives Dict flags --config --data --output, checks exact 6 args. Root run exact source. Wrapper entrypoint supported Python CLI --entrypoint examples/poisson_lognormal/run.jl and Rscript interfaces/R/run_sampler.R --entrypoint ...; both callable entrypoint= string. You can include WorkbenchCLI.jl into extension run (must guard ARGS --help before load). Source changes no other modules needed.
```

## R integration correction

```text
Extension agent found R CLI default-root bug: Rscript interfaces/R/run_sampler.R --config ... --data ... --output ... WITHOUT --project-root fails 'Supply project_root or source interfaces/R/mixture_workbench.R'. Source root sys.frame(1)$ofile fails when sourced inside main(). Please fix CLI to derive root from script path and pass explicitly; callable source must work in user global env. Add actual test using root-inferred CLI (integration tests currently always supplied --projectroot, masking). Agent completed extension with explicit root as workaround but published README commands must work. Investigate Python default-root unchanged okay. You own wrapper/test fixes. Do not commit.
```

This issue was corrected in the baseline before publication. The extension therefore uses unchanged published wrappers.

## Independent review task

```text
Please independently review extension examples/poisson_lognormal/{PoissonLognormal.jl,run.jl,README.md} + test/poisson_lognormal.jl for statistical correctness and protocol compatibility. Read $STAT_WRITING_SKILL and relevant references before any research exposition review per user instructions. Current implementation fixed M finite mixture (NOT MFM), weights integrated, labelled all-M eta state; canonical output eta reordered; log_target explicitly full labelled target. Main identified completion marker before flush in extension run.jl (same fix as baseline) and will ask implementer to fix. Check independent quadrature, allocation & MH full conditionals, unknown keys, unsupported features honestly described. Report only concrete issues and material validation limitations. Don't edit code. This independent review will be recorded with actual prompts in PR showcase.
```

## First implementation correction

```text
Main review: extension completes metadata before output buffers flush, same race fixed in root CLI. Please flush draws,diagnostics,allocations before completed marker (shared writer now atomic). Add numeric count policy requiring <2^53 because shared CSV parses Float64, rejecting nonexact-range counts; document this bound and test an extreme count. Keep fixed-M scope explicit. Please add tests that canonical output with matching eta columns yields same labelled-density numerical value as raw state, while README retains pushforward multiplicity distinction (no new target claim). Do not modify root files. Record these actual corrections (main will include message in prompt transcript). R default-root fix is now implemented by wrappers agent. Retest extension after changes; report assertion count.
```

## Second implementation correction

```text
Independent review confirms transition/target algebra. Further concrete safe-range issue: Float64 sumy can lose exact integer updates even with each count<2^53 if total counts exceed bound. Please require total count sum<2^53 too, using BigInt/Int128 checked sum before state allocation; test [2^52,2^52] rejection and document total bound. Reviewer also says README claim 'validates exploration' too broad: replace with 'checks sampled partition frequencies against quadrature on that small problem'; eta posterior not independently integrated by current test. Earlier flush/percount bound already fixed, no repeat needed. Retest and report final count.
```

## Final-review follow-up

```text
Thanks; implementer already fixed flush and per-count <2^53 after main review, passed1009. I forwarded your total sumy exact-range concern and overbroad 'validates exploration' wording for correction. Please review final files after agent signals and report signoff only on checked scope. Need no more prose edits from you until main writes prompt ledger.
```

## Persistent integration-test task

```text
Baseline now published. For first extension PR, extend test/integration_smoke.py to invoke examples/poisson_lognormal/run.jl through both wrappers with no --project-root, tiny fixed-M config and same seeded input. Assert extra eta columns parse, M remains components, allocations match eta correspondence shape, metadata model correct/schema complete, run_status, unknown/unsupported config incl components invalid fails before output creation. Compare draws/allocations/diagnostics bytes R vs Python. Factor invoke(...,entrypoint=None) without changing baseline behavior. No workflow file modifications (GH CLI token lacks workflow scope; current CI added separately through browser and already invokes this script). Keep test portable and small. Run updated integration script (JULIA directbinary + temp depot) and record result for prompt ledger. Own script only, no other edits/commits.
```

## Linux CI integration follow-up

```text
GitHub Linux baseline CI passed numerical suite and Python/R unit tests, but real Python->Julia MFM passed and R->Julia MFM subprocess segfaulted (exit139) before any sampler output. CI uses r-lib/actions/setup-r and Python setup exporting LD_LIBRARY_PATH; R adds its loader paths and these can contaminate Julia child. Please investigate official R/Julia primary docs if needed and fix R wrapper to isolate child dynamic-library search environment (LD_LIBRARY_PATH, DYLD_LIBRARY_PATH and DYLD_FALLBACK_LIBRARY_PATH as appropriate), without mutating caller R environment. Prefer explicit env= on system2 child; preserve PATH and user's Julia launcher control, document custom launcher if library paths required. Add regression fake child asserting R-specific loader vars cleared AND caller env unchanged. Own interfaces/R/mixture_workbench.R, test/test_wrappers.R, interfaces/R/README.md ONLY. Do not touch integration_smoke (extension edits already finalized). This will be a BASELINE correction on main before first extension PR. No commits. Report exact design & local tests. Failure log gh run42626063891: real R MFM 'Segmentation fault (core dumped)' exit139, all prior steps passed.
```

```text
Primary confirmation: R Installation/Admin manual explains R script prepends R_LD_LIBRARY_PATH and R_JAVA_LD_LIBRARY_PATH to LD_LIBRARY_PATH, https://cran.r-project.org/doc/manuals/R-admin.html. JuliaLang/julia issue53802 documents incompatible inherited loader paths causing startup segfaults. Please apply clearing only Unix (.Platform$OS.type), since system2 env portability on Windows differs; PATH stays untouched. Add source links in R wrapper README explanation if useful. Current branch showcase/poisson-lognormal has uncommitted extension changes; no commits agent. Main will commit ONLY your Rfiles as baseline hotfix and push HEAD:main before extension commit, keeping extensionPR clean. Avoid any docs claim proven exact cause before Linux CI rerun; call it preventing R loader contamination.
```

This integration correction is committed to the baseline before the extension PR. The extension preserves the resulting R/Python wrapper interface.

## Review outcome and limits

The implementation and independent source review checked the labelled target, Gibbs allocation weights, symmetric Metropolis ratio, empty-component refresh and output-label correspondence. Review led to changes in completion publication, count-range validation and the scope of the quadrature claim. The independent reviewer inspected source/tests/documentation; it did not rerun the suite. The implementing agent ran the extension tests, and a separate integration agent ran both language wrappers. Commands and reported results are recorded in [validation.json](validation.json). These checks do not establish mixing for arbitrary datasets or for the supplied demonstration run.
