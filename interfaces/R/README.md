# R interface

The wrapper uses base R. It runs an implemented Julia sampler as a separate
process and reads the output into data frames. First instantiate the Julia
environment as described in the repository README.

From R, with the repository as your working directory:

```r
source("interfaces/R/mixture_workbench.R")
fit <- run_sampler("path/to/config.toml", "path/to/data.csv", "results/r-run")
summary(fit)
head(as.data.frame(fit))
fit$diagnostics
fit$metadata
fit$allocations

# Read an existing run without starting Julia:
fit <- read_sampler_output("results/r-run")
```

`summary(fit)` gives marginal numerical summaries, not convergence diagnostics.
`fit$allocations` is `NULL` when allocation output was disabled. Component labels
are local to each draw; the wrapper does not relabel components across draws.
The wrapper does not interpret a model language or choose an inference algorithm.
The TOML configuration must select an implemented model and supported settings.

To run from the command line:

```sh
Rscript interfaces/R/run_sampler.R \
  --config path/to/config.toml \
  --data path/to/data.csv \
  --output results/r-run
```

Both interfaces accept an optional `julia` / `--julia` executable and
`project_root` / `--project-root` repository path. Arguments are quoted for
`system2()`, including paths containing spaces. A failed sampler raises an R
condition of class `mixture_sampler_error` with a `status` field; the command-line
wrapper forwards that status. Configuration and data validation are performed by
the Julia engine. Runs are synchronous; this wrapper provides no background-job
or checkpointing API.

On Unix, the wrapper clears `LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH` and
`DYLD_FALLBACK_LIBRARY_PATH` for the Julia child. This prevents R's shared-library
search paths from overriding Julia's libraries. The caller's R environment,
`PATH`, Julia settings such as `JULIA_DEPOT_PATH`, and the selected executable
remain unchanged. R documents its loader-path setup in
[R Installation and Administration](https://cran.r-project.org/doc/manuals/R-admin.html#External-software);
the child-only override uses [`system2(env=)`](https://stat.ethz.ch/R-manual/R-devel/library/base/html/system2.html).
Windows uses its usual process environment because `system2(env=)` has different
support there.

If a custom Julia build needs additional library paths, pass a launcher script
through `julia = "/path/to/launcher"` or `--julia /path/to/launcher`. That script
can set the required paths and then `exec /path/to/julia "$@"`.

Use `entrypoint = "path/to/implemented-run.jl"` (or `--entrypoint` on the
command line) to run another implemented Julia script following the same
protocol. Relative entrypoints resolve against `project_root`; the default is
`bin/run.jl`. This selects source code, not a declarative model specification.
