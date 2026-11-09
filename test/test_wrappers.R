# Run with Rscript test/test_wrappers.R. No Julia dependencies are needed.
arguments <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", arguments[startsWith(arguments, "--file=")])
root <- normalizePath(file.path(dirname(script), ".."), mustWork = TRUE)
source(file.path(root, "interfaces", "R", "mixture_workbench.R"))

if (.Platform$OS.type != "unix") {
  cat("Skipping POSIX fake-executable tests on this platform\n")
  quit(save = "no", status = 0L)
}

directory <- tempfile("workbench wrappers ")
dir.create(directory)
project <- file.path(directory, "project with spaces and ' quote")
dir.create(file.path(project, "bin"), recursive = TRUE)
writeLines("# test placeholder", file.path(project, "bin", "run.jl"))
config <- file.path(directory, "settings with $literal.toml")
data <- file.path(directory, "data with ' quote.csv")
output <- file.path(directory, "output with spaces")
fake_julia <- file.path(directory, "fake julia")
writeLines("seed = 123", config)
writeLines(c("x1", "1", "2"), data)
writeLines(c(
  "#!/bin/sh",
  "if [ ! -f \"$5\" ] || [ ! -f \"$7\" ]; then exit 12; fi",
  "if /usr/bin/grep -q 'fail = true' \"$5\"; then exit 17; fi",
  "mkdir -p \"$9\" || exit 13",
  "printf '%s\\n' \"$@\" > \"$9/argv.txt\"",
  "printf '%s\\n' 'chain,iteration,k,M,entropy,log_target' '1,10,2,3,0.693,-12.5' > \"$9/draws.csv\"",
  "printf '%s\\n' 'chain,metric,attempted,accepted,value,status' '1,split_merge,4,1,0.25,assessed' > \"$9/diagnostics.csv\"",
  "printf '%s\\n' 'key,value' 'schema_version,1' 'run_status,completed' 'description,\"quoted, value\"' > \"$9/metadata.csv\"",
  "printf '%s\\n' 'schema_version = 1' > \"$9/metadata.toml\""
), fake_julia)
Sys.chmod(fake_julia, "0755")

fit <- run_sampler(config, data, output, julia = fake_julia, project_root = project)
expected <- c("--startup-file=no", paste0("--project=", normalizePath(project)),
              normalizePath(file.path(project, "bin", "run.jl")),
              "--config", normalizePath(config), "--data", normalizePath(data),
              "--output", output)
stopifnot(identical(readLines(file.path(output, "argv.txt")), expected))
stopifnot(inherits(fit, "mixture_fit"), nrow(fit$draws) == 1L,
          fit$draws$k == 2, fit$draws$log_target == -12.5,
          fit$diagnostics$accepted == 1L,
          fit$metadata[["schema_version"]] == "1",
          fit$metadata[["description"]] == "quoted, value",
          is.null(fit$allocations), identical(as.data.frame(fit), fit$draws))
stopifnot(identical(.mixture_workbench_root, root))
inferred_api_output <- file.path(directory, "inferred API output")
inferred_api_fit <- run_sampler(config, data, inferred_api_output, julia = fake_julia)
inferred_api_argv <- readLines(file.path(inferred_api_output, "argv.txt"))
stopifnot(inherits(inferred_api_fit, "mixture_fit"))
stopifnot(identical(inferred_api_argv[[2]], paste0("--project=", root)))

entrypoint <- file.path("examples", "model with spaces", "sample.jl")
dir.create(dirname(file.path(project, entrypoint)), recursive = TRUE)
writeLines("# alternate model", file.path(project, entrypoint))
alternate <- run_sampler(config, data, output, julia = fake_julia,
                         project_root = project, entrypoint = entrypoint)
alternate_arguments <- readLines(file.path(output, "argv.txt"))
stopifnot(identical(alternate_arguments[[3]], normalizePath(file.path(project, entrypoint))))
missing_entrypoint <- tryCatch(
  run_sampler(config, data, output, julia = fake_julia, project_root = project,
              entrypoint = "missing.jl"),
  error = identity
)
stopifnot(inherits(missing_entrypoint, "error"))

writeLines(c("chain,iteration,observation,cluster", "1,10,1,2"),
           file.path(output, "allocations.csv"))
reloaded <- read_sampler_output(output)
stopifnot(reloaded$allocations$cluster == 2)

writeLines("fail = true", config)
failure <- tryCatch(
  run_sampler(config, data, file.path(directory, "failed output"),
              julia = fake_julia, project_root = project),
  mixture_sampler_error = identity
)
stopifnot(inherits(failure, "mixture_sampler_error"), failure$status == 17L)

cli <- file.path(root, "interfaces", "R", "run_sampler.R")
cli_arguments <- c(cli, "--config", config, "--data", data,
                   "--output", file.path(directory, "cli output"),
                   "--julia", fake_julia, "--project-root", project,
                   "--entrypoint", entrypoint)
cli_status <- system2(file.path(R.home("bin"), "Rscript"), shQuote(cli_arguments),
                      stdout = FALSE, stderr = FALSE)
stopifnot(cli_status == 17L)
writeLines("seed = 123", config)
cli_status <- system2(file.path(R.home("bin"), "Rscript"), shQuote(cli_arguments),
                      stdout = FALSE, stderr = FALSE)
stopifnot(cli_status == 0L)

# The published CLI command omits --project-root: infer it from the script even
# when invoked from another working directory.
inferred_output <- file.path(directory, "inferred root output")
inferred_arguments <- c(cli, "--config", config, "--data", data,
                         "--output", inferred_output, "--julia", fake_julia)
old_directory <- getwd()
setwd(directory)
inferred_status <- system2(file.path(R.home("bin"), "Rscript"), shQuote(inferred_arguments),
                           stdout = FALSE, stderr = FALSE)
setwd(old_directory)
stopifnot(inferred_status == 0L)
inferred_argv <- readLines(file.path(inferred_output, "argv.txt"))
stopifnot(identical(inferred_argv[[2]], paste0("--project=", root)))
stopifnot(identical(inferred_argv[[3]], file.path(root, "bin", "run.jl")))

for (run_status in c("running", "failed")) {
  writeLines(c("key,value", "schema_version,1", paste0("run_status,", run_status)),
             file.path(output, "metadata.csv"))
  partial <- tryCatch(read_sampler_output(output), error = identity)
  stopifnot(inherits(partial, "error"), grepl("Run is not completed", conditionMessage(partial)))
}
writeLines(c("key,value", "schema_version,2", "run_status,completed"),
           file.path(output, "metadata.csv"))
unsupported <- tryCatch(read_sampler_output(output), error = identity)
stopifnot(inherits(unsupported, "error"), grepl("Unsupported output schema", conditionMessage(unsupported)))
writeLines(c("key,value", "schema_version,1", "run_status,completed"),
           file.path(output, "metadata.csv"))

writeLines(c("chain,k", "1,2"), file.path(output, "draws.csv"))
malformed <- tryCatch(read_sampler_output(output), error = identity)
stopifnot(inherits(malformed, "error"), grepl("Missing columns", conditionMessage(malformed)))

unlink(directory, recursive = TRUE)
cat("R wrapper contract tests passed\n")
