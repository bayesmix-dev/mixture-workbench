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
  "if [ -n \"${LD_LIBRARY_PATH-}\" ] || [ -n \"${DYLD_LIBRARY_PATH-}\" ] || [ -n \"${DYLD_FALLBACK_LIBRARY_PATH-}\" ]; then exit 23; fi",
  "if /usr/bin/grep -q 'fail = true' \"$5\"; then exit 17; fi",
  "mkdir -p \"$9\" || exit 13",
  "printf '%s\\n' \"$@\" > \"$9/argv.txt\"",
  "printf '%s\\n' \"${PATH-}\" \"${JULIA_DEPOT_PATH-}\" > \"$9/environment.txt\"",
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
          is.null(fit$allocations), is.null(fit$parameters),
          identical(as.data.frame(fit), fit$draws))
stopifnot(identical(.mixture_workbench_root, root))
inferred_api_output <- file.path(directory, "inferred API output")
inferred_api_fit <- run_sampler(config, data, inferred_api_output, julia = fake_julia)
inferred_api_argv <- readLines(file.path(inferred_api_output, "argv.txt"))
stopifnot(inherits(inferred_api_fit, "mixture_fit"))
stopifnot(identical(inferred_api_argv[[2]], paste0("--project=", root)))

test_child_loader_environment <- function() {
  variables <- c("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
                 "JULIA_DEPOT_PATH")
  saved <- Sys.getenv(variables, unset = NA_character_)
  on.exit({
    Sys.unsetenv(variables)
    present <- saved[!is.na(saved)]
    if (length(present)) do.call(Sys.setenv, as.list(present))
  }, add = TRUE)
  contaminated <- setNames(
    c("/test/R libraries/linux", "/test/R libraries/macos", "/test/R libraries/fallback",
      "/test/Julia depot with spaces"), variables
  )
  do.call(Sys.setenv, as.list(contaminated))
  caller_before <- Sys.getenv(c(variables, "PATH"), unset = NA_character_)
  isolated_output <- file.path(directory, "isolated loader environment")
  isolated_fit <- run_sampler(config, data, isolated_output,
                              julia = fake_julia, project_root = project)
  stopifnot(inherits(isolated_fit, "mixture_fit"))
  stopifnot(identical(Sys.getenv(c(variables, "PATH"), unset = NA_character_), caller_before))
  child_environment <- readLines(file.path(isolated_output, "environment.txt"))
  preserved <- unname(caller_before[c("PATH", "JULIA_DEPOT_PATH")])
  stopifnot(identical(child_environment, preserved))

  failed_config <- file.path(directory, "loader environment failure.toml")
  writeLines("fail = true", failed_config)
  failed <- tryCatch(
    run_sampler(failed_config, data, file.path(directory, "loader environment failed output"),
                julia = fake_julia, project_root = project),
    mixture_sampler_error = identity
  )
  stopifnot(inherits(failed, "mixture_sampler_error"), failed$status == 17L)
  stopifnot(identical(Sys.getenv(c(variables, "PATH"), unset = NA_character_), caller_before))
}
test_child_loader_environment()

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

# Parameter table fixtures are written directly, independently of the fake
# executable and the Julia engine.
parameter_output <- file.path(directory, "parameter contract fixtures")
dir.create(parameter_output)
writeLines(c("key,value", "schema_version,1", "run_status,completed", "parameter_name,log_rate"),
           file.path(parameter_output, "metadata.csv"))
writeLines(c("chain,iteration,k,M,entropy,log_target", "1,10,2,2,.693,-12.5"),
           file.path(parameter_output, "draws.csv"))
writeLines(c("chain,metric,attempted,accepted,value,status", "1,mixing,0,0,NaN,unassessed"),
           file.path(parameter_output, "diagnostics.csv"))
stopifnot(is.null(read_sampler_output(parameter_output)$parameters))
parameter_file <- file.path(parameter_output, "parameters.csv")
writeLines(c("chain,iteration,cluster,theta", "1,10,2,-.75", "2,20,1,3"), parameter_file)
parameter_fit <- read_sampler_output(parameter_output)
stopifnot(is.data.frame(parameter_fit$parameters), nrow(parameter_fit$parameters) == 2L,
          identical(parameter_fit$parameters$chain, c(1L, 2L)),
          identical(parameter_fit$parameters$iteration, c(10L, 20L)),
          identical(parameter_fit$parameters$cluster, c(2L, 1L)),
          identical(parameter_fit$parameters$theta, c(-.75, 3)),
          typeof(parameter_fit$parameters$theta) == "double",
          parameter_fit$metadata[["parameter_name"]] == "log_rate")
writeLines(c("chain,iteration,cluster,theta,gem_index", "1,10,2,.7,5"), parameter_file)
ordered_parameters <- read_sampler_output(parameter_output)$parameters
stopifnot(identical(ordered_parameters$cluster, 2L),
          identical(ordered_parameters$gem_index, 5L))
writeLines("chain,iteration,cluster,theta", parameter_file)
empty_parameters <- read_sampler_output(parameter_output)$parameters
stopifnot(is.data.frame(empty_parameters), nrow(empty_parameters) == 0L,
          typeof(empty_parameters$chain) == "integer", typeof(empty_parameters$theta) == "double")
for (missing in c("chain", "iteration", "cluster", "theta")) {
  columns <- setdiff(c("chain", "iteration", "cluster", "theta"), missing)
  writeLines(c(paste(columns, collapse = ","), "1,1,1"), parameter_file)
  invalid <- tryCatch(read_sampler_output(parameter_output), error = identity)
  stopifnot(inherits(invalid, "error"), grepl("Missing columns in parameters.csv", conditionMessage(invalid)))
}
invalid_parameters <- list(
  c("chain", "1.5"), c("iteration", "NaN"), c("cluster", "oops"),
  c("gem_index", "2.5"), c("theta", "oops"), c("theta", "NaN"), c("theta", "Inf")
)
for (column in c("chain", "iteration", "cluster", "gem_index")) {
  invalid_parameters <- c(invalid_parameters, list(c(column, "0"), c(column, "-1"),
                                                  c(column, "2147483648")))
}
for (invalid_value in invalid_parameters) {
  row <- c(chain = "1", iteration = "10", cluster = "2", theta = ".7", gem_index = "5")
  row[[invalid_value[[1]]]] <- invalid_value[[2]]
  writeLines(c(paste(names(row), collapse = ","), paste(row, collapse = ",")), parameter_file)
  invalid <- tryCatch(read_sampler_output(parameter_output), error = identity)
  stopifnot(inherits(invalid, "error"), grepl("Invalid", conditionMessage(invalid)))
}

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
