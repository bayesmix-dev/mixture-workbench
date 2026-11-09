#!/usr/bin/env Rscript
# Command-line entry point; source mixture_workbench.R for the R API.
main <- function() {
  raw_arguments <- commandArgs(trailingOnly = FALSE)
  file_argument <- raw_arguments[startsWith(raw_arguments, "--file=")]
  if (length(file_argument) != 1L) stop("Cannot locate the wrapper script")
  script <- normalizePath(sub("^--file=", "", file_argument), mustWork = TRUE)
  source(file.path(dirname(script), "mixture_workbench.R"))
  arguments <- commandArgs(trailingOnly = TRUE)
  usage <- paste(
    "Usage: Rscript interfaces/R/run_sampler.R --config FILE.toml",
    "--data FILE.csv --output DIRECTORY [--julia EXECUTABLE]",
    "[--project-root DIRECTORY] [--entrypoint JULIA_SCRIPT]"
  )
  if ("--help" %in% arguments || "-h" %in% arguments) {
    cat(usage, "\n")
    return(0L)
  }
  # Sourcing inside main() does not reliably expose sys.frame(1)$ofile to the
  # callable wrapper. The CLI already knows its own location, so pass its root.
  parsed <- list(julia = "julia", project_root = normalizePath(
    file.path(dirname(script), "..", ".."), mustWork = TRUE
  ))
  accepted <- c("config", "data", "output", "julia", "project-root", "entrypoint")
  seen <- character()
  i <- 1L
  while (i <= length(arguments)) {
    option <- sub("^--", "", arguments[[i]])
    if (!startsWith(arguments[[i]], "--") || !option %in% accepted ||
        option %in% seen || i == length(arguments)) {
      stop(paste("Invalid or incomplete arguments.", usage), call. = FALSE)
    }
    parsed[[gsub("-", "_", option)]] <- arguments[[i + 1L]]
    seen <- c(seen, option)
    i <- i + 2L
  }
  if (!all(c("config", "data", "output") %in% names(parsed))) {
    stop(usage, call. = FALSE)
  }
  fit <- do.call(run_sampler, parsed)
  print(fit)
  0L
}

status <- tryCatch(main(), mixture_sampler_error = function(error) {
  message(conditionMessage(error))
  error$status
}, error = function(error) {
  message(conditionMessage(error))
  1L
})
quit(save = "no", status = status)
