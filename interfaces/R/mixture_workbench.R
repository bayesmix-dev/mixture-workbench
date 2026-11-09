# Base-R interface to the implemented Julia command-line sampler.
.mixture_workbench_root <- local({
  source_file <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (is.null(source_file)) NULL else normalizePath(
    file.path(dirname(source_file), "..", ".."), mustWork = TRUE
  )
})

.mixture_path <- function(path, must_work = TRUE) {
  stopifnot(is.character(path), length(path) == 1L, !is.na(path), nzchar(path))
  normalizePath(path.expand(path), winslash = "/", mustWork = must_work)
}

.mixture_read_numeric <- function(path, required) {
  table <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  missing <- setdiff(required, names(table))
  if (length(missing)) stop("Missing columns in ", basename(path), ": ",
                            paste(missing, collapse = ", "), call. = FALSE)
  for (column in names(table)) {
    values <- suppressWarnings(as.numeric(table[[column]]))
    if (any(is.na(values) & !is.na(table[[column]]))) {
      stop("Invalid numeric value for ", column, " in ", path, call. = FALSE)
    }
    table[[column]] <- values
  }
  table
}

.mixture_read_parameters <- function(path) {
  table <- .mixture_read_numeric(path, c("chain", "iteration", "cluster", "theta"))
  for (column in intersect(c("chain", "iteration", "cluster", "gem_index"), names(table))) {
    values <- table[[column]]
    if (any(!is.finite(values) | values < 1 | values != floor(values) |
            values > .Machine$integer.max)) {
      stop("Invalid positive R integer for ", column, " in ", path, call. = FALSE)
    }
    table[[column]] <- as.integer(values)
  }
  if (any(!is.finite(table$theta))) {
    stop("Invalid finite theta in ", path, call. = FALSE)
  }
  table
}

read_sampler_output <- function(output) {
  output <- .mixture_path(output)
  metadata_table <- read.csv(file.path(output, "metadata.csv"),
                             check.names = FALSE, stringsAsFactors = FALSE,
                             colClasses = "character")
  if (!all(c("key", "value") %in% names(metadata_table))) {
    stop("metadata.csv must contain key and value columns", call. = FALSE)
  }
  metadata <- setNames(metadata_table$value, metadata_table$key)
  if (!identical(unname(metadata["schema_version"]), "1")) {
    stop("Unsupported output schema_version", call. = FALSE)
  }
  if (!identical(unname(metadata["run_status"]), "completed")) {
    stop("Run is not completed", call. = FALSE)
  }
  draws <- .mixture_read_numeric(
    file.path(output, "draws.csv"),
    c("chain", "iteration", "k", "M", "entropy", "log_target")
  )
  diagnostics <- read.csv(file.path(output, "diagnostics.csv"),
                          check.names = FALSE, stringsAsFactors = FALSE)
  allocations_path <- file.path(output, "allocations.csv")
  allocations <- if (file.exists(allocations_path)) .mixture_read_numeric(
    allocations_path, c("chain", "iteration", "observation", "cluster")
  ) else NULL
  parameters_path <- file.path(output, "parameters.csv")
  parameters <- if (file.exists(parameters_path)) .mixture_read_parameters(parameters_path) else NULL
  structure(list(
    output = output,
    draws = draws,
    diagnostics = diagnostics,
    metadata = metadata,
    allocations = allocations,
    parameters = parameters
  ), class = "mixture_fit")
}

run_sampler <- function(config, data, output, julia = "julia", project_root = NULL,
                        entrypoint = "bin/run.jl") {
  if (is.null(project_root)) project_root <- .mixture_workbench_root
  if (is.null(project_root)) {
    stop("Supply project_root or source interfaces/R/mixture_workbench.R", call. = FALSE)
  }
  project_root <- .mixture_path(project_root)
  config <- .mixture_path(config)
  data <- .mixture_path(data)
  # normalizePath does not make a nonexistent path absolute on every platform.
  output <- path.expand(output)
  if (!grepl("^(/|[A-Za-z]:[/\\\\]|\\\\\\\\)", output)) {
    output <- file.path(getwd(), output)
  }
  output <- .mixture_path(output, must_work = FALSE)
  entrypoint <- path.expand(entrypoint)
  if (!grepl("^(/|[A-Za-z]:[/\\\\]|\\\\\\\\)", entrypoint)) {
    entrypoint <- file.path(project_root, entrypoint)
  }
  runner <- .mixture_path(entrypoint)
  stopifnot(is.character(julia), length(julia) == 1L, !is.na(julia), nzchar(julia))
  arguments <- c("--startup-file=no", paste0("--project=", project_root), runner,
                 "--config", config, "--data", data, "--output", output)
  # R configures loader paths for its own libraries. Do not let those override
  # Julia's libraries in the child process. system2(env=) is Unix-only for this
  # use; it leaves the caller, PATH and Julia-specific variables unchanged.
  child_environment <- if (.Platform$OS.type == "unix") c(
    "LD_LIBRARY_PATH=", "DYLD_LIBRARY_PATH=", "DYLD_FALLBACK_LIBRARY_PATH="
  ) else character()
  status <- system2(julia, args = shQuote(arguments), stdout = "", stderr = "",
                    env = child_environment)
  if (status != 0L) {
    condition <- structure(
      list(message = paste("Sampler failed with exit status", status),
           call = NULL, status = as.integer(status)),
      class = c("mixture_sampler_error", "error", "condition")
    )
    stop(condition)
  }
  read_sampler_output(output)
}

print.mixture_fit <- function(x, ...) {
  cat("Mixture sampler fit:", nrow(x$draws), "draws\n")
  cat("Output:", x$output, "\n")
  invisible(x)
}

summary.mixture_fit <- function(object, ...) {
  # These marginal summaries are not convergence diagnostics.
  summary(object$draws[c("k", "M", "entropy", "log_target")], ...)
}

as.data.frame.mixture_fit <- function(x, ...) x$draws
