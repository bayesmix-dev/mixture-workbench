# Execution and run assembly

This directory contains [chain execution](chains.jl), [within-chain scheduling](within_chain.jl), [synthetic data generation](synthetic_data.jl), the persistent replica worker pool and the original benchmark run machinery. These files schedule work and preserve run state; family-specific targets, state initialization and moves stay under [models](../models/README.md).

[ReplicaWorkerPool](ReplicaWorkerPool.jl) requires jobs to use explicit RNGs and private mutable state. The pool is a runtime resource owned outside serialized ensembles and checkpoints. Read its source contract before changing parallel execution or sharing mutable workspaces.

[Benchmark.jl](Benchmark.jl) remains the assembly entry point for the original Gaussian benchmark modules and is loaded by `WorkbenchCLI`. Its legacy application-data helpers require assets absent from this portable repository. Use the implemented [CLI entry points](../cli/README.md) for file-based R/Python runs and portable provenance hashing. [CovarianceExchange.jl](CovarianceExchange.jl) retains its original benchmark-specific exchange orchestration.

Changes to scheduling, checkpoint ownership or random-stream consumption require execution-specific checks in addition to posterior calculations. A diagnostic must not consume a production random stream or mutate the state it describes.
