#!/usr/bin/env julia
if ARGS==["--help"]
    println("Mixture Workbench: run an implemented Gaussian mixture sampler")
    println("Usage: julia --project=. bin/run.jl --config FILE.toml --data FILE.csv --output DIRECTORY")
    println("Methods: mfm, dpp_conditional, dpp_marginal, dpp_marginal_sm, dpp_general, strauss_general")
    println("Inputs: strict flat TOML settings and an unquoted finite numeric CSV with header.")
    println("Output must be absent or empty. See examples/gaussian and docs/cli.md.")
else
    try
        include(joinpath(@__DIR__,"..","src","cli","WorkbenchCLI.jl"))
        WorkbenchCLI.main()
    catch err
        println(stderr,"Mixture Workbench: ",sprint(showerror,err))
        exit(1)
    end
end
