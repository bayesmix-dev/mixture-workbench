#!/usr/bin/env julia
if ARGS == ["--help"]
    println("Dependent slice-efficient sampler for a Dirichlet-process Gaussian location mixture")
    println("Usage: julia --project=. examples/dp_slice/run.jl --config FILE.toml --data FILE.csv --output DIRECTORY")
else
    include(joinpath(@__DIR__,"..","..","src","cli","SpeciesSamplingCLI.jl"))
    try
        SpeciesSamplingCLI.main(;family=:slice)
    catch error
        println(stderr,"DP slice example: ",sprint(showerror,error))
        exit(1)
    end
end
