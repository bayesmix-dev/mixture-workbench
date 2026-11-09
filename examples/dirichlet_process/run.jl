#!/usr/bin/env julia
if ARGS == ["--help"]
    println("Dirichlet-process mixtures: Neal algorithms 2, 3 and 8")
    println("Usage: julia --project=. examples/dirichlet_process/run.jl --config FILE.toml --data FILE.csv --output DIRECTORY")
else
    include(joinpath(@__DIR__,"..","..","src","SpeciesSamplingCLI.jl"))
    try
        SpeciesSamplingCLI.main(;family=:dirichlet_process)
    catch error
        println(stderr,"Dirichlet-process example: ",sprint(showerror,error))
        exit(1)
    end
end
