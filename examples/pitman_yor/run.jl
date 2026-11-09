#!/usr/bin/env julia
if ARGS == ["--help"]
    println("Pitman-Yor Gaussian location mixture with component means integrated out")
    println("Usage: julia --project=. examples/pitman_yor/run.jl --config FILE.toml --data FILE.csv --output DIRECTORY")
else
    include(joinpath(@__DIR__,"..","..","src","cli","SpeciesSamplingCLI.jl"))
    try
        SpeciesSamplingCLI.main()
    catch error
        println(stderr,"Pitman-Yor example: ",sprint(showerror,error))
        exit(1)
    end
end
