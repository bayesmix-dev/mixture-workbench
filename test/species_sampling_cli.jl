module SpeciesSamplingCLITests
using Test, TOML, GeneralDPPMixtures
include("../src/cli/SpeciesSamplingCLI.jl")
const C = SpeciesSamplingCLI

@testset "Pitman-Yor execution contract" begin
    mktempdir() do dir
        data=joinpath(dir,"observations.csv")
        write(data,"y\n-1.2\n0.0\n1.1\n")
        path=joinpath(dir,"settings.toml")
        raw=Dict{String,Any}("method"=>"pitman_yor", "iterations"=>12, "warmup"=>4,
            "thin"=>2,"chains"=>2,"seed"=>42,"initial_clusters"=>2)
        function config(c)
            open(path,"w") do io; TOML.print(io,c); end
            path
        end
        for (key,value) in (("method","hdp"),("discount",1.0),("strength",-1.0),
                ("observation_sd",0.0),("base_sd",Inf),("base_mean",true),
                ("iterations",true),("warmup",12),("thin",100),("seed",-1),
                ("initial_clusters",4),("save_allocations",1),("split_merge",true),
                ("betas",[1.0,0.8]),("components",3))
            out=joinpath(dir,"invalid")
            @test_throws ArgumentError C.run(config(merge(raw,Dict(key=>value))),data,out)
            @test !ispath(out)
        end
        out=joinpath(dir,"first")
        @test C.run(config(raw),data,out)==out
        meta=TOML.parsefile(joinpath(out,"metadata.toml"))
        @test meta["run_status"]=="completed"
        @test meta["M_definition"]=="represented_occupied_atoms_equals_k"
        @test meta["total_components"]=="infinite"
        @test meta["empirical_bayes"]===false
        @test meta["mixing_assessment"]=="unassessed"
        @test length(readlines(joinpath(out,"draws.csv")))==9
        @test length(readlines(joinpath(out,"allocations.csv")))==25
        duplicate=joinpath(dir,"duplicate")
        C.run(config(raw),data,duplicate)
        for file in ("draws.csv","allocations.csv","diagnostics.csv")
            @test read(joinpath(out,file))==read(joinpath(duplicate,file))
        end
        @test_throws ArgumentError C.run(path,data,out)
        # Negative strength is valid for positive discount. A single observation
        # exercises the deterministic first table without log(strength).
        write(data,"y\n0.0\n")
        singleton=merge(raw,Dict("strength"=>-0.2,"discount"=>0.4,"initial_clusters"=>1,
            "save_allocations"=>false))
        one=joinpath(dir,"singleton")
        C.run(config(singleton),data,one)
        @test !isfile(joinpath(one,"allocations.csv"))
        @test TOML.parsefile(joinpath(one,"metadata.toml"))["run_status"]=="completed"
    end
end
end
