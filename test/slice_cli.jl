module SliceCLITests
using Test, TOML
include("../src/SpeciesSamplingCLI.jl")
const C=SpeciesSamplingCLI
@testset "Efficient slice execution contract" begin
    mktempdir() do dir
        data=joinpath(dir,"data.csv")
        write(data,"y\n-1.2\n0.0\n1.1\n")
        path=joinpath(dir,"settings.toml")
        base=Dict{String,Any}("method"=>"slice_efficient","iterations"=>12,"warmup"=>4,
            "thin"=>2,"chains"=>2,"seed"=>42,"initial_clusters"=>2)
        function run(raw,out)
            open(path,"w") do io;TOML.print(io,raw);end
            C.run(path,data,out;family=:slice)
        end
        for (key,value) in (("method","neal8"),("strength",0.0),("discount",0.2),
                ("likelihood","poisson_lognormal"),("auxiliaries",3),("max_components",20),
                ("split_merge",true),("betas",[1.0,0.8]))
            out=joinpath(dir,"invalid")
            @test_throws ArgumentError run(merge(base,Dict(key=>value)),out)
            @test !ispath(out)
        end
        firstout=joinpath(dir,"first")
        run(base,firstout)
        meta=TOML.parsefile(joinpath(firstout,"metadata.toml"))
        @test meta["run_status"]=="completed"
        @test meta["M_definition"]=="materialized_ordered_gem_prefix_including_empty_atoms"
        @test meta["log_target_is_projection"]===true
        @test meta["tail_stopping"]=="residual_mass_below_minimum_slice_no_fixed_truncation"
        @test occursin("ordered_log_target",first(readlines(joinpath(firstout,"draws.csv"))))
        @test first(readlines(joinpath(firstout,"parameters.csv")))=="chain,iteration,cluster,theta,gem_index"
        second=joinpath(dir,"second")
        run(base,second)
        for name in ("draws.csv","allocations.csv","parameters.csv","diagnostics.csv")
            @test read(joinpath(firstout,name))==read(joinpath(second,name))
        end
        @test_throws ArgumentError run(base,firstout)
        silent=joinpath(dir,"noallocations")
        run(merge(base,Dict("save_allocations"=>false)),silent)
        @test !isfile(joinpath(silent,"allocations.csv"))
        @test isfile(joinpath(silent,"parameters.csv"))
    end
end
end
