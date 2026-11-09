module NealCLITests
using Test, TOML
include("../src/SpeciesSamplingCLI.jl")
const C=SpeciesSamplingCLI
@testset "Neal sampler execution contract" begin
    mktempdir() do dir
        data=joinpath(dir,"data.csv")
        write(data,"y\n0\n1\n3\n")
        path=joinpath(dir,"settings.toml")
        base=Dict{String,Any}("method"=>"neal8","iterations"=>12,"warmup"=>4,
            "thin"=>2,"chains"=>1,"seed"=>12,"initial_clusters"=>2)
        function run(raw,out)
            open(path,"w") do io;TOML.print(io,raw);end
            C.run(path,data,out;family=:dirichlet_process)
        end
        for (key,val) in (("method","invented"),("discount",0.0),("strength",0.0),
                ("likelihood","invented"),("auxiliaries",0),("auxiliaries",true),
                ("split_merge",true),("proposal_sd",0.5))
            out=joinpath(dir,"invalid")
            @test_throws ArgumentError run(merge(base,Dict(key=>val)),out)
            @test !ispath(out)
        end
        for method in ("neal2","neal3")
            out=joinpath(dir,"unsupported")
            @test_throws ArgumentError run(merge(base,Dict("method"=>method,"likelihood"=>"poisson_lognormal")),out)
            @test !ispath(out)
            @test_throws ArgumentError run(merge(base,Dict("method"=>method,"auxiliaries"=>1)),out)
            @test !ispath(out)
        end
        for method in ("neal2","neal3","neal8"), likelihood in ("gaussian_location","poisson_lognormal")
            likelihood=="poisson_lognormal" && method!="neal8" && continue
            out=joinpath(dir,method*likelihood)
            raw=merge(base,Dict("method"=>method,"likelihood"=>likelihood))
            run(raw,out)
            meta=TOML.parsefile(joinpath(out,"metadata.toml"))
            @test meta["run_status"]=="completed"
            @test meta["sampler"]==method
            @test meta["M_definition"]=="represented_occupied_atoms_equals_k"
            @test isfile(joinpath(out,"parameters.csv"))==(method!="neal3")
            @test occursin("partition_log_target",first(readlines(joinpath(out,"draws.csv")))) ==
                (method!="neal3" && likelihood=="gaussian_location")
        end
    end
end
end
