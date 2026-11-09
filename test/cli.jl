module CLITests
using Test, TOML
include("../src/cli/WorkbenchCLI.jl")
const W=WorkbenchCLI

@testset "Strict settings and portable input contract" begin
    mktempdir() do dir
        function config(values)
            p=joinpath(dir,"settings.toml")
            open(p,"w") do io; TOML.print(io,values); end
            p
        end
        c=W.settings(config(Dict("method"=>"mfm","iterations"=>10,"warmup"=>2,"thin"=>2)))
        @test c["betas"]==[1.0]
        @test c["iterations"]==10
        for values in (
            Dict("likelihood"=>"invented"), Dict("method"=>"hdp"),
            Dict("iterations"=>true), Dict("seed"=>-1), Dict("chains"=>0),
            Dict("iterations"=>10,"warmup"=>10), Dict("weight_shape"=>0.2),
            Dict("spectrum_target"=>3.), Dict("method"=>"dpp_marginal","split_merge"=>true),
            Dict("method"=>"dpp_marginal_sm","betas"=>[1.,.8]),
            Dict("betas"=>[.8,.6]), Dict("betas"=>[1.,1.]), Dict("betas"=>[1.,NaN]))
            @test_throws ArgumentError W.settings(config(values))
        end
        data=joinpath(dir,"input with spaces.csv")
        write(data,"a,b\n1,2\n3,4\n")
        names,A=W.read_numeric_csv(data)
        @test names==["a","b"]
        @test A==[1. 2.;3. 4.]
        for invalid in ("a,a\n1,2\n", "a,b\n1\n", "a,b\n1,NaN\n", "a,b\n1,\n", "a,b\n1,Inf\n")
            write(data,invalid)
            @test_throws ArgumentError W.read_numeric_csv(data)
        end
        @test_throws ArgumentError W.parse_args(["--config","a","--config","b","--output","c"])
        @test_throws ArgumentError W.parse_args(["--config","a","--data","b","--surprise","c"])
        @test W.csvcell("a,\"b\"")=="\"a,\"\"b\"\"\""
        out=joinpath(dir,"should not be created")
        @test_throws ArgumentError W.run(config(Dict("unknown"=>1)),data,out)
        @test !ispath(out)
    end
end
end
