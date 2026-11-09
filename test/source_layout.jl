module SourceLayoutChecks
using Test

# The protocol can be included by an extension without loading the benchmark
# execution layer or its model implementations.
include("../src/cli/IOProtocol.jl")

@testset "Standalone I/O protocol and recursive source provenance" begin
    @test !isdefined(@__MODULE__, :Benchmark)
    @test !isdefined(@__MODULE__, :GeneralDPPMixtures)
    mktempdir() do root
        mkpath(joinpath(root,"src","models","example","algorithms"))
        mkpath(joinpath(root,"vendor","projection_DPP","src"))
        write(joinpath(root,"Project.toml"), "name = \"Fixture\"\n")
        write(joinpath(root,"Manifest.toml"), "julia_version = \"1.12.3\"\n")
        write(joinpath(root,"vendor","projection_DPP","src","DPPSplitMerge.jl"), "# vendored transition\n")
        write(joinpath(root,"src","Fixture.jl"), "module Fixture\nend\n")
        nested=joinpath(root,"src","models","example","algorithms","sweep.jl")
        write(nested,"sweep!() = 1\n")
        first=IOProtocol.source_digest(root)
        @test length(first)==64
        @test IOProtocol.source_digest(root)==first
        write(joinpath(root,"src","README.md"), "Navigation only.\n")
        @test IOProtocol.source_digest(root)==first
        write(nested,"sweep!() = 2\n")
        @test IOProtocol.source_digest(root)!=first
        write(nested,"sweep!() = 1\n")
        @test IOProtocol.source_digest(root)==first
        mktempdir() do destination
            relocated=joinpath(destination,"checkout")
            cp(root,relocated)
            @test IOProtocol.source_digest(relocated)==first
        end
        extra=joinpath(dirname(nested),"new_move.jl")
        write(extra,"# new transition\n")
        @test IOProtocol.source_digest(root)!=first
        rm(extra)
        @test IOProtocol.source_digest(root)==first
        csv=joinpath(root,"input.csv")
        write(csv,"observation\n1.5\n2.5\n")
        names,values=IOProtocol.read_numeric_csv(csv)
        @test names==["observation"]
        @test values==reshape([1.5,2.5],2,1)
    end
end
end
