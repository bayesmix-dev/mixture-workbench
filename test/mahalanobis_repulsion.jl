# Standalone: julia --project=. test/mahalanobis_repulsion.jl
if !isdefined(Main,:Benchmark);include("../src/execution/Benchmark.jl");end
if !isdefined(Main.Benchmark,:MahalanobisRepulsion)
    Base.include(Main.Benchmark,joinpath(@__DIR__,"../src/priors/point_process/MahalanobisRepulsion.jl"))
end
module MahalanobisRepulsionTests
using Test,GeneralDPPMixtures,LinearAlgebra,StaticArrays,Statistics,Serialization,Random
const G=GeneralDPPMixtures
const R=Main.Benchmark.MahalanobisRepulsion

function bytes(x)
    io=IOBuffer();serialize(io,x);take!(io)
end

# This enumerator uses a different spherical bound, an eigendecomposition, and
# both signed integer frequencies. It does not reuse the factory's box bounds,
# representative rule, shell grouping, or Cholesky factor.
function brute(A,a,c,v,cutoff;extra=3)
    d=size(A,1);radius=(log(c)-log(cutoff))^(1/v)/a
    b=ceil(Int,radius/sqrt(eigmin(Symmetric(A))))+extra
    out=Dict{NTuple{d,Int},Float64}()
    for w in Iterators.product(ntuple(_->(-b):b,d)...)
        q=sum(w[i]*A[i,j]*w[j] for i in 1:d,j in 1:d)
        value=c*exp(-(a*sqrt(q))^v)
        value>=cutoff && (out[w]=value)
    end
    out
end

function signed_spectrum(s)
    d=size(s.frequencies,1);out=Dict{NTuple{d,Int},Float64}()
    for (h,r) in enumerate(s.ranges)
        w=Tuple(s.frequencies[:,first(r)])
        out[w]=s.lambda[h]
        any(!iszero,w) && (out[map(-,w)]=s.lambda[h])
    end
    out
end

function countlaw(ls)
    p=[1.]
    for l in ls
        out=zeros(length(p)+1)
        out[1:end-1].+=(1-l).*p;out[2:end].+=l.*p;p=out
    end
    p
end

function kernel(s,delta)
    sum(s.lambda[s.shell[j]]*cospi(2sum(s.frequencies[e,j]*delta[e] for e in eachindex(delta)))
        for j in eachindex(s.kind))
end

@testset "Mahalanobis Fourier DPP spectrum" begin
    @testset "exact ellipsoid support, frequency pairs and eigenvalues" begin
        for A in ([.7 .3;.3 1.4],[.15 -.12;-.12 .18],[1. 0.;0. 2.]),v in (2.,8.)
            a=.9;c=.81;cutoff=.015
            s=R.build_spectrum(Val(2),a,c,v,cutoff,A)
            actual=signed_spectrum(s);expected=brute(A,a,c,v,cutoff)
            @test Set(keys(actual))==Set(keys(expected))
            for w in keys(expected)
                @test actual[w]≈expected[w] rtol=5e-14 atol=0
            end
            @test size(s.frequencies,2)==length(s.kind)==length(s.shell)
            @test length(s.lambda)==length(s.ranges)==length(s.norms)
            @test all(l->0<l<1,s.lambda)
            @test s.kind[1]==0 && all(iszero,s.frequencies[:,1]) && s.lambda[1]==c
            @test reduce(vcat,collect.(s.ranges))==collect(eachindex(s.kind))
            for (h,r) in enumerate(s.ranges)
                w=s.frequencies[:,first(r)]
                @test s.norms[h]==sum(abs2,w)
                @test all(==(h),s.shell[r])
                @test all(j->s.frequencies[:,j]==w,r)
                @test s.kind[r]==(all(iszero,w) ? UInt8[0] : UInt8[1,2])
                @test s.lambda[h]≈c*exp(-(a*sqrt(dot(w,A*w)))^v) rtol=1e-15
            end
            ls=collect(values(expected));mu=sum(ls);lp0=sum(log1p.(-ls))
            @test s.retained_sum≈mu rtol=1e-14
            @test s.logp0≈lp0 rtol=1e-14
            @test s.conditional_mean≈mu/(-expm1(lp0)) rtol=1e-14
            @test s.logodds≈log.(s.lambda).-log1p.(-s.lambda) rtol=1e-15
        end
        for A in (reshape([.4],1,1),[1. .4 .2;.4 1.2 -.25;.2 -.25 .8]),v in (2.,8.)
            d=size(A,1);s=R.build_spectrum(Val(d),1.3,.81,v,.02,A)
            actual=signed_spectrum(s);expected=brute(A,1.3,.81,v,.02)
            @test Set(keys(actual))==Set(keys(expected))
            for w in keys(expected)
                @test actual[w]≈expected[w] rtol=5e-14 atol=0
            end
        end
        A=[1. 0.;0. 2.]
        s=R.build_spectrum(Val(2),.9,.81,2.,.015,A)
        actual=signed_spectrum(s)
        @test actual[(1,0)]!=actual[(0,1)] # Equal Euclidean norm is not a shell.
        cutoff=.7exp(-1.)
        boundary=R.build_spectrum(Val(2),1.,.7,2.,cutoff,Matrix{Float64}(I,2,2))
        @test Set(keys(signed_spectrum(boundary)))==Set([(0,0),(1,0),(-1,0),(0,1),(0,-1)])
        constant=R.build_spectrum(Val(2),10.,.7,8.,.1,Matrix{Float64}(I,2,2))
        @test constant.frequencies==zeros(Int,2,1)
        @test constant.kind==UInt8[0] && constant.ranges==[1:1]
        @test constant.conditional_mean==1.
    end

    @testset "isotropic kernel and full count law agree with original spectrum" begin
        for d in 1:3,v in (2.,8.)
            A=Matrix{Float64}(I,d,d)
            old=G.build_spectrum(Val(d),1.1,.73,v,.008)
            new=R.build_spectrum(Val(d),1.1,.73,v,.008,A)
            @test new.frequencies==old.frequencies
            @test new.kind==old.kind
            @test new.lambda[new.shell]==old.lambda[old.shell]
            @test countlaw(new.lambda[new.shell])≈countlaw(old.lambda[old.shell]) atol=2e-15
            @test new.conditional_mean≈old.conditional_mean rtol=2e-15
            @test new.logp0≈old.logp0 rtol=2e-15
            for delta in (fill(.07,d),[.13+.14*(e-1)/max(d-1,1) for e in 1:d])
                @test kernel(new,delta)≈kernel(old,delta) atol=2e-14
            end
        end
    end

    @testset "positive kernel, stationarity and orthonormal raw-box Fourier basis" begin
        A=[1. .25;.25 .8]
        for v in (2.,8.)
            s=R.build_spectrum(Val(2),1.3,.7,v,.02,A)
            y=[SVector(-.8,2.2),SVector(.2,4.1),SVector(.7,3.3)]
            lower=SVector(-1.,2.);widths=SVector(2.,3.)
            m=G.Model(y,ones(Int,length(y)),lower,lower+widths,s;
                tau=4.,omega=Matrix{Float64}(I,2,2))
            xs=[SVector(-.83,2.12),SVector(.41,4.72),SVector(.13,3.36),SVector(-.29,4.08)]
            F=G.feature_matrix(m,collect(eachindex(s.kind)),xs)
            K=F'*Diagonal(s.lambda[s.shell])*F/m.volume
            independent=[kernel(s,(x-z)./widths)/m.volume for x in xs,z in xs]
            @test K≈independent atol=3e-14
            @test eigmin(Symmetric(K))>=-1e-13
            @test all(diag(K).≈s.retained_sum/m.volume)
            shifted=xs.+Ref(SVector(.17,.08))
            shiftedF=G.feature_matrix(m,collect(eachindex(s.kind)),shifted)
            @test shiftedF'*Diagonal(s.lambda[s.shell])*shiftedF/m.volume≈K atol=3e-14
            n=2maximum(abs,s.frequencies)+1
            grid=[lower+widths.*SVector(i/n,j/n) for i in 0:n-1 for j in 0:n-1]
            phi=G.feature_matrix(m,collect(eachindex(s.kind)),grid)
            @test phi*phi'/length(grid)≈Matrix{Float64}(I,length(s.kind),length(s.kind)) atol=2e-14
            for t in (SVector(0.,0.),SVector(.03,.11),SVector(.2,.4))
                ratio=kernel(s,t)/s.retained_sum
                @test -1e-14<=1-ratio^2<=1+1e-14
            end
        end
    end

    @testset "raw covariance factory, coordinate permutations and default bytes" begin
        y=[SVector(-2.,1.,.3),SVector(.2,2.,1.1),SVector(1.3,-.8,.9),
            SVector(.7,.5,-1.4),SVector(-.1,1.7,.2)]
        p=Dict{String,Any}("dpp_alpha"=>1.1,"dpp_c0"=>.73,"dpp_nu"=>2.,"dpp_cutoff"=>.008)
        original=G.build_spectrum(Val(3),1.1,.73,2.,.008)
        before=bytes(p)
        @test bytes(R.spectrum_from_prior(y,p,"padded"))==bytes(original)
        @test bytes(p)==before
        p["dpp_metric"]="euclidean";p["dpp_metric_matrix"]="ignored by old path"
        @test bytes(R.spectrum_from_prior(y,p,"padded"))==bytes(original)
        C=[1.2 .2 -.15;.2 .8 .1;-.15 .1 1.1]
        p["dpp_metric"]="mahalanobis";p["dpp_metric_matrix"]=[collect(row) for row in eachrow(C)]
        p["dpp_alpha"]=9.
        for domain in ("padded","data_rectangle")
            lower,upper=R.domain_bounds(y,domain)
            m=Main.Benchmark.model(y,101;spectrum=original,domain)
            @test collect(lower)==collect(m.lower)
            @test collect(upper)==collect(m.upper)
            A=R.metric_from_covariance(C,upper-lower)
            @test A≈Diagonal(1 ./(upper-lower))*C*Diagonal(1 ./(upper-lower)) rtol=2e-15
            direct=R.build_spectrum(Val(3),9.,.73,2.,.008,A)
            before=bytes(p);factory=R.spectrum_from_prior(y,p,domain)
            @test bytes(factory)==bytes(direct)
            @test bytes(p)==before
            perm=[3,1,2];yp=[SVector{3,Float64}(x[perm]) for x in y]
            pp=deepcopy(p);Cp=C[perm,perm]
            pp["dpp_metric_matrix"]=[collect(row) for row in eachrow(Cp)]
            permuted=R.spectrum_from_prior(yp,pp,domain)
            old=signed_spectrum(factory);new=signed_spectrum(permuted)
            @test Set(keys(new))==Set(Tuple(collect(w)[perm]) for w in keys(old))
            for (w,l) in old
                @test new[Tuple(collect(w)[perm])]≈l rtol=3e-14
            end
            @test countlaw(permuted.lambda[permuted.shell])≈countlaw(factory.lambda[factory.shell]) atol=3e-14
        end
        A=[.9 .15 -.2;.15 .8 .1;-.2 .1 1.1]
        before=bytes(A);reference=R.build_spectrum(Val(3),1.1,.73,8.,.008,A)
        rngbefore=copy(Random.default_rng())
        for _ in 1:3
            @test bytes(R.build_spectrum(Val(3),1.1,.73,8.,.008,copy(A)))==bytes(reference)
        end
        @test bytes(A)==before
        @test rand(copy(Random.default_rng()),UInt64)==rand(copy(rngbefore),UInt64)
        restored=deserialize(IOBuffer(bytes(reference)))
        @test bytes(restored)==bytes(reference)
    end

    @testset "honest finite omitted-tail bound" begin
        for A in ([1. .2;.2 .7],[.2 -.13;-.13 .2]),v in (2.,8.)
            a=1.1;c=.73;cutoff=.02
            s=R.build_spectrum(Val(2),a,c,v,cutoff,A)
            bigger=brute(A,a,c,v,1e-14)
            omitted=sum(l for l in values(bigger) if l<cutoff)
            @test isfinite(s.tail_estimate) && s.tail_estimate>0
            @test s.tail_estimate>=omitted
            @test s.tail_estimate==R.omitted_tail_bound(A,a,c,v,cutoff)
            @test R.omitted_tail_bound(A,a,c,v,1e-4)<=s.tail_estimate
        end
    end

    @testset "invalid metric/parameters fail before constructing a target" begin
        I2=Matrix{Float64}(I,2,2)
        for A in ([1. .2;.3 1.],[1. 2.;2. 1.],[1. 0.;0. 0.],fill(NaN,2,2),ones(3,3))
            @test_throws ArgumentError R.build_spectrum(Val(2),1.,.7,2.,.01,A)
        end
        for pars in ((0.,.7,2.,.01),(Inf,.7,2.,.01),(1.,1.,2.,.01),
                (1.,.7,0.,.01),(1.,.7,2.,0.),(1.,.7,2.,.8),(1.,NaN,2.,.01))
            @test_throws ArgumentError R.build_spectrum(Val(2),pars...,I2)
        end
        @test_throws ArgumentError R.build_spectrum(Val(2),1.,.7,2.,.01,I2;max_box_nodes=1)
        @test_throws ArgumentError R.metric_from_covariance(I2,[1.,0.])
        y=[SVector(0.,0.),SVector(1.,2.),SVector(.5,.2)]
        p=Dict{String,Any}("dpp_metric"=>"unknown","dpp_alpha"=>1.,"dpp_c0"=>.7,"dpp_nu"=>2.,"dpp_cutoff"=>.01)
        @test_throws ArgumentError R.spectrum_from_prior(y,p,"padded")
        p["dpp_metric"]="mahalanobis"
        @test_throws ArgumentError R.spectrum_from_prior(y,p,"padded")
        p["dpp_metric_matrix"]=[[1.,0.]]
        @test_throws ArgumentError R.spectrum_from_prior(y,p,"padded")
        p["dpp_metric_matrix"]=[[1.,0.],[0.,1.]]
        @test_throws ArgumentError R.spectrum_from_prior(y,p,"bad domain")
    end
end
end
