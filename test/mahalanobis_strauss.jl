# Standalone: julia --project=. test/mahalanobis_strauss.jl
if !isdefined(Main,:Benchmark);include("../src/execution/Benchmark.jl");end
if !isdefined(Main.Benchmark,:MahalanobisStrauss)
    Base.include(Main.Benchmark,joinpath(@__DIR__,"../src/priors/point_process/MahalanobisStrauss.jl"))
end
module MahalanobisStraussTests
using Test, GeneralDPPMixtures, LinearAlgebra, StaticArrays, Random, Statistics, Distributions, Serialization
const G=GeneralDPPMixtures
const PP=Main.Benchmark.PointProcessSampler
const MS=Main.Benchmark.MahalanobisStrauss

function fixture(d=2)
    y=[SVector{d,Float64}(fill(x,d)) for x in (-.4,.2,.7)]
    G.Model(y,ones(Int,3),fill(-2.,d),fill(3.,d),G.make_spectrum(d);
        a=.37,tau=d+2.,omega=Matrix{Float64}(I,d,d),seed=UInt64(17))
end
function bytes(x)
    io=IOBuffer();serialize(io,x);take!(io)
end

@testset "Optional Mahalanobis Strauss geometry" begin
    @testset "Identity preserves original geometry and every edit exactly" begin
        rng=Xoshiro(2341)
        for d in (1,2,3,5),M in 0:8,radius in (0.,.7,2.)
            m=fixture(d);old=PP.StraussPrior(2.7,radius,-3.4)
            metric=MS.Prior(old.beta,old.radius,old.logalpha,Matrix{Float64}(I,d,d))
            xs=[SVector{d,Float64}(randn(rng,d)) for _ in 1:M]
            a=PP.geometry(old,m,xs);b=PP.geometry(metric,m,xs)
            @test a.lp===b.lp
            @test a.xs==b.xs && a.F==b.F && a.K==b.K
            choices=Any[(SVector{d,Float64}(randn(rng,d)),0)]
            M>0 && append!(choices,[(nothing,1),(SVector{d,Float64}(randn(rng,d)),M)])
            before=bytes(b)
            for (x,j) in choices
                aa=PP.edit(old,m,a,x,j);bb=PP.edit(metric,m,b,x,j)
                @test aa.lp===bb.lp
                @test aa.xs==bb.xs && aa.F==bb.F && aa.K==bb.K
                @test bytes(b)==before
            end
        end
        # Exact boundary and coincident-point decisions use the old arithmetic.
        for radius in (0.,1.,sqrt(2.))
            m=fixture(2);xs=[SVector(0.,0.),SVector(radius,0.),SVector(0.,0.)]
            @test PP.geometry(MS.Prior(3.,radius,-2.,Matrix{Float64}(I,2,2)),m,xs).lp===
                PP.geometry(PP.StraussPrior(3.,radius,-2.),m,xs).lp
        end
    end

    @testset "Metric scale, rotation and whitening" begin
        C=[2. .7 -.2;.7 1.4 .3;-.2 .3 1.1]
        p=MS.Prior(3.2,1.1,-2.7,C);m=fixture(3)
        xs=[SVector(-1.2,.2,.8),SVector(.1,-.7,.3),SVector(.3,.6,-.8),SVector(.4,.2,.1)]
        for t in (.25,2.,9.),i in eachindex(xs),j in eachindex(xs)
            pt=MS.Prior(p.beta,p.radius,p.logalpha,t*C)
            @test MS.distance2(pt,xs[i],xs[j])≈MS.distance2(p,xs[i],xs[j])/t atol=2e-14
        end
        for t in (.25,2.,9.)
            iso=MS.Prior(2.3,.8,-1.1,t*Matrix{Float64}(I,3,3))
            raw=PP.StraussPrior(iso.beta,sqrt(t)*iso.radius,iso.logalpha)
            @test PP.geometry(iso,m,xs).lp==PP.geometry(raw,m,xs).lp
        end
        Q=Matrix(qr([1. .2 -.3;.4 1. .2;-.1 .5 1.]).Q)
        rotated=MS.Prior(p.beta,p.radius,p.logalpha,Symmetric(Q*C*Q'))
        rx=[SVector{3,Float64}(Q*x) for x in xs]
        white=[SVector{3,Float64}(cholesky(Symmetric(C)).L\x) for x in xs]
        raw=PP.StraussPrior(p.beta,p.radius,p.logalpha)
        for i in eachindex(xs),j in eachindex(xs)
            @test MS.distance2(rotated,rx[i],rx[j])≈MS.distance2(p,xs[i],xs[j]) atol=2e-14
            @test MS.distance2(p,xs[i],xs[j])≈sum(abs2,white[i]-white[j]) atol=2e-14
        end
        @test PP.geometry(p,m,xs).lp==PP.geometry(rotated,m,rx).lp==PP.geometry(raw,m,white).lp
        # Model carries raw volume only: rotating the interaction adds no Jacobian.
        shifted=[x+SVector(3.,-7.,2.) for x in xs]
        @test PP.geometry(p,m,shifted).lp==PP.geometry(p,m,xs).lp
    end

    @testset "Independent dense quadratic-form and centre-density calculation" begin
        rng=Xoshiro(12306)
        for d in (1,2,3,5),M in 0:7,trial in 1:4
            A=randn(rng,d,d);C=A*A'+.3Matrix{Float64}(I,d,d)
            beta=.3+rand(rng)*8;radius=.2+rand(rng)*2;logalpha=-.1-rand(rng)*9
            p=MS.Prior(beta,radius,logalpha,C);m=fixture(d)
            xs=[SVector{d,Float64}(randn(rng,d)) for _ in 1:M]
            inverse=inv(C);pairs=0;normal=MvNormal(zeros(d),C)
            for i in eachindex(xs),j in 1:i-1
                delta=Vector(xs[i]-xs[j]);quadratic=dot(delta,inverse*delta)
                @test MS.distance2(p,xs[i],xs[j])≈quadratic rtol=3e-13 atol=3e-13
                @test quadratic≈-2*(logpdf(normal,delta)-logpdf(normal,zeros(d))) rtol=3e-13 atol=3e-13
                pairs+=quadratic<=radius^2
            end
            independent=M*log(beta/exp(m.logvolume))+pairs*logalpha
            @test PP.geometry(p,m,xs).lp≈independent atol=2e-13
            @test MS.conflicts(p,xs)==pairs
            # Check each edit directly from the dense inverse, not geometry reuse.
            for (x,j) in ((SVector{d,Float64}(randn(rng,d)),0),)
                edited=PP.edit(p,m,PP.geometry(p,m,xs),x,j)
                dense_pairs=sum((dot(Vector(edited.xs[a]-edited.xs[b]),inverse*Vector(edited.xs[a]-edited.xs[b]))<=radius^2
                    for a in eachindex(edited.xs) for b in 1:a-1);init=0)
                @test edited.lp≈length(edited.xs)*(log(beta)-m.logvolume)+dense_pairs*logalpha atol=2e-13
            end
        end
    end

    @testset "Unchanged activity convention and exact Poisson tilt" begin
        # At alpha=1 the raw-volume integral of each ordered k-point density,
        # divided by k!, is beta^k/k! for every C. No determinant factor enters.
        for d in (1,3),beta in (.3,4.2),u in (.2,3.),k in 0:12
            m=fixture(d);C=3Matrix{Float64}(I,d,d)
            p=MS.Prior(beta,10.,0.,C);old=PP.StraussPrior(beta,10.,0.)
            xs=[SVector{d,Float64}(fill(i/(k+2),d)) for i in 1:k]
            new=PP.edit(p,m,PP.geometry(p,m,xs),zero(first(m.y)),0)
            psi=(1+u)^(-m.a)
            lr=new.lp-PP.geometry(p,m,xs).lp+log(psi)+m.logvolume-log(k+1)
            @test lr≈logpdf(Poisson(beta*psi),k+1)-logpdf(Poisson(beta*psi),k) atol=3e-14
            @test PP.geometry(p,m,xs).lp===PP.geometry(old,m,xs).lp
        end
    end

    @testset "Input validation, copied metric and serialization compatibility" begin
        C=[2. .3;.3 1.]
        p=MS.Prior(2.,1.,-3.,C);expected=copy(p.C);factor=copy(p.factor.L)
        C[1,1]=100.;C[1,2]=100.
        @test p.C==expected && p.factor.L==factor
        @test Matrix(p.factor.L)*Matrix(p.factor.L)'≈p.C
        @test deserialize(IOBuffer(bytes(p))).C==p.C
        old=PP.StraussPrior(2.,1.,-3.);@test deserialize(IOBuffer(bytes(old)))==old
        @test fieldnames(PP.StraussPrior)==(:beta,:radius,:logalpha)
        for Cbad in (zeros(0,0),ones(2,3),[1. .1;0. 1.],[1. 2.;2. 1.],
                [1. 0.;0. 0.],[NaN 0.;0. 1.],[1. 0.;0. Inf])
            @test_throws ArgumentError MS.Prior(2.,1.,-3.,Cbad)
        end
        for args in ((0.,1.,-3.),(Inf,1.,-3.),(2.,-1.,-3.),(2.,Inf,-3.),
                (2.,1.,.1),(2.,1.,NaN),(2.,1.,-Inf))
            @test_throws ArgumentError MS.Prior(args...,Matrix{Float64}(I,2,2))
        end
        @test_throws DimensionMismatch MS.distance2(p,[0.],[1.])
        @test_throws DimensionMismatch PP.geometry(p,fixture(1),SVector{1,Float64}[])
        @test_throws ArgumentError PP.geometry(p,fixture(2),[SVector(0.,NaN)])
        @test_throws ArgumentError PP.geometry(p,fixture(2),[SVector(0.)])
    end
end
end
