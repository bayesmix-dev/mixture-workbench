module MahalanobisStrauss

using LinearAlgebra
import ..PointProcessSampler as PP

export Prior, distance2, conflicts

"""
Strauss interaction in d_C(x,y)^2 = (x-y)' C^(-1) (x-y).

`beta` retains the existing activity-times-raw-domain-volume convention. The
metric changes the interaction alone; the base measure remains raw volume.
The constructor copies C, validates symmetry/positive definiteness and stores
its Cholesky factor. No existing serialized prior or sampler type is changed.
"""
struct Prior <: PP.PointPrior
    beta::Float64
    radius::Float64
    logalpha::Float64
    C::Matrix{Float64}
    factor::Cholesky{Float64,Matrix{Float64}}
    identity::Bool
    function Prior(beta::Real,radius::Real,logalpha::Real,C::AbstractMatrix)
        b,r,a=Float64.((beta,radius,logalpha))
        isfinite(b) && b>0 || throw(ArgumentError("Strauss beta must be finite and positive"))
        isfinite(r) && r>=0 || throw(ArgumentError("Strauss radius must be finite and nonnegative"))
        isfinite(a) && a<=0 || throw(ArgumentError("Strauss logalpha must be finite and nonpositive"))
        n,m=size(C)
        n==m && n>0 || throw(ArgumentError("The Strauss metric must be a nonempty square matrix"))
        matrix=Matrix{Float64}(C)
        all(isfinite,matrix) && issymmetric(matrix) ||
            throw(ArgumentError("The Strauss metric must be finite and symmetric"))
        factor=try
            cholesky(Symmetric(matrix))
        catch e
            e isa PosDefException || rethrow()
            throw(ArgumentError("The Strauss metric must be positive definite"))
        end
        identity=all(matrix[i,j]==(i==j ? 1. : 0.) for i in 1:n,j in 1:n)
        new(b,r,a,matrix,factor,identity)
    end
end

"""Squared Mahalanobis separation, using a factor solve instead of an inverse."""
function distance2(p::Prior,x,y)
    length(x)==length(y)==size(p.C,1) || throw(DimensionMismatch("Point/metric dimensions differ"))
    delta=x-y
    # This exact path also preserves the original floating-point <= decision
    # for identity metrics, including points precisely at the radius boundary.
    p.identity ? sum(abs2,delta) : sum(abs2,p.factor.L\delta)
end

function conflicts(p::Prior,xs)
    sum((distance2(p,xs[i],xs[j])<=p.radius^2 for i in eachindex(xs) for j in 1:i-1);init=0)
end

function PP.geometry(p::Prior,m,xs)
    size(p.C,1)==length(m.lower) || throw(DimensionMismatch("Model/metric dimensions differ"))
    all(x->length(x)==size(p.C,1) && all(isfinite,x),xs) ||
        throw(ArgumentError("Strauss points must have the metric dimension and finite coordinates"))
    pairs=conflicts(p,xs)
    PP.Geometry(xs,zeros(0,0),zeros(0,0),length(xs)*(log(p.beta)-m.logvolume)+pairs*p.logalpha)
end

function PP.edit(p::Prior,m,g,x,j)
    xs=copy(g.xs)
    x===nothing ? deleteat!(xs,j) : j==0 ? push!(xs,x) : (xs[j]=x)
    PP.geometry(p,m,xs)
end

end
