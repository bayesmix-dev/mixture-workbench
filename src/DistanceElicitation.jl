module DistanceElicitation
using Statistics, LinearAlgebra
function distances(y)
    n=length(y); ds=Vector{Float64}(undef,n*(n-1)÷2);p=0
    for i in 2:n, j in 1:i-1
        p+=1;ds[p]=norm(y[i]-y[j])
    end
    ds
end
"Gaussian KDE with Scott's univariate bandwidth, evaluated by linear binning."
function kde(ds;multiplier=1.,bins=16385)
    h=multiplier*std(ds)*length(ds)^(-.2)
    h>0 || error("Degenerate distances")
    grid=collect(range(0,maximum(ds)+4h;length=bins));dx=grid[2]
    counts=zeros(bins)
    for x in ds
        u=x/dx; j=floor(Int,u)+1;f=u-floor(u)
        counts[j]+=1-f;counts[j+1]+=f
    end
    density=zeros(bins); width=ceil(Int,6h/dx)
    kernel=[exp(-.5*(j*dx/h)^2)/(length(ds)*h*sqrt(2pi)) for j in 0:width]
    for i in eachindex(grid), j in max(1,i-width):min(bins,i+width)
        density[i]+=counts[j]*kernel[abs(i-j)+1]
    end
    minima=[i for i in 2:bins-1 if density[i]<density[i-1] && density[i]<=density[i+1]]
    maxima=[i for i in 2:bins-1 if density[i]>density[i-1] && density[i]>=density[i+1]]
    (grid=grid,density=density,bandwidth=h,minima=minima,maxima=maxima,
        radius=isempty(minima) ? NaN : grid[first(minima)])
end
end
