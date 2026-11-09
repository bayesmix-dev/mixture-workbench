function canonical_labels(c)
    map=Dict{Int,Int}(); out=similar(c)
    for i in eachindex(c); out[i]=get!(map,c[i],length(map)+1); end
    out
end
