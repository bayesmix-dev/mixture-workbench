"""Deterministic-seeded k-means++ with Lloyd updates and empty-cluster repair."""
function kmeans_start(rng,y,k)
    k<=length(y) || throw(ArgumentError("k exceeds data size"))
    centers=[y[rand(rng,eachindex(y))]]
    for _ in 2:k
        w=[minimum(sum(abs2,v-x) for x in centers) for v in y]
        push!(centers,y[draw_weights(rng,w)])
    end
    c=zeros(Int,length(y))
    for _ in 1:100
        old=copy(c)
        for i in eachindex(y); c[i]=argmin([sum(abs2,y[i]-x) for x in centers]); end
        for h in 1:k
            inds=findall(==(h),c)
            if isempty(inds)
                counts=[count(==(j),c) for j in 1:k]
                i=argmax([counts[c[j]]>1 ? sum(abs2,y[j]-centers[c[j]]) : -Inf for j in eachindex(y)])
                c[i]=h; inds=[i]
            end
            centers[h]=mean(view(y,inds))
        end
        c==old && break
    end
    c,centers
end
