mutable struct Streams
    root::UInt64
    obs::Vector{Xoshiro}
    cov::Vector{Xoshiro}
    loc::Vector{Xoshiro}
    accept::Xoshiro
    spectral::Xoshiro
    palm::Xoshiro
    weights::Xoshiro
end
function Streams(seed,n)
    Streams(UInt64(seed),[Xoshiro(seedof(seed,1,i)) for i in 1:n],Xoshiro[],Xoshiro[],
        Xoshiro(seedof(seed,4)),Xoshiro(seedof(seed,5)),Xoshiro(seedof(seed,6)),Xoshiro(seedof(seed,7)))
end
function grow_streams!(s,n)
    for i in length(s.loc)+1:n
        push!(s.loc,Xoshiro(seedof(s.root,2,i))); push!(s.cov,Xoshiro(seedof(s.root,3,i)))
    end
end
