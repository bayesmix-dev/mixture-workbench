function run_chain(s,iterations,burnin,ex=Execution();storage_interval=25)
    0<=burnin<iterations || throw(ArgumentError("need 0 <= burnin < iterations"))
    storage_interval>=1 || throw(ArgumentError("storage interval must be positive"))
    names=scalar_names(); traces=zeros(Float64,iterations,length(names))
    partitions=Vector{Vector{Int}}(); saved_iterations=Int[]
    burn_seconds=0.0; retained_seconds=0.0; burn_bytes=0; retained_bytes=0; gcseconds=0.0; diagnostic_seconds=0.0
    t0=time_ns()
    for it in 1:iterations
        timed=@timed sweep!(s,ex)
        if it<=burnin
            burn_seconds+=timed.time; burn_bytes+=timed.bytes
        else
            retained_seconds+=timed.time; retained_bytes+=timed.bytes
        end
        gcseconds+=timed.gctime
        diagtime=@elapsed begin
            traces[it,:].=scalar_summary(s)
            if it%storage_interval==0
                push!(partitions,canonical_labels(core(s).c)); push!(saved_iterations,it)
            end
        end
        diagnostic_seconds+=diagtime
    end
    (names=names,traces=traces,partitions=partitions,saved_iterations=saved_iterations,
        burn_seconds=burn_seconds,retained_seconds=retained_seconds,burn_bytes=burn_bytes,retained_bytes=retained_bytes,
        gc_seconds=gcseconds,diagnostic_seconds=diagnostic_seconds,chain_wall_seconds=(time_ns()-t0)/1e9,
        counters=Dict(string(f)=>getfield(core(s).counters,f) for f in fieldnames(Counters)),
        final_c=canonical_labels(core(s).c),final_S=copy(core(s).S))
end
