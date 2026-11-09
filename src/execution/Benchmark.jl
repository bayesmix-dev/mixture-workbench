module Benchmark
using GeneralDPPMixtures, LinearAlgebra, StaticArrays, Statistics, Random, Serialization, SHA, TOML, Printf
const G=GeneralDPPMixtures
include("../models/point_process/PointProcessSampler.jl")
include("../priors/point_process/MahalanobisRepulsion.jl")
include("../priors/point_process/MahalanobisStrauss.jl")
include("../models/mfm/MFMBaseline.jl")
include("../calibration/Elicitation.jl")
include("../../vendor/projection_DPP/src/DPPSplitMerge.jl")
include("../models/point_process/algorithms/DPPReallocation.jl")
include("../models/point_process/algorithms/DPPMultiReallocation.jl")
include("../models/point_process/algorithms/DPPCollapsedReallocation.jl")
include("../models/point_process/algorithms/DPPSequentialSplitMerge.jl")
include("../models/point_process/algorithms/DPPGlobalSplitMerge.jl")
include("../models/point_process/algorithms/DPPLikelihoodTempering.jl")
include("../models/point_process/algorithms/DPPPriorRefresh.jl")
include("../models/point_process/algorithms/DPPEnsembleReferences.jl")
include("../models/point_process/algorithms/DPPFixedCount.jl")
include("../models/mfm/MFMTarget.jl")
include("../models/mfm/algorithms/MFMHyperparameters.jl")
include("../models/mfm/algorithms/MFMBlockMoves.jl")
include("../models/mfm/algorithms/MFMSequentialSplitMerge.jl")
include("../models/mfm/algorithms/MFMConcentrationSplitMerge.jl")
include("../models/mfm/algorithms/MFMLikelihoodTempering.jl")
include("../models/mfm/algorithms/MFMReferenceIndependence.jl")
const PP=PointProcessSampler
const MF=MFMBaseline
const SM=DPPSplitMerge
const SM_LOCK=ReentrantLock() # protects the rare global BigFloat precision fallback
const METHODS=[:mfm,:dpp_conditional,:dpp_marginal,:dpp_marginal_sm,:dpp_general,:strauss_general]
const ROOT=dirname(dirname(@__DIR__))
const DATA_DIR=get(ENV,"REPULSIVE_DATA_DIR",joinpath(ROOT,"data"))

spectrum_from_prior(y,p,domain=get(p,"domain","padded")) =
    MahalanobisRepulsion.spectrum_from_prior(y,p,domain)

function data(sample;limit=0,markers=["CD3","CD4","CD8b","CD8"])
    lines=readlines(joinpath(DATA_DIR,sample*".csv"))
    first(lines)=="CD3,CD4,CD8b,CD8" || error("unexpected markers")
    columns=split(first(lines),','); indices=[findfirst(==(s),columns) for s in markers]
    all(!isnothing,indices) && length(unique(markers))==length(markers) || error("invalid markers")
    y=[SVector{length(markers),Float64}(parse.(Float64,split(l,','))[indices]) for l in lines[2:end]]
    length(y)==(sample=="positive" ? 1809 : 6809) || error("incorrect sample size")
    limit>0 ? y[1:min(limit,length(y))] : y
end
function model(y,seed; spectrum=G.make_spectrum(length(first(y))),weight_shape=.1,covariance_scale=1.,domain="padded")
    d=length(first(y));center=mean(y); zeta=map(e->maximum(abs(x[e]-center[e]) for x in y),1:d)
    weight_shape>0 && covariance_scale>0 || error("invalid mixture prior")
    omega=covariance_scale*cov(reduce(hcat,y);dims=2)
    # The same raw-scale domain and IW prior are used by ALL repulsive samplers.
    domain in ("padded","data_rectangle") || error("unknown location domain")
    lower=domain=="padded" ? center-2.5zeta : [minimum(x[j] for x in y) for j in 1:d]
    upper=domain=="padded" ? center+2.5zeta : [maximum(x[j] for x in y) for j in 1:d]
    G.Model(y,ones(Int,length(y)),lower,upper,spectrum;
        a=weight_shape,tau=Float64(d+2),omega=omega,seed=UInt64(seed))
end
function source_digest()
    files=sort(vcat([joinpath(dir,f) for (dir,_,files) in walkdir(joinpath(ROOT,"src")) for f in files if endswith(f,".jl")],
        [joinpath(ROOT,"vendor/projection_DPP/src/DPPSplitMerge.jl"),joinpath(ROOT,"Project.toml"),joinpath(ROOT,"Manifest.toml")],
        [joinpath(ROOT,"scripts",f) for f in readdir(joinpath(ROOT,"scripts")) if endswith(f,".jl")]))
    bytes2hex(sha256(join([relpath(f,ROOT)*":"*bytes2hex(sha256(read(f))) for f in files],"\n")))
end
function setup(m,init,method,seed,radius;strauss_beta=10.,strauss_logalpha=-length(m.y)/20,strauss_metric_matrix=nothing)
    method==:mfm && return MF.initialize(m,init,seed)
    method==:dpp_general && return PP.initialize(m,init,PP.DPPPrior(m),seed)
    if method==:strauss_general
        p=isnothing(strauss_metric_matrix) ? PP.StraussPrior(strauss_beta,radius,strauss_logalpha) :
            MahalanobisStrauss.Prior(strauss_beta,radius,strauss_logalpha,strauss_metric_matrix)
        return PP.initialize(m,init,p,seed)
    end
    G.initialize(m,init,method==:dpp_conditional ? :conditional_notes : :marginal_collapsed,seed)
end
function step!(s,method,work,rng)
    if method==:mfm; MF.sweep!(s)
    elseif method in (:dpp_general,:strauss_general); PP.sweep!(s)
    elseif method==:dpp_marginal_sm
        G.marginal_collapsed!(s)
        lock(SM_LOCK) do; SM.attempt!(work,s,rng); end
    else; G.sweep!(s)
    end
end
base(s)=s
base(s::G.ConditionalState)=s.base
function scalars(s,method)
    b=base(s); ns=[c.n for c in b.clusters if c.n>0]; n=length(b.c)
    M=method==:mfm ? s.M : method in (:dpp_general,:strauss_general) ? length(s.clusters) : length(b.S)
    entropy=-sum(v->(v/n)*log(v/n),ns)
    wss=sum(tr(c.scatter) for c in b.clusters if c.n>0)
    Float64[length(ns),M,entropy,sum(v->v*(v-1),ns)/(n*(n-1)),wss,
        method==:mfm ? s.gamma : b.model.a,method==:mfm ? s.lambda : NaN]
end
const NAMES=["k","M","entropy","coclustering","withinss","gamma","Lambda"]
function predictive(s,method,indices,rng)
    b=base(s); m=b.model; n=length(b.c); d=length(first(m.y))
    if method==:mfm
        s=deepcopy(s) # a diagnostic must not change slot allocation in the sampler
        active=s.active; k=length(active); empty=MF.empty_slot!(s)
        return [begin
            vals=[log(s.clusters[h].n+s.gamma)+MF.logpred(s,h,m.y[i]) for h in active]
            s.M>k && push!(vals,log((s.M-k)*s.gamma)+MF.logpred(s,empty,m.y[i]))
            PP.logsumexp(vals)-log(n+s.M*s.gamma)
        end for i in indices]
    end
    active=findall(c->c.n>0,b.clusters); xs=[b.clusters[h].x for h in active]
    if method in (:dpp_general,:strauss_general)
        append!(xs,[s.geom.xs[h] for h in eachindex(b.clusters) if b.clusters[h].n==0])
    elseif method==:dpp_conditional
        append!(xs,s.unused)
    else
        xs=G.palm_sample(rng,m,b.S,xs)
    end
    k=length(active); M=length(xs); Ls=typeof(m.omegaL)[]; dfs=Float64[]; weights=Float64[]
    for j in 1:M
        count=j<=k ? b.clusters[active[j]].n : 0
        scale=if j<=k
            c=b.clusters[active[j]]; delta=c.mean-xs[j]
            m.omega+c.scatter+count*(delta*delta')
        else; m.omega
        end
        push!(Ls,cholesky(Symmetric(scale)).L); push!(dfs,m.tau+count-d+1)
        push!(weights,log(count+m.a)-log(n+M*m.a))
    end
    [PP.logsumexp([weights[j]+G.student_logdensity(m.y[i],xs[j],Ls[j],dfs[j]) for j in 1:M]) for i in indices]
end
function writecsv(path,names,A)
    open(path,"w") do io
        println(io,join(names,','))
        for row in eachrow(A); println(io,join(row,',')); end
    end
end
function atomicserialize(path,x)
    serialize(path*".tmp",x); mv(path*".tmp",path;force=true)
end
include("../execution/CovarianceExchange.jl")
function runjob(m,init,elic,method,chain,out,iterations,burnin,digest;checkpoint_every=250,ladder=Float64[],targetscale=1.)
    dir=joinpath(out,string(method),"chain$chain"); mkpath(dir)
    spec=(method=method,chain=chain,iterations=iterations,burnin=burnin,source=digest,
        datahash=bytes2hex(sha256(reinterpret(UInt8,reduce(hcat,m.y)[:]))),radius=elic.radius,
        prior=(m.spectrum.alpha,m.spectrum.c0,m.spectrum.nu,m.spectrum.cutoff,
            get(elic,:strauss_beta,10.),get(elic,:strauss_logalpha,-length(m.y)/20),m.a,m.tau,Tuple(m.omega)),
        domain=(Tuple(m.lower),Tuple(m.upper)))
    isempty(ladder) || (spec=merge(spec,(covariance_ladder=ladder,targetscale=targetscale,exchange_every=5)))
    metric=get(elic,:strauss_metric_matrix,nothing)
    if method==:strauss_general && !isnothing(metric)
        isempty(ladder) || error("Mahalanobis Strauss covariance exchange requires an explicitly calibrated metric at every rung")
        spec=merge(spec,(strauss_metric_matrix=Tuple(vec(metric)),))
    end
    final=joinpath(dir,"result.jls"); checkpoint=joinpath(dir,"checkpoint.jls")
    if isfile(final)
        r=deserialize(final); r.spec==spec || error("completed run configuration mismatch: $dir")
        println("SKIP $dir"); return
    end
    seed=G.seedof(20261106,m.seed,findfirst(==(method),METHODS),chain)
    if isfile(checkpoint)
        r=deserialize(checkpoint); r.spec==spec || error("checkpoint configuration mismatch: $dir")
        s=r.state; rng=r.sm_rng; trace=r.trace; partitions=r.partitions; saved=r.saved
        start=r.iteration+1; seconds=r.seconds; smstats=r.smstats
        predictions=r.predictions; diagnostic_seconds=r.diagnostic_seconds
        exchange=isempty(ladder) ? nothing : r.exchange
    else
        s=setup(m,init,method,seed,elic.radius;strauss_beta=spec.prior[5],strauss_logalpha=spec.prior[6],strauss_metric_matrix=metric); rng=Xoshiro(G.seedof(seed,77))
        trace=fill(NaN,iterations,length(NAMES)); partitions=Vector{Vector{Int}}(); saved=Int[]
        start=1; seconds=zeros(2); smstats=zeros(Int,4)
        predictions=Vector{Vector{Float64}}(); diagnostic_seconds=0.
        exchange=isempty(ladder) ? nothing : initialize_exchange(m,init,elic,method,seed,ladder,targetscale)
        isnothing(exchange) || (s=exchange.states[exchange.target])
    end
    work=method==:dpp_marginal_sm ? SM.Workspace(m) : nothing
    eval_indices=sort(randperm(Xoshiro(G.seedof(m.seed,991)),length(m.y))[1:min(128,length(m.y))])
    # Warm a disposable copy, never the production state or RNG; excluded from timers.
    step!(deepcopy(s),method,deepcopy(work),deepcopy(rng))
    for it in start:iterations
        elapsed=@elapsed begin
            if !isnothing(exchange)
                s=exchange_step!(exchange,method,it)
                smstats=exchange.smstats
            elseif method==:dpp_marginal_sm
                G.marginal_collapsed!(s)
                p=lock(SM_LOCK) do; SM.attempt!(work,s,rng); end
                if p!==nothing
                    j=p.kind==:split ? 1 : 2; smstats[j]+=1
                    p.outcome==:accepted && (smstats[j+2]+=1)
                end
            else
                step!(s,method,work,rng)
            end
        end
        seconds[it<=burnin ? 1 : 2]+=elapsed
        trace[it,:]=scalars(s,method)
        if it>burnin && (it-burnin)%25==0
            push!(partitions,G.canonical_labels(base(s).c)); push!(saved,it)
            diagnostic_seconds+=@elapsed push!(predictions,predictive(s,method,eval_indices,Xoshiro(G.seedof(seed,992,it))))
        end
        if it%checkpoint_every==0 || it==iterations
            record=(spec=spec,state=s,sm_rng=rng,trace=trace,partitions=partitions,saved=saved,
                iteration=it,seconds=seconds,smstats=smstats,predictions=predictions,diagnostic_seconds=diagnostic_seconds,exchange=exchange)
            atomicserialize(checkpoint,record)
            println("$(basename(out)) $method chain=$chain iteration=$it/$iterations k=$(Int(trace[it,1])) seconds=$(round(sum(seconds);digits=1))"); flush(stdout)
        end
    end
    counters=method==:mfm ? Dict("gamma_attempts"=>s.gamma_attempts,"gamma_accepts"=>s.gamma_accepts) :
        method in (:dpp_general,:strauss_general) ? Dict("birth_attempts"=>s.attempts[1],"birth_accepts"=>s.accepts[1],
            "death_attempts"=>s.attempts[2],"death_accepts"=>s.accepts[2],"location_attempts"=>s.attempts[3],"location_accepts"=>s.accepts[3]) :
        Dict(string(f)=>getfield(base(s).counters,f) for f in fieldnames(G.Counters))
    counters["split_attempts"]=smstats[1]; counters["merge_attempts"]=smstats[2]
    counters["split_accepts"]=smstats[3]; counters["merge_accepts"]=smstats[4]
    if !isnothing(exchange)
        open(joinpath(dir,"exchange.toml"),"w") do io
            TOML.print(io,Dict("ladder"=>ladder,"targetscale"=>targetscale,"attempts"=>exchange.attempts,
                "accepts"=>exchange.accepts,"target_visits_by_replica_tag"=>exchange.target_visits,
                "auxiliary_update_every"=>5,"initialization"=>"target 4 or 15; auxiliary counts decrease from 15 to 2 across ladder",
                "split_merge_counter_scope"=>"all replicas"))
        end
    end
    result=(spec=spec,trace=trace,partitions=partitions,saved=saved,seconds=seconds,counters=counters,
        predictions=predictions,eval_indices=eval_indices,diagnostic_seconds=diagnostic_seconds)
    writecsv(joinpath(dir,"trace.csv"),vcat("iteration",NAMES),hcat(1:iterations,trace))
    writecsv(joinpath(dir,"partitions.csv"),vcat("iteration",["observation_$i" for i in eachindex(m.y)]),
        hcat(saved,permutedims(reduce(hcat,partitions))))
    writecsv(joinpath(dir,"logpredictive.csv"),vcat("iteration",["observation_$i" for i in eval_indices]),
        hcat(saved,permutedims(reduce(hcat,predictions))))
    open(joinpath(dir,"timing.toml"),"w") do io
        TOML.print(io,Dict("burnin_seconds"=>seconds[1],"retained_seconds"=>seconds[2],
            "predictive_diagnostic_seconds"=>diagnostic_seconds,"seed"=>string(seed),"counters"=>counters))
    end
    atomicserialize(final,result)
end

function main(args=ARGS)
    opts=Dict{String,String}()
    iseven(length(args)) || error("options require values")
    for i in 1:2:length(args); opts[args[i]]=args[i+1]; end
    allowed=["--iterations","--burnin","--output","--samples","--methods","--limit","--workers","--chains","--prior-config","--markers","--covariance-ladder"]
    all(k->k in allowed,keys(opts)) || error("unknown option")
    iterations=parse(Int,get(opts,"--iterations","50000")); burnin=parse(Int,get(opts,"--burnin","25000"))
    0<=burnin<iterations && iterations-burnin>=25 || error("need at least 25 retained sweeps")
    limit=parse(Int,get(opts,"--limit","0")); workers=parse(Int,get(opts,"--workers","2"))
    1<=workers<=Threads.nthreads() || error("workers exceed Julia threads")
    chains=parse(Int,get(opts,"--chains","2")); chains==2 || error("benchmark requires two chains")
    methods=Symbol.(split(get(opts,"--methods",join(METHODS,',')),',')); all(in(METHODS),methods) || error("unknown method")
    samples=split(get(opts,"--samples","positive,control"),','); all(in(["positive","control"]),samples) || error("unknown sample")
    out=abspath(get(opts,"--output",joinpath(ROOT,"results","benchmark"))); mkpath(out)
    priorfile=get(opts,"--prior-config","")
    priors=isempty(priorfile) ? nothing : TOML.parsefile(priorfile)
    if !isnothing(priors) && :strauss_general in methods && !get(priors,"strauss_calibration_selected",true)
        error("This prior file contains unselected Strauss parameters; use a separately verified Strauss calibration")
    end
    ladder=haskey(opts,"--covariance-ladder") ? parse.(Float64,split(opts["--covariance-ladder"],',')) : Float64[]
    isempty(ladder) || (length(ladder)>1 && issorted(ladder) && all(>(0),ladder) && length(unique(ladder))==length(ladder)) || error("invalid ladder")
    isempty(ladder) || all(in([:mfm,:dpp_marginal_sm,:strauss_general,:dpp_general]),methods) || error("unsupported exchange method")
    priorhash=isempty(priorfile) ? "default" : bytes2hex(sha256(read(priorfile)))
    markers=split(get(opts,"--markers","CD3,CD4,CD8b,CD8"),',');d=length(markers)
    isnothing(priors) || get(priors,"markers",["CD3","CD4","CD8b","CD8"])==markers || error("calibration markers do not match likelihood")
    planpath=joinpath(out,"run.toml")
    if isfile(planpath)
        previous=TOML.parsefile(planpath)
        for (key,value) in (("iterations",iterations),("burnin",burnin),("limit",limit),("workers",workers))
            get(previous,key,nothing)==value || error("run setting changed: $key; use a new output directory")
        end
        get(previous,"prior_config_sha256","default")==priorhash || error("prior settings changed; use a new output directory")
        get(previous,"markers",["CD3","CD4","CD8b","CD8"])==markers || error("markers changed; use a new output directory")
        get(previous,"covariance_ladder",Float64[])==ladder || error("ladder changed; use a new output directory")
    end
    complete=joinpath(out,"COMPLETE")
    isfile(complete) && rm(complete)
    BLAS.set_num_threads(1); digest=source_digest()
    open(planpath,"w") do io
        TOML.print(io,Dict("methods"=>string.(methods),"samples"=>samples,"chains"=>chains,
            "iterations"=>iterations,"burnin"=>burnin,"limit"=>limit,"source_sha256"=>digest,
            "workers"=>workers,"cpu"=>Sys.CPU_NAME,"machine"=>Sys.MACHINE,"started_unix"=>time(),
            "prior_config_sha256"=>priorhash,"markers"=>markers,"covariance_ladder"=>ladder))
    end
    isnothing(priors) || cp(priorfile,joinpath(out,"prior_config.toml");force=true)
    jobs=Tuple[]
    for (sid,sample) in enumerate(samples)
        y=data(sample;limit,markers)
        spectrum_params=isnothing(priors) ? nothing : merge(priors,get(priors,sample,Dict{String,Any}()))
        spectrum=isnothing(priors) ? G.make_spectrum(d) : spectrum_from_prior(y,spectrum_params,get(priors,"domain","padded"))
        weight_shape=isnothing(priors) ? .1 : get(priors,"weight_shape",.1)
        covariance_scale=isnothing(priors) ? 1. : get(priors,"covariance_scale",1.)
        isempty(ladder) || covariance_scale in ladder || error("target scale absent from ladder")
        domain=isnothing(priors) ? "padded" : get(priors,"domain","padded")
        m=model(y,sample=="positive" ? 101 : 102;spectrum,weight_shape,covariance_scale,domain)
        elic=Elicitation.radius(y); dir=joinpath(out,sample); mkpath(dir)
        if !isnothing(priors)
            p=priors[sample]
            elic=merge(elic,(radius=p["radius"],strauss_beta=p["strauss_beta"],strauss_logalpha=p["strauss_logalpha"]))
            if haskey(p,"strauss_metric_matrix")
                metric=reduce(vcat,permutedims.(p["strauss_metric_matrix"]))
                elic=merge(elic,(strauss_metric_matrix=metric,))
            end
        end
        init=[G.starting_configuration(m,k,c) for (c,k) in enumerate((4,min(15,length(y))))]
        metadata=Dict("n"=>length(y),"markers"=>markers,"scale"=>"original",
            "source_sha256"=>digest,"julia"=>string(VERSION),"iterations"=>iterations,"burnin"=>burnin,"chains"=>chains,
            "lower"=>collect(m.lower),"upper"=>collect(m.upper),"weight_shape"=>m.a,"IW_df"=>m.tau,"covariance_scale_multiplier"=>covariance_scale,
            "IW_scale_rows"=>[collect(m.omega[i,:]) for i in 1:d],"spectrum_mean"=>m.spectrum.retained_sum,
            "spectrum_cutoff"=>m.spectrum.cutoff,"spectrum_modes"=>length(m.spectrum.kind),
            "strauss_beta"=>get(elic,:strauss_beta,10.),"strauss_logalpha"=>get(elic,:strauss_logalpha,-length(y)/20),"radius"=>elic.radius,
            "spectrum_alpha"=>spectrum.alpha,"spectrum_c0"=>spectrum.c0,"spectrum_nu"=>spectrum.nu,
            "spectrum_conditional_mean"=>spectrum.conditional_mean,"prior_config_sha256"=>priorhash,
            "radius_kstar"=>elic.kstar,"radius_kmax"=>elic.kmax,"radius_minfraction"=>elic.minfraction,
            "radius_seed"=>elic.seed,"radius_starts"=>elic.starts,"workers"=>workers,"blas_threads"=>1,
            "data_sha256"=>bytes2hex(sha256(read(joinpath(DATA_DIR,sample*".csv")))))
        open(joinpath(dir,"model.toml"),"w") do io; TOML.print(io,metadata); end
        writecsv(joinpath(dir,"radius.csv"),["k","minimum_separation","retained_clusters","withinss"],
            hcat(elic.ks,elic.dmin,elic.retained,elic.objectives))
        for c in 1:chains
            writecsv(joinpath(dir,"initialization_$c.csv"),["observation","cluster"],hcat(1:length(y),init[c].c))
        end
        for method in methods, chain in 1:chains
            push!(jobs,(m,init[chain],elic,method,chain,dir,iterations,burnin,digest))
        end
    end
    queue=Channel{Tuple}(length(jobs)); foreach(j->put!(queue,j),jobs); close(queue)
    @sync for _ in 1:workers
        Threads.@spawn for job in queue
            runjob(job...;ladder,targetscale=isnothing(priors) ? 1. : get(priors,"covariance_scale",1.))
        end
    end
    open(joinpath(out,"COMPLETE"),"w") do io; println(io,"$(length(jobs)) jobs completed; source=$digest"); end
end
end
