module WorkbenchCLI

using GeneralDPPMixtures, Random, LinearAlgebra, StaticArrays, Statistics, SHA, TOML
include("IOProtocol.jl")
using .IOProtocol: validate_keys, read_numeric_csv, csvcell, writerow,
    write_metadata, source_digest, parse_args
include("../execution/Benchmark.jl")
const B = Benchmark
const G = GeneralDPPMixtures
const ROOT = dirname(dirname(@__DIR__))
const METHODS = string.(B.METHODS)
const DEFAULTS = Dict{String,Any}(
    "method"=>"mfm", "iterations"=>1000, "warmup"=>500, "thin"=>5,
    "chains"=>2, "seed"=>123, "initial_clusters"=>2,
    "weight_shape"=>0.5, "covariance_scale"=>1.0,
    "spectrum_target"=>3.0, "spectrum_c0"=>0.9, "spectrum_cutoff"=>0.02,
    "strauss_activity"=>3.0, "strauss_radius"=>0.5,
    "strauss_log_interaction"=>-2.0, "save_allocations"=>true,
    "split_merge"=>false, "betas"=>[1.0], "swap_every"=>5)

function inactive_keys(method)
    spectral = ("spectrum_target", "spectrum_c0", "spectrum_cutoff")
    strauss = ("strauss_activity", "strauss_radius", "strauss_log_interaction")
    method=="mfm" ? ("weight_shape", spectral..., strauss...) : method=="strauss_general" ? spectral : strauss
end

function settings(path)
    supplied = TOML.parsefile(path)
    validate_keys(supplied, keys(DEFAULTS))
    c = merge(copy(DEFAULTS), supplied)
    c["method"] in METHODS || throw(ArgumentError("unsupported method $(c["method"]); supported: $(join(METHODS, ", "))"))
    specific = inactive_keys(c["method"])
    ignored = intersect(collect(keys(supplied)),collect(specific))
    isempty(ignored) || throw(ArgumentError("settings $(join(ignored, ", ")) do not apply to method $(c["method"])"))
    for key in ("iterations", "warmup", "thin", "chains", "seed", "initial_clusters", "swap_every")
        v=c[key]
        v isa Integer && !(v isa Bool) && 0<=v<=typemax(Int) || throw(ArgumentError("$key must be a nonnegative machine integer"))
        c[key]=Int(v)
    end
    for key in ("iterations", "thin", "chains", "initial_clusters", "swap_every")
        c[key]>0 || throw(ArgumentError("$key must be positive"))
    end
    0<=c["warmup"]<c["iterations"] || throw(ArgumentError("require 0 <= warmup < iterations"))
    c["iterations"]-c["warmup"]>=c["thin"] || throw(ArgumentError("thin leaves no retained draws"))
    for key in ("weight_shape", "covariance_scale", "spectrum_target", "spectrum_c0", "spectrum_cutoff", "strauss_activity", "strauss_radius", "strauss_log_interaction")
        v=c[key]
        v isa Real && !(v isa Bool) && isfinite(v) || throw(ArgumentError("$key must be finite and numeric"))
        c[key]=Float64(v)
    end
    for key in ("weight_shape", "covariance_scale", "strauss_activity", "strauss_radius")
        c[key]>0 || throw(ArgumentError("$key must be positive"))
    end
    0<c["spectrum_cutoff"]<c["spectrum_c0"]<1 || throw(ArgumentError("require 0 < spectrum_cutoff < spectrum_c0 < 1"))
    c["spectrum_target"]>c["spectrum_c0"] || throw(ArgumentError("spectrum_target must exceed spectrum_c0"))
    c["strauss_log_interaction"]<=0 || throw(ArgumentError("Strauss log interaction must be nonpositive"))
    for key in ("save_allocations", "split_merge")
        c[key] isa Bool || throw(ArgumentError("$key must be true or false"))
    end
    betas=c["betas"]
    betas isa AbstractVector && !isempty(betas) && all(x->x isa Real && !(x isa Bool) && isfinite(x) && 0<=x<=1,betas) ||
        throw(ArgumentError("betas must be a nonempty numeric array in [0,1]"))
    betas=Float64.(betas)
    betas[1]==1 && all(diff(betas).<0) || throw(ArgumentError("betas must start at 1 and strictly decrease"))
    length(betas)==1 || c["method"] in ("mfm","dpp_marginal") || throw(ArgumentError("likelihood tempering is implemented here only for mfm and dpp_marginal"))
    !c["split_merge"] || c["method"]=="mfm" || throw(ArgumentError("split_merge flag is supported only for mfm; use method=dpp_marginal_sm for the DPP split-merge sampler"))
    c["betas"]=betas
    c
end


function run(config_path, data_path, output)
    c=settings(config_path)
    names,A=read_numeric_csv(data_path)
    n,d=size(A)
    n>max(d,2) || throw(ArgumentError("Gaussian examples require more observations than dimensions and at least three rows"))
    c["initial_clusters"]<=n || throw(ArgumentError("initial_clusters exceeds observations"))
    empirical=cov(A;dims=1)
    isposdef(Symmetric(empirical)) || throw(ArgumentError("empirical covariance must be positive definite; supply nondegenerate data"))
    y=[SVector{d,Float64}(A[i,:]) for i in 1:n]
    method=Symbol(c["method"])
    isdpp=startswith(c["method"],"dpp_")
    # MFM/Strauss reuse Model's Gaussian fields; their targets do not use a DPP spectrum.
    spectrum=isdpp ? G.make_spectrum(d;c0=c["spectrum_c0"],target=c["spectrum_target"],cutoff=c["spectrum_cutoff"]) : G.build_spectrum(Val(d),100.,.9,2.,.5)
    !isdpp || c["initial_clusters"]<=length(spectrum.kind) || throw(ArgumentError("initial_clusters exceeds retained spectral rank"))
    model=B.model(y,UInt64(c["seed"]);spectrum,weight_shape=c["weight_shape"],covariance_scale=c["covariance_scale"])
    out=abspath(output)
    !ispath(out) || isdir(out) && isempty(readdir(out)) || throw(ArgumentError("output must be absent or an empty directory"))
    mkpath(out)
    metadata=Dict{String,Any}("schema_version"=>1,"model"=>c["method"],"seed"=>c["seed"],"chains"=>c["chains"],
        "iterations"=>c["iterations"],"warmup"=>c["warmup"],"thin"=>c["thin"],"n"=>n,"d"=>d,"column_names"=>names,
        "source_digest"=>source_digest(),"input_sha256"=>bytes2hex(sha256(read(data_path))),"julia_version"=>string(VERSION),
        "run_status"=>"running","empirical_bayes"=>true,"prior_covariance_scale"=>c["covariance_scale"],
        "prior_tau"=>model.tau,"prior_omega"=>[collect(row) for row in eachrow(model.omega)],
        "location_lower"=>collect(model.lower),"location_upper"=>collect(model.upper),"betas"=>c["betas"],
        "log_target_definition"=>method in (:dpp_general,:strauss_general) ? "unavailable_NaN" :
            method==:dpp_conditional ? "cold_marginal_density_of_base_state" : "cold_collapsed_target")
    active_config=Dict(k=>v for (k,v) in c if !(k in inactive_keys(c["method"])))
    open(joinpath(out,"config.toml"),"w") do io;TOML.print(io,active_config;sorted=true);end
    write_metadata(out,metadata)
    BLAS.set_num_threads(1)
    draws=open(joinpath(out,"draws.csv"),"w"); diagnostics=open(joinpath(out,"diagnostics.csv"),"w")
    alloc=c["save_allocations"] ? open(joinpath(out,"allocations.csv"),"w") : nothing
    transport=length(c["betas"])>1 ? open(joinpath(out,"transport.csv"),"w") : nothing
    writerow(draws,("chain","iteration","k","M","entropy","log_target"))
    writerow(diagnostics,("chain","metric","attempted","accepted","value","status"))
    alloc===nothing || writerow(alloc,("chain","iteration","observation","cluster"))
    transport===nothing || writerow(transport,("chain","iteration","slot","beta","replica"))
    started=time()
    try
        for chain in 1:c["chains"]
            seed=G.seedof(c["seed"],chain,8001)
            init=if isdpp
                G.starting_configuration(model,c["initial_clusters"],chain)
            else
                labels,centers=G.kmeans_start(Xoshiro(G.seedof(seed,8000)),model.y,c["initial_clusters"])
                G.Initialization(labels,centers,Int[])
            end
            states=[B.setup(model,init,method,G.seedof(seed,r),c["strauss_radius"];strauss_beta=c["strauss_activity"],strauss_logalpha=c["strauss_log_interaction"]) for r in eachindex(c["betas"])]
            rng=Xoshiro(G.seedof(seed,8002)); sm_rng=Xoshiro(G.seedof(seed,8003))
            work=method==:dpp_marginal_sm ? B.SM.Workspace(model) : c["split_merge"] ? B.MFMSequentialSplitMerge.Workspace(model) : nothing
            tags=collect(eachindex(states)); phases=zeros(Int,length(states)); phases[1]=1; roundtrips=zeros(Int,length(states))
            swaps=zeros(Int,2,max(0,length(states)-1)); sm_attempts=0; sm_accepts=0
            for iteration in 1:c["iterations"]
                for r in eachindex(states)
                    s=states[r]; beta=c["betas"][r]
                    if length(states)>1
                        method==:mfm ? B.MFMLikelihoodTempering.hot_sweep!(s,beta) : B.DPPLikelihoodTempering.sweep!(s,beta)
                    elseif method==:dpp_marginal_sm
                        G.marginal_collapsed!(s)
                    else
                        B.step!(s,method,nothing,sm_rng)
                    end
                    proposal=if method==:dpp_marginal_sm
                        B.SM.attempt!(work,s,sm_rng)
                    elseif c["split_merge"]
                        B.MFMSequentialSplitMerge.attempt!(work,s,sm_rng;joint=true,beta)
                    else
                        nothing
                    end
                    if proposal!==nothing
                        sm_attempts+=1;sm_accepts+=Int(proposal.outcome==:accepted)
                    end
                end
                if length(states)>1 && iteration%c["swap_every"]==0
                    for r in 1:length(states)-1
                        ratio=method==:mfm ? B.MFMLikelihoodTempering.swap_logratio(states[r],states[r+1],c["betas"][r],c["betas"][r+1]) : B.DPPLikelihoodTempering.swap_logratio(states[r],states[r+1],c["betas"][r],c["betas"][r+1])
                        swaps[1,r]+=1
                        if log(rand(rng))<min(0.,ratio)
                            states[r],states[r+1]=states[r+1],states[r]
                            tags[r],tags[r+1]=tags[r+1],tags[r]
                            swaps[2,r]+=1
                        end
                        cold=tags[1]; hot=tags[end]
                        phases[cold]==2 && (roundtrips[cold]+=1)
                        phases[cold]=1
                        phases[hot]==1 && (phases[hot]=2)
                    end
                end
                if iteration>c["warmup"] && (iteration-c["warmup"])%c["thin"]==0
                    s=states[1];v=B.scalars(s,method)
                    lp=method==:mfm ? B.MFMTarget.logtarget(s) : method in (:dpp_general,:strauss_general) ? NaN : G.collapsed_logtarget(B.base(s))
                    writerow(draws,(chain,iteration,Int(v[1]),Int(v[2]),v[3],lp))
                    if alloc!==nothing
                        labels=G.canonical_labels(B.base(s).c)
                        for i in eachindex(labels);writerow(alloc,(chain,iteration,i,labels[i]));end
                    end
                    transport===nothing || foreach(r->writerow(transport,(chain,iteration,r,c["betas"][r],tags[r])),eachindex(states))
                end
            end
            writerow(diagnostics,(chain,"split_merge",sm_attempts,sm_accepts,sm_attempts>0 ? sm_accepts/sm_attempts : NaN,sm_attempts>0 ? "observed" : "unassessed"))
            for r in axes(swaps,2)
                a,b=swaps[:,r];writerow(diagnostics,(chain,"swap_edge_$r",a,b,a>0 ? b/a : NaN,a>0 ? "observed" : "unassessed"))
            end
            length(states)>1 && writerow(diagnostics,(chain,"roundtrips",0,0,sum(roundtrips),"observed"))
            println("Completed chain $chain/$(c["chains"])")
        end
        for io in (draws,diagnostics,alloc,transport);io===nothing || flush(io);end
        metadata["run_status"]="completed";metadata["elapsed_seconds"]=time()-started
        write_metadata(out,metadata)
    catch err
        metadata["run_status"]="failed";metadata["error"]=sprint(showerror,err)
        write_metadata(out,metadata)
        rethrow()
    finally
        close(draws);close(diagnostics);alloc===nothing || close(alloc);transport===nothing || close(transport)
    end
    out
end

function main(args=ARGS)
    p=parse_args(args)
    run(p["--config"],p["--data"],p["--output"])
end
end
