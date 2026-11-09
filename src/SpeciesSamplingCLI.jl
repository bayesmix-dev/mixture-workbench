module SpeciesSamplingCLI

using GeneralDPPMixtures, TOML, SHA
using GeneralDPPMixtures.SpeciesSampling
include("WorkbenchCLI.jl")
const IOProtocol = WorkbenchCLI
const S = SpeciesSampling
const G = GeneralDPPMixtures

function integer_setting(value, name; minimum=1)
    value isa Integer && !(value isa Bool) && minimum <= value <= typemax(Int) ||
        throw(ArgumentError("$name must be an integer >= $minimum"))
    Int(value)
end
function real_setting(value, name)
    value isa Real && !(value isa Bool) && isfinite(value) && isfinite(Float64(value)) ||
        throw(ArgumentError("$name must be a finite real number"))
    Float64(value)
end

const CONFIG_KEYS = Set(["method", "strength", "discount", "observation_sd", "base_mean",
    "base_sd", "initial_clusters", "iterations", "warmup", "thin", "chains", "seed",
    "save_allocations"])

function configuration(raw::AbstractDict, y)
    unknown = setdiff(Set(keys(raw)), CONFIG_KEYS)
    isempty(unknown) || throw(ArgumentError("unsupported species-sampling settings: $(join(sort!(collect(unknown)), ", "))"))
    get(raw, "method", nothing) == "pitman_yor" ||
        throw(ArgumentError("method must be 'pitman_yor' for this entrypoint"))
    alpha = real_setting(get(raw, "strength", 1.0), "strength")
    d = real_setting(get(raw, "discount", 0.3), "discount")
    prior = S.PitmanYor(alpha, d)
    model = S.GaussianLocation(y;
        observation_sd=real_setting(get(raw,"observation_sd",1.0), "observation_sd"),
        base_mean=real_setting(get(raw,"base_mean",0.0), "base_mean"),
        base_sd=real_setting(get(raw,"base_sd",2.0), "base_sd"))
    iterations = integer_setting(get(raw,"iterations",4000), "iterations")
    warmup = integer_setting(get(raw,"warmup",1000), "warmup"; minimum=0)
    thin = integer_setting(get(raw,"thin",5), "thin")
    warmup < iterations || throw(ArgumentError("warmup must be smaller than iterations"))
    iterations-warmup >= thin || throw(ArgumentError("settings must retain at least one draw"))
    initial = integer_setting(get(raw,"initial_clusters",1), "initial_clusters")
    initial <= length(y) || throw(ArgumentError("initial_clusters must not exceed the observation count"))
    chains = integer_setting(get(raw,"chains",4), "chains")
    seed = integer_setting(get(raw,"seed",123), "seed"; minimum=0)
    save = get(raw,"save_allocations",true)
    save isa Bool || throw(ArgumentError("save_allocations must be a Boolean"))
    resolved = Dict{String,Any}("method"=>"pitman_yor", "strength"=>alpha, "discount"=>d,
        "observation_sd"=>model.observation_sd, "base_mean"=>model.base_mean, "base_sd"=>model.base_sd,
        "initial_clusters"=>initial, "iterations"=>iterations, "warmup"=>warmup, "thin"=>thin,
        "chains"=>chains, "seed"=>seed, "save_allocations"=>save)
    model, prior, resolved
end

function run(config_path, data_path, output)
    names, A = IOProtocol.read_numeric_csv(data_path)
    size(A,2)==1 || throw(ArgumentError("species-sampling example requires exactly one observation column"))
    model, prior, c = configuration(TOML.parsefile(config_path), vec(A))
    out = abspath(output)
    !ispath(out) || isdir(out) && isempty(readdir(out)) ||
        throw(ArgumentError("output must be absent or an empty directory"))
    S.initialize(model, prior, G.seedof(c["seed"],1,9201);initial_clusters=c["initial_clusters"])
    mkpath(out)
    open(joinpath(out,"config.toml"),"w") do io; TOML.print(io,c;sorted=true); end
    metadata = Dict{String,Any}("schema_version"=>1, "model"=>"pitman_yor_gaussian_location",
        "sampler"=>"collapsed_gibbs", "strength"=>S.strength(prior), "discount"=>S.discount(prior),
        "observation_sd"=>model.observation_sd, "base_mean"=>model.base_mean, "base_sd"=>model.base_sd,
        "n"=>length(model.y), "d"=>1, "column_names"=>names, "chains"=>c["chains"],
        "iterations"=>c["iterations"], "warmup"=>c["warmup"], "thin"=>c["thin"], "seed"=>c["seed"],
        "source_digest"=>IOProtocol.source_digest(),
        "input_sha256"=>bytes2hex(sha256(read(data_path))), "config_sha256"=>bytes2hex(sha256(read(config_path))),
        "julia_version"=>string(VERSION), "run_status"=>"running", "empirical_bayes"=>false,
        "log_target_definition"=>"partition_eppf_times_integrated_gaussian_likelihood",
        "allocation_labels"=>"canonical_by_first_appearance",
        "M_definition"=>"represented_occupied_atoms_equals_k", "total_components"=>"infinite",
        "counter_scope"=>"warmup_and_sampling", "mixing_assessment"=>"unassessed")
    IOProtocol.write_metadata(out,metadata)
    draws = open(joinpath(out,"draws.csv"),"w")
    diagnostics = open(joinpath(out,"diagnostics.csv"),"w")
    allocations = c["save_allocations"] ? open(joinpath(out,"allocations.csv"),"w") : nothing
    IOProtocol.writerow(draws,("chain","iteration","k","M","entropy","log_target"))
    IOProtocol.writerow(diagnostics,("chain","metric","attempted","accepted","value","status"))
    allocations===nothing || IOProtocol.writerow(allocations,("chain","iteration","observation","cluster"))
    started = time()
    try
        for chain in 1:c["chains"]
            s = S.initialize(model,prior,G.seedof(c["seed"],chain,9201);initial_clusters=c["initial_clusters"])
            for iteration in 1:c["iterations"]
                S.collapsed_sweep!(s)
                if iteration>c["warmup"] && (iteration-c["warmup"])%c["thin"]==0
                    labels = S.canonical_labels(s.z)
                    counts = S.cluster_counts(labels)
                    entropy = -sum(nh/length(labels)*log(nh/length(labels)) for nh in counts)
                    lp = S.collapsed_logtarget(s)
                    isfinite(lp) || error("sampled state has a nonfinite target")
                    IOProtocol.writerow(draws,(chain,iteration,length(counts),length(counts),entropy,lp))
                    if allocations!==nothing
                        for i in eachindex(labels)
                            IOProtocol.writerow(allocations,(chain,iteration,i,labels[i]))
                        end
                    end
                end
            end
            IOProtocol.writerow(diagnostics,(chain,"allocation_gibbs",length(model.y)*s.sweeps,
                length(model.y)*s.sweeps,1.0,"exact_conditional"))
            IOProtocol.writerow(diagnostics,(chain,"mixing",0,0,NaN,"unassessed"))
            println("Completed Pitman-Yor chain $chain/$(c["chains"])")
        end
        flush(draws); flush(diagnostics)
        allocations===nothing || flush(allocations)
        metadata["run_status"]="completed"
        metadata["elapsed_seconds"]=time()-started
        IOProtocol.write_metadata(out,metadata)
    catch error
        metadata["run_status"]="failed"
        metadata["error"]=sprint(showerror,error)
        IOProtocol.write_metadata(out,metadata)
        rethrow()
    finally
        close(draws); close(diagnostics)
        allocations===nothing || close(allocations)
    end
    out
end

function main(args=ARGS)
    parsed=IOProtocol.parse_args(args)
    run(parsed["--config"],parsed["--data"],parsed["--output"])
end
end
