module SpeciesSamplingCLI

using GeneralDPPMixtures, TOML, SHA
using GeneralDPPMixtures.SpeciesSampling
include("IOProtocol.jl")
const S = SpeciesSampling
const G = GeneralDPPMixtures
const N = GeneralDPPMixtures.NealSamplers
const E = GeneralDPPMixtures.EfficientSlice

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

const RUN_KEYS = Set(["method", "strength", "base_mean", "base_sd", "initial_clusters",
    "iterations", "warmup", "thin", "chains", "seed", "save_allocations"])

function configuration(raw::AbstractDict, y; family=:pitman_yor)
    method = get(raw,"method",nothing)
    if family == :pitman_yor
        method == "pitman_yor" || throw(ArgumentError("method must be 'pitman_yor' for this entrypoint"))
        likelihood = "gaussian_location"
        allowed = union(RUN_KEYS,Set(["discount","observation_sd"]))
    elseif family == :slice
        method == "slice_efficient" || throw(ArgumentError("method must be 'slice_efficient' for this entrypoint"))
        likelihood = get(raw,"likelihood","gaussian_location")
        likelihood == "gaussian_location" || throw(ArgumentError("efficient slice currently requires gaussian_location"))
        allowed = union(RUN_KEYS,Set(["likelihood","observation_sd"]))
    elseif family == :dirichlet_process
        method in ("neal2","neal3","neal8") ||
            throw(ArgumentError("method must be 'neal2', 'neal3' or 'neal8' for this entrypoint"))
        likelihood = get(raw,"likelihood","gaussian_location")
        likelihood in ("gaussian_location","poisson_lognormal") ||
            throw(ArgumentError("likelihood must be 'gaussian_location' or 'poisson_lognormal'"))
        likelihood == "poisson_lognormal" && method != "neal8" &&
            throw(ArgumentError("Poisson-lognormal observations require Neal 8; Neal 2/3 need implemented conjugate operations"))
        allowed = union(RUN_KEYS,Set(["likelihood"]))
        push!(allowed,likelihood == "gaussian_location" ? "observation_sd" : "proposal_sd")
        method == "neal8" && push!(allowed,"auxiliaries")
    else
        throw(ArgumentError("unsupported concrete entrypoint family"))
    end
    unknown = setdiff(Set(keys(raw)),allowed)
    isempty(unknown) || throw(ArgumentError("unsupported species-sampling settings: $(join(sort!(collect(unknown)), ", "))"))
    alpha = real_setting(get(raw,"strength",1.0), "strength")
    prior = family == :pitman_yor ? S.PitmanYor(alpha,
        real_setting(get(raw,"discount",0.3),"discount")) : S.DirichletProcess(alpha)
    mu = real_setting(get(raw,"base_mean",0.0),"base_mean")
    sd = real_setting(get(raw,"base_sd",2.0),"base_sd")
    model = likelihood == "gaussian_location" ? S.GaussianLocation(y;
        observation_sd=real_setting(get(raw,"observation_sd",1.0),"observation_sd"),base_mean=mu,base_sd=sd) :
        N.PoissonLogRate(y;base_mean=mu,base_sd=sd,
            proposal_sd=real_setting(get(raw,"proposal_sd",0.4),"proposal_sd"))
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
    resolved = Dict{String,Any}("method"=>method, "strength"=>alpha,
        "base_mean"=>mu,"base_sd"=>sd,"initial_clusters"=>initial,
        "iterations"=>iterations,"warmup"=>warmup,"thin"=>thin,"chains"=>chains,
        "seed"=>seed,"save_allocations"=>save)
    if family == :pitman_yor
        resolved["discount"] = S.discount(prior)
    else
        resolved["likelihood"] = likelihood
    end
    resolved[likelihood == "gaussian_location" ? "observation_sd" : "proposal_sd"] =
        likelihood == "gaussian_location" ? model.observation_sd : model.proposal_sd
    method == "neal8" && (resolved["auxiliaries"] = integer_setting(get(raw,"auxiliaries",3),"auxiliaries"))
    model, prior, resolved
end

function initial_state(model,prior,c,chain)
    seed=G.seedof(c["seed"],chain,9201)
    if c["method"] == "slice_efficient"
        E.initialize(model,prior,seed;initial_labels=[mod1(i,c["initial_clusters"]) for i in eachindex(model.y)])
    elseif c["method"] in ("pitman_yor","neal3")
        S.initialize(model,prior,seed;initial_clusters=c["initial_clusters"])
    else
        N.initialize(model,prior,seed;initial_clusters=c["initial_clusters"])
    end
end

function sampler_sweep!(s,c)
    method=c["method"]
    if method == "slice_efficient"
        E.sweep!(s)
    elseif method == "pitman_yor"
        S.collapsed_sweep!(s)
    elseif method == "neal2"
        N.neal2_sweep!(s)
    elseif method == "neal3"
        N.neal3_sweep!(s)
    elseif method == "neal8"
        N.neal8_sweep!(s;auxiliaries=c["auxiliaries"])
    else
        throw(ArgumentError("unsupported implemented sampler"))
    end
end

function run(config_path, data_path, output;family=:pitman_yor)
    names, A = IOProtocol.read_numeric_csv(data_path)
    size(A,2)==1 || throw(ArgumentError("species-sampling example requires exactly one observation column"))
    model, prior, c = configuration(TOML.parsefile(config_path), vec(A);family)
    out = abspath(output)
    !ispath(out) || isdir(out) && isempty(readdir(out)) ||
        throw(ArgumentError("output must be absent or an empty directory"))
    initial_state(model,prior,c,1)
    mkpath(out)
    open(joinpath(out,"config.toml"),"w") do io; TOML.print(io,c;sorted=true); end
    slice = c["method"] == "slice_efficient"
    retained = c["method"] in ("neal2","neal8") || slice
    gaussian = model isa S.GaussianLocation
    modelname = family == :pitman_yor ? "pitman_yor_gaussian_location" :
        gaussian ? "dirichlet_process_gaussian_location" : "dirichlet_process_poisson_lognormal"
    metadata = Dict{String,Any}("schema_version"=>1, "model"=>modelname,
        "sampler"=>c["method"], "strength"=>S.strength(prior), "discount"=>S.discount(prior),
        "base_mean"=>model.base_mean, "base_sd"=>model.base_sd,
        "n"=>length(model.y), "d"=>1, "column_names"=>names, "chains"=>c["chains"],
        "iterations"=>c["iterations"], "warmup"=>c["warmup"], "thin"=>c["thin"], "seed"=>c["seed"],
        "source_digest"=>IOProtocol.source_digest(),
        "input_sha256"=>bytes2hex(sha256(read(data_path))), "config_sha256"=>bytes2hex(sha256(read(config_path))),
        "julia_version"=>string(VERSION), "run_status"=>"running", "empirical_bayes"=>false,
        "log_target_definition"=>(retained && !slice ? "partition_eppf_times_occupied_base_and_likelihood" :
            "partition_eppf_times_integrated_gaussian_likelihood"),
        "allocation_labels"=>"canonical_by_first_appearance",
        "M_definition"=>(slice ? "materialized_ordered_gem_prefix_including_empty_atoms" :
            "represented_occupied_atoms_equals_k"), "total_components"=>"infinite",
        "counter_scope"=>"warmup_and_sampling", "mixing_assessment"=>"unassessed")
    metadata[gaussian ? "observation_sd" : "proposal_sd"] = gaussian ? model.observation_sd : model.proposal_sd
    metadata["parameter_name"] = gaussian ? "mean" : "log_rate"
    retained && gaussian && !slice && (metadata["partition_log_target_definition"] = "partition_eppf_times_integrated_gaussian_likelihood")
    if slice
        metadata["internal_labels"]="ordered_gem_indices_with_gaps"
        metadata["log_target_is_projection"]=true
        metadata["ordered_log_target_definition"]="ordered_gem_labels_and_occupied_atoms_sticks_slices_empty_atoms_integrated"
        metadata["tail_stopping"]="residual_mass_below_minimum_slice_no_fixed_truncation"
    end
    haskey(c,"auxiliaries") && (metadata["auxiliaries"] = c["auxiliaries"])
    IOProtocol.write_metadata(out,metadata)
    draws = open(joinpath(out,"draws.csv"),"w")
    diagnostics = open(joinpath(out,"diagnostics.csv"),"w")
    allocations = c["save_allocations"] ? open(joinpath(out,"allocations.csv"),"w") : nothing
    parameters = retained ? open(joinpath(out,"parameters.csv"),"w") : nothing
    headers=String["chain","iteration","k","M","entropy","log_target"]
    retained && gaussian && !slice && push!(headers,"partition_log_target")
    slice && push!(headers,"ordered_log_target")
    IOProtocol.writerow(draws,headers)
    parameter_headers=String["chain","iteration","cluster","theta"]
    slice && push!(parameter_headers,"gem_index")
    parameters===nothing || IOProtocol.writerow(parameters,parameter_headers)
    IOProtocol.writerow(diagnostics,("chain","metric","attempted","accepted","value","status"))
    allocations===nothing || IOProtocol.writerow(allocations,("chain","iteration","observation","cluster"))
    started = time()
    try
        for chain in 1:c["chains"]
            s = initial_state(model,prior,c,chain)
            for iteration in 1:c["iterations"]
                sampler_sweep!(s,c)
                if iteration>c["warmup"] && (iteration-c["warmup"])%c["thin"]==0
                    labels = S.canonical_labels(s.z)
                    counts = S.cluster_counts(labels)
                    entropy = -sum(nh/length(labels)*log(nh/length(labels)) for nh in counts)
                    lp = slice ? E.marginal_logtarget(s) : retained ? N.joint_logtarget(s) : S.collapsed_logtarget(s)
                    isfinite(lp) || error("sampled state has a nonfinite target")
                    row=Any[chain,iteration,length(counts),slice ? E.materialized_count(s) : length(counts),entropy,lp]
                    retained && gaussian && !slice && push!(row,S.collapsed_logtarget(s))
                    slice && push!(row,E.ordered_logtarget(s))
                    IOProtocol.writerow(draws,row)
                    if parameters!==nothing
                        if slice
                            for (h,gem) in enumerate(unique(s.z))
                                IOProtocol.writerow(parameters,(chain,iteration,h,s.theta[gem],gem))
                            end
                        else
                            for h in eachindex(s.theta)
                                IOProtocol.writerow(parameters,(chain,iteration,h,s.theta[h]))
                            end
                        end
                    end
                    if allocations!==nothing
                        for i in eachindex(labels)
                            IOProtocol.writerow(allocations,(chain,iteration,i,labels[i]))
                        end
                    end
                end
            end
            updates=slice ? s.allocation_updates : length(model.y)*s.sweeps
            IOProtocol.writerow(diagnostics,(chain,"allocation_gibbs",updates,updates,1.0,"exact_conditional"))
            if slice
                IOProtocol.writerow(diagnostics,(chain,"slice_stick_draws",s.stick_draws,s.stick_draws,
                    s.stick_draws,"observed"))
            elseif retained
                IOProtocol.writerow(diagnostics,(chain,"atom_mh",s.atom_attempts,s.atom_accepts,
                    s.atom_attempts>0 ? s.atom_accepts/s.atom_attempts : NaN,
                    gaussian ? "not_required_exact_conditional" : "observed"))
            end
            IOProtocol.writerow(diagnostics,(chain,"mixing",0,0,NaN,"unassessed"))
            println("Completed $(c["method"]) chain $chain/$(c["chains"])")
        end
        flush(draws); flush(diagnostics)
        parameters===nothing || flush(parameters)
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
        parameters===nothing || close(parameters)
        allocations===nothing || close(allocations)
    end
    out
end

function main(args=ARGS;family=:pitman_yor)
    parsed=IOProtocol.parse_args(args)
    run(parsed["--config"],parsed["--data"],parsed["--output"];family)
end
end
