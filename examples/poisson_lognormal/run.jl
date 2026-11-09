#!/usr/bin/env julia
if abspath(PROGRAM_FILE)==(@__FILE__) && ARGS==["--help"]
    println("Fixed-component Poisson mixture with Gaussian log-rate base measure")
    println("Usage: julia --project=. examples/poisson_lognormal/run.jl --config FILE.toml --data COUNTS.csv --output DIRECTORY")
    println("The CSV must have one numeric count column and a header. Output must be absent or empty.")
    exit(0)
end

module PoissonLognormalCLI
using GeneralDPPMixtures, TOML, SHA
include(joinpath(@__DIR__,"..","..","src","WorkbenchCLI.jl"))
include("PoissonLognormal.jl")
const IOProtocol=WorkbenchCLI
const P=PoissonLognormal
const G=GeneralDPPMixtures

function run(config_path, data_path, output)
    names,A=IOProtocol.read_numeric_csv(data_path)
    size(A,2)==1 || throw(ArgumentError("Poisson-lognormal data must have exactly one count column"))
    raw=TOML.parsefile(config_path)
    m,c=P.configuration(raw,vec(A))
    out=abspath(output)
    !ispath(out) || isdir(out) && isempty(readdir(out)) ||
        throw(ArgumentError("output must be absent or an empty directory"))
    # Validate the initial target before creating output files.
    P.initialize(m,G.seedof(c.seed,1,9101))
    mkpath(out)
    effective=Dict{String,Any}("method"=>"poisson_lognormal","components"=>m.M,
        "weight_shape"=>m.gamma,"base_mean"=>m.mu,"base_sd"=>m.sd,"proposal_sd"=>m.proposal_sd,
        "iterations"=>c.iterations,"warmup"=>c.warmup,"thin"=>c.thin,"chains"=>c.chains,
        "seed"=>Int(c.seed),"save_allocations"=>c.save_allocations)
    open(joinpath(out,"config.toml"),"w") do io;TOML.print(io,effective;sorted=true);end
    extension_files=["PoissonLognormal.jl","run.jl"]
    extension_digest=bytes2hex(sha256(join([file*":"*bytes2hex(sha256(read(joinpath(@__DIR__,file)))) for file in extension_files],"\n")))
    metadata=Dict{String,Any}("schema_version"=>1,"model"=>"poisson_lognormal_fixed_finite",
        "components"=>m.M,"weight_shape"=>m.gamma,"base_mean"=>m.mu,"base_sd"=>m.sd,
        "proposal_sd"=>m.proposal_sd,"n"=>length(m.y),"d"=>1,"column_names"=>names,
        "chains"=>c.chains,"iterations"=>c.iterations,"warmup"=>c.warmup,"thin"=>c.thin,"seed"=>Int(c.seed),
        "source_digest"=>IOProtocol.source_digest(),"extension_source_digest"=>extension_digest,
        "input_sha256"=>bytes2hex(sha256(read(data_path))),"config_sha256"=>bytes2hex(sha256(read(config_path))),
        "julia_version"=>string(VERSION),"run_status"=>"running","empirical_bayes"=>false,
        "log_target_definition"=>"labelled_z_all_M_log_rates_weights_integrated_normalized_prior_and_likelihood",
        "allocation_labels"=>"canonical_by_first_appearance_with_matching_eta_columns_empty_components_last",
        "parameter_columns"=>["eta_$h" for h in 1:m.M],"counter_scope"=>"warmup_and_sampling",
        "mixing_assessment"=>"unassessed")
    IOProtocol.write_metadata(out,metadata)
    draws=open(joinpath(out,"draws.csv"),"w")
    diagnostics=open(joinpath(out,"diagnostics.csv"),"w")
    allocations=c.save_allocations ? open(joinpath(out,"allocations.csv"),"w") : nothing
    IOProtocol.writerow(draws,vcat(["chain","iteration","k","M","entropy","log_target"],["eta_$h" for h in 1:m.M]))
    IOProtocol.writerow(diagnostics,("chain","metric","attempted","accepted","value","status"))
    allocations===nothing || IOProtocol.writerow(allocations,("chain","iteration","observation","cluster"))
    started=time()
    try
        for chain in 1:c.chains
            s=P.initialize(m,G.seedof(c.seed,chain,9101))
            for iteration in 1:c.iterations
                P.sweep!(s)
                if iteration>c.warmup && (iteration-c.warmup)%c.thin==0
                    labels,eta=P.canonical_output(s)
                    entropy=-sum(nh>0 ? nh/length(m.y)*log(nh/length(m.y)) : 0.0 for nh in s.n)
                    lp=P.logtarget(s)
                    isfinite(lp) || error("sampled state has a nonfinite target")
                    IOProtocol.writerow(draws,vcat(Any[chain,iteration,length(unique(labels)),m.M,entropy,lp],eta))
                    if allocations!==nothing
                        for i in eachindex(labels)
                            IOProtocol.writerow(allocations,(chain,iteration,i,labels[i]))
                        end
                    end
                end
            end
            IOProtocol.writerow(diagnostics,(chain,"log_rate_mh",s.attempted,s.accepted,
                s.attempted>0 ? s.accepted/s.attempted : NaN,"observed"))
            IOProtocol.writerow(diagnostics,(chain,"empty_component_prior_refresh",s.empty_refreshes,
                s.empty_refreshes,s.empty_refreshes,"observed"))
            IOProtocol.writerow(diagnostics,(chain,"mixing",0,0,NaN,"unassessed"))
            println("Completed Poisson-lognormal chain $chain/$(c.chains)")
        end
        # Publish completion only after every result table is readable.
        flush(draws);flush(diagnostics)
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
        close(draws);close(diagnostics)
        allocations===nothing || close(allocations)
    end
    out
end

function main(args=ARGS)
    parsed=IOProtocol.parse_args(args)
    run(parsed["--config"],parsed["--data"],parsed["--output"])
end
end

if abspath(PROGRAM_FILE)==(@__FILE__)
    try
        PoissonLognormalCLI.main()
    catch error
        println(stderr,"Poisson-lognormal example: ",sprint(showerror,error))
        exit(1)
    end
end
