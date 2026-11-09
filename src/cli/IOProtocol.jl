module IOProtocol

using SHA, TOML

const ROOT = dirname(dirname(@__DIR__))

function validate_keys(config, allowed)
    unknown = sort!(collect(setdiff(keys(config), allowed)))
    isempty(unknown) || throw(ArgumentError("unknown settings: $(join(unknown, ", ")); this program runs only implemented models and settings"))
end


"""Read an unquoted numeric CSV with one header row. Missing/nonfinite cells fail."""
function read_numeric_csv(path)
    lines=readlines(path)
    length(lines)>=2 || throw(ArgumentError("data CSV needs a header and observations"))
    names=strip.(split(replace(lines[1], '\ufeff'=>""),','))
    !isempty(names) && all(!isempty,names) && length(unique(names))==length(names) || throw(ArgumentError("CSV column names must be nonempty and unique"))
    rows=Vector{Vector{Float64}}()
    for (i,line) in enumerate(lines[2:end])
        fields=split(strip(line),',')
        length(fields)==length(names) || throw(ArgumentError("wrong number of CSV fields on line $(i+1)"))
        values=tryparse.(Float64,fields)
        all(x->x!==nothing && isfinite(x),values) || throw(ArgumentError("missing, nonnumeric or nonfinite CSV value on line $(i+1)"))
        push!(rows,Float64.(values))
    end
    names,reduce(vcat,permutedims.(rows))
end

csvcell(x) = begin
    s=string(x)
    occursin(r"[,\"\n\r]",s) ? "\""*replace(s,"\""=>"\"\"")*"\"" : s
end
writerow(io, values)=println(io,join(csvcell.(collect(values)),','))
function write_metadata(out, metadata)
    open(joinpath(out,"metadata.toml.tmp"),"w") do io; TOML.print(io,metadata;sorted=true); end
    open(joinpath(out,"metadata.csv.tmp"),"w") do io
        writerow(io,("key","value"))
        for key in sort!(collect(keys(metadata))); writerow(io,(key,metadata[key])); end
    end
    mv(joinpath(out,"metadata.toml.tmp"),joinpath(out,"metadata.toml");force=true)
    mv(joinpath(out,"metadata.csv.tmp"),joinpath(out,"metadata.csv");force=true)
end
"""Hash all Julia source recursively, plus the pinned dependencies and vendored move.

`root` may select another repository tree when checking provenance tooling.
"""
function source_digest(root::AbstractString=ROOT)
    root=abspath(root)
    files=String[]
    for (directory, _, names) in walkdir(joinpath(root,"src"))
        append!(files,[joinpath(directory,name) for name in names if endswith(name,".jl")])
    end
    append!(files,[joinpath(root,"vendor/projection_DPP/src/DPPSplitMerge.jl"),
        joinpath(root,"Project.toml"),joinpath(root,"Manifest.toml")])
    relative(p)=replace(relpath(p,root),'\\'=>'/')
    sort!(files;by=relative)
    bytes2hex(sha256(join([relative(p)*":"*bytes2hex(sha256(read(p))) for p in files],"\n")))
end

function parse_args(args)
    length(args)==6 || throw(ArgumentError("usage: run.jl --config FILE.toml --data FILE.csv --output DIRECTORY"))
    parsed=Dict{String,String}()
    for i in 1:2:length(args)
        args[i] in ("--config","--data","--output") || throw(ArgumentError("unknown argument $(args[i])"))
        haskey(parsed,args[i]) && throw(ArgumentError("duplicate argument $(args[i])"))
        parsed[args[i]]=args[i+1]
    end
    length(parsed)==3 || throw(ArgumentError("config, data and output are required"))
    parsed
end

end
