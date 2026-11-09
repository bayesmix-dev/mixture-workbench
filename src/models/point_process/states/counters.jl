Base.@kwdef mutable struct Counters
    births::Int=0
    deletions::Int=0
    loc_attempts::Int=0
    loc_accepts::Int=0
    loc_outside::Int=0
    spectral_added::Int=0
    spectral_removed::Int=0
    new_attempts::Int=0
    new_accepts::Int=0
    new_outside::Int=0
    new_rejects::Int=0
    palm_attempts::Int=0
    unused_occupied::Int=0
    near_rank::Int=0
    clamps::Int=0
    projector_rebuilds::Int=0
end
