module ReplicaWorkerPool

using Random,Serialization
import Base: map,close
export Pool,map,close

struct Completion
    position::Int
    value::Any
    errors::Vector{CapturedException}
end
struct WorkItem{F,I}
    f::F
    index::I
    position::Int
    replies::Channel{Completion}
end
struct JobError <: Exception
    position::Int
    index::Any
    cause::CapturedException
end
function Base.showerror(io::IO,e::JobError)
    print(io,"Replica worker job ",e.position," (index ",repr(e.index),") failed: ")
    showerror(io,e.cause)
end
bytes(x)=(io=IOBuffer();serialize(io,x);take!(io))

function run_job(job::WorkItem)
    before=bytes(copy(Random.default_rng()));errors=CapturedException[];value=nothing
    try
        # Workers outlive callbacks created later during setup or later maps.
        # Enter their current method world only at this outer job boundary.
        value=Base.invokelatest(job.f,job.index)
    catch error
        push!(errors,CapturedException(error,catch_backtrace()))
    end
    if bytes(copy(Random.default_rng()))!=before
        push!(errors,CapturedException(ErrorException("job consumed the worker task-default RNG (implicit draws or child-task creation)"),backtrace()))
    end
    # The per-batch reply buffer holds EVERY issued job. A parent can submit
    # its full batch before collecting, even when some jobs have failed.
    put!(job.replies,Completion(job.position,value,errors))
    nothing
end
function worker_loop(queue)
    for job in queue;run_job(job);end
    nothing
end

"""Persistent bounded workers; construct during setup before parent RNG guards.

No task is created by map. Jobs must use explicit RNGs, private mutable state
and serial inner kernels: child-task creation also advances TaskLocalRNG and
is detected. One caller owns a pool at a time, including close. The pool is a
runtime resource and belongs outside ensembles/checkpoints. Caller code must
bound the sum of worker counts across concurrent ensembles/processes.
"""
mutable struct Pool
    queue::Channel{Any}
    workers::Vector{Task}
    lifecycle_lock::ReentrantLock
    busy::Bool
    closed::Bool
end
function Pool(nworkers::Integer)
    1<=nworkers<=Threads.nthreads(:default) || throw(ArgumentError("worker count must lie in 1:Threads.nthreads(:default)"))
    queue=Channel{Any}(Int(nworkers));workers=Task[]
    try
        for _ in 1:nworkers;push!(workers,Threads.@spawn worker_loop(queue));end
    catch
        close(queue)
        foreach(wait,workers)
        rethrow()
    end
    Pool(queue,workers,ReentrantLock(),false,false)
end
function acquire!(pool)
    lock(pool.lifecycle_lock) do
        pool.closed && throw(InvalidStateException("replica worker pool is closed",:closed))
        pool.busy && throw(ArgumentError("replica worker pool has a caller already; map/close are nonreentrant"))
        pool.busy=true
    end
    nothing
end
function release!(pool)
    lock(pool.lifecycle_lock) do;pool.busy=false;end
    nothing
end

"""Run f(index) once per input, returning results in input order.

All issued jobs finish before this call returns or throws. Kernel exceptions
and worker TaskLocalRNG violations are reported together as CompositeException
with input-position/index context. The pool stays open and reusable afterward;
user state modified by a failing job is NOT rolled back, so an application run
must abort/discard its failed ensemble before any swaps or checkpoints.
Reusing the pool never restores or reseeds any RNG. No per-batch tasks spawn.
"""
function map(f,pool::Pool,indices)
    acquire!(pool)
    try
        items=collect(indices);n=length(items);n==0 && return Any[]
        replies=Channel{Completion}(n);values=Vector{Any}(undef,n)
        errors=Any[];issued=0;received=0
        try
            for (position,index) in enumerate(items)
                put!(pool.queue,WorkItem(f,index,position,replies));issued+=1
            end
        catch error
            push!(errors,CapturedException(error,catch_backtrace()))
        end
        try
            while received<issued
                result=take!(replies);received+=1
                values[result.position]=result.value
                for error in result.errors
                    push!(errors,JobError(result.position,items[result.position],error))
                end
            end
        finally
            # An interrupted collection still drains the already-issued batch.
            # This is the barrier required before caller-side state exchanges.
            while received<issued;take!(replies);received+=1;end
        end
        isempty(errors) || throw(CompositeException(errors))
        values
    finally
        release!(pool)
    end
end

"""Stop and join every persistent worker. Idempotent; rejects an active map."""
function close(pool::Pool)
    lock(pool.lifecycle_lock) do
        pool.busy && throw(ArgumentError("cannot close a replica worker pool during an active map"))
        if !pool.closed;pool.closed=true;close(pool.queue);end
    end
    foreach(fetch,pool.workers)
    nothing
end

end
