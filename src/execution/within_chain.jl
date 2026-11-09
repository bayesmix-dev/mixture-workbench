struct Execution
    mode::Symbol
    threads::Int
    function Execution(mode=:serial,threads=1;budget=Sys.CPU_THREADS)
        mode in (:serial,:chains,:within) || throw(ArgumentError("unknown parallel mode"))
        1<=threads<=min(Threads.nthreads(:default),budget) || throw(ArgumentError("requested concurrency exceeds pool/CPU budget"))
        mode!=:within && threads!=1 && throw(ArgumentError("internally serial mode requires threads=1"))
        new(mode,threads)
    end
end
function foreach_chunk(f,n,ex::Execution)
    if ex.threads==1
        f(1:n)
    else
        # Fixed item streams preserve draw order regardless of chunk assignment.
        nt=min(ex.threads,n)
        @sync for t in 1:nt
            r=fld((t-1)*n,nt)+1:fld(t*n,nt)
            Threads.@spawn f(r)
        end
    end
end
