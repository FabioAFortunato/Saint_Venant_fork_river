n_man = 101

include("../src/obj_func.jl")
include("../src/BFGS_BOBYQA.jl")
include("../src/BOBYQA_application.jl")
include("../src/MADS_application.jl")
include("../src/assimilacao.jl")

struct struct_Result
    method::String
    dim::Int
    tmax::Float64
    RMSD::Float64
    evaluation::Int
    time::Float64
    x_final::Vector
end

function teste_todos_novo()
    todos_resultados=[]
    for t in [5.0, 15.0, 31.0]
        for i = 1:3
            res_todos = run_problems(X0 = fill(0.09, i), tins = t)
            for res in res_todos
                append!(todos_resultados, struct_Result(res[2].metodo, 
                        res[2].dim, 
                        res[2].tend,
                        res[2].RMSD,
                        res[2].evaluation,
                        res[2].time,
                        res[2].x_final))
            end
        end
    end
end

function teste_todos(; t = 5.0, x = [0.1; 0.06])

    BFGS_BOBYQA_full_dim_problem(tmax = t, x0 = collect(range(x[1], x[2], length = 101)))
    BFGS_BOBYQA_two_dim_problem(tmax = t, x0 = x)
    full_dim_problem(tmax = t, x0 = collect(range(x[1], x[2], length = 101)))
    two_dim_problem(tmax = t, x0 = x)
    bobyqa_full_dim_problem(tmax = t, x0 = collect(range(x[1], x[2], length = 101)))
    bobyqa_two_dim_problem(tmax = t, x0 = x)
    mads_full_dim_problem(tmax = t, x0 = collect(range(x[1], x[2], length = 101)))
    mads_two_dim_problem(tmax = t, x0 = x)
end






