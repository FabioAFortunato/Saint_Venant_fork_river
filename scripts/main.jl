n_man = 101

include("../src/obj_func.jl")
include("../src/BFGS_BOBYQA.jl")
include("../src/BOBYQA_application.jl")
include("../src/MADS_application.jl")
include("../src/assimilacao.jl")


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






