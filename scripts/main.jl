n_man = 101

include("../src/obj_func.jl")
include("../src/BFGS_BOBYQA.jl")
include("../src/BOBYQA_application.jl")
include("../src/MADS_application.jl")


BFGS_BOBYQA_full_dim_problem(tmax = 5.0)
BFGS_BOBYQA_two_dim_problem(tmax = 5.0)
full_dim_problem(tmax = 5.0)
two_dim_problem(tmax = 5.0)
bobyqa_full_dim_problem(tmax = 5.0)
bobyqa_two_dim_problem(tmax = 5.0)
mads_full_dim_problem(tmax = 5.0)
mads_two_dim_problem(tmax = 5.0)


BFGS_BOBYQA_full_dim_problem(tmax = 15.0)
BFGS_BOBYQA_two_dim_problem(tmax = 15.0)
full_dim_problem(tmax = 15.0)
two_dim_problem(tmax = 15.0)
bobyqa_full_dim_problem(tmax = 15.0)
bobyqa_two_dim_problem(tmax = 15.0)
mads_full_dim_problem(tmax = 15.0)
mads_two_dim_problem(tmax = 15.0)



