module SidPsm

using LinearAlgebra
using Random
using Optim
using Logging

export Parameters, LinearDomain, Problem, CacheUtility, PenaltyUtility, GenClass,
       SearchStep, PollStep, Simplex, SidPsmAlgorithm, minimize!

# =========================================================================
# Parameters  (translated from Parameters.m)
# =========================================================================

mutable struct Parameters
    always::Bool
    cache::Bool
    economic::Int
    mesh_option::Int
    min_norm::Bool
    order_option::Int
    pruning::Int
    pss::Int
    regopt::Int
    search_option::Int
    shessian::Int
    store_all::Bool
    trs_solver::Int

    stop_alfa::Bool
    tol_alfa::Float64
    stop_fevals::Bool
    fevals_max::Int
    stop_grad::Int
    tol_grad::Float64
    stop_iter::Bool
    iter_max::Int

    phi::Float64
    phi_search::Float64
    theta::Float64

    epsilon_ini::Float64
    tol_feasible::Float64
    penalty_approach::Bool
    eta::Float64
    v::Float64
    beta_par::Float64
    zeta_par::Float64
    gamma_par::Float64
    OPM_problems::Int
    centralpath_option::Bool
    alfa_max::Float64
    const_log::Float64
    rho_in::Float64
    L_ini::Int
end

function Parameters(n::Int)
    theta = 0.5
    phi = 1.0
    phi_search = (1.0 / theta)^(1.0 / n)
    Parameters(
        true, true, 0, 0, true, 5, 0, 2, 1, 1, 0, true, 0,
        true, 1e-8, true, 4000, 0, 1e-5, false, 4000,
        phi, phi_search, theta,
        1e-2, 1e-5, true, 0.35, 1.1, 1.0, 1e2, 1e-5, 0, false, 1000.0, 1.0, 1e-3, 0,
    )
end

# =========================================================================
# LinearDomain (box constraints only: lb <= x <= ub)
# =========================================================================

mutable struct LinearDomain
    lb::Vector{Float64}
    ub::Vector{Float64}
end

function linear_feas(ld::LinearDomain, x::Vector{Float64})
    return all(x .>= ld.lb) && all(x .<= ld.ub)
end

# =========================================================================
# Problem
# =========================================================================

mutable struct Problem
    x0::Vector{Float64}
    n::Int
    m::Int
    p::Int
    linear_domain::LinearDomain
    bb::Union{Function,Nothing}       # x -> (f, g)
    func_f::Union{Function,Nothing}   # x -> f
end

function Problem(x0::Vector{Float64}, m::Int, p::Int, lb::Vector{Float64}, ub::Vector{Float64};
                  bb::Union{Function,Nothing}=nothing, func_f::Union{Function,Nothing}=nothing)
    return Problem(x0, length(x0), m, p, LinearDomain(lb, ub), bb, func_f)
end

# =========================================================================
# Cache_Utility
# =========================================================================

mutable struct CacheUtility
    X::Matrix{Float64}
    X_norms::Vector{Float64}
    M_values::Vector{Float64}
    F_values::Vector{Float64}
    G_values::Matrix{Float64}
    label::Vector{Int}
    succ::Vector{Int}
    tol_match::Float64
end

function CacheUtility(tol_alfa::Float64, n::Int, m::Int)
    CacheUtility(zeros(n, 0), Float64[], Float64[], Float64[], zeros(m, 0), Int[], Int[], 1e-2 * tol_alfa)
end

function initialize_cache!(cache::CacheUtility, problem::Problem, x0::Vector{Float64}, f_obj_0::Float64,
                            g_0::Union{Vector{Float64},Nothing}=nothing, f_m_0::Union{Float64,Nothing}=nothing)
    xnorm = norm(x0, 1)
    cache.X = reshape(copy(x0), :, 1)
    cache.X_norms = [xnorm]
    cache.succ = [1]

    if problem.m > 0
        if isfinite(f_m_0)
            cache.M_values = [f_m_0]
            cache.label = [1]
        else
            cache.M_values = [1e20]
            cache.label = [0]
        end
        if !isfinite(f_obj_0)
            cache.M_values = [1e20]
            cache.label = [-1]
        end
        cache.F_values = [f_obj_0]
        cache.G_values = reshape(copy(g_0), :, 1)
    else
        if isfinite(f_obj_0)
            cache.M_values = [f_obj_0]
            cache.F_values = [f_obj_0]
            cache.label = [1]
        else
            cache.M_values = [1e20]
            cache.F_values = [f_obj_0]
            cache.label = [-1]
        end
        cache.G_values = zeros(0, 1)
    end
    return nothing
end

function add_to_cache!(cache::CacheUtility, problem::Problem, x::Vector{Float64}, xnorm::Float64,
                        f_obj::Float64, g::Vector{Float64}, f_m::Float64, success::Bool)
    if success
        pos = 1
    else
        pos = 2
    end

    cache.X = hcat(cache.X[:, 1:pos-1], x, cache.X[:, pos:end])
    cache.X_norms = vcat(cache.X_norms[1:pos-1], xnorm, cache.X_norms[pos:end])
    cache.succ = vcat(cache.succ[1:pos-1], success ? 1 : 0, cache.succ[pos:end])

    if problem.m > 0
        if isfinite(f_m)
            cache.M_values = vcat(cache.M_values[1:pos-1], f_m, cache.M_values[pos:end])
            cache.label = vcat(cache.label[1:pos-1], 1, cache.label[pos:end])
        else
            cache.M_values = vcat(cache.M_values[1:pos-1], 1e20, cache.M_values[pos:end])
            lbl = f_obj < 1e20 ? 0 : -1
            cache.label = vcat(cache.label[1:pos-1], lbl, cache.label[pos:end])
        end
        cache.F_values = vcat(cache.F_values[1:pos-1], f_obj, cache.F_values[pos:end])
        cache.G_values = hcat(cache.G_values[:, 1:pos-1], g, cache.G_values[:, pos:end])
    else
        cache.F_values = vcat(cache.F_values[1:pos-1], f_obj, cache.F_values[pos:end])
        if f_obj < 1e20
            cache.M_values = vcat(cache.M_values[1:pos-1], f_obj, cache.M_values[pos:end])
            cache.label = vcat(cache.label[1:pos-1], 1, cache.label[pos:end])
        else
            cache.M_values = vcat(cache.M_values[1:pos-1], 1e20, cache.M_values[pos:end])
            cache.label = vcat(cache.label[1:pos-1], -1, cache.label[pos:end])
        end
        cache.G_values = hcat(cache.G_values[:, 1:pos-1], zeros(0), cache.G_values[:, pos:end])
    end
    return nothing
end

function move_top!(cache::CacheUtility, index_to_top::Int, m::Int, success::Bool)
    top = success ? 1 : 2

    if top != index_to_top
        rest = [i for i in top:length(cache.label) if i != index_to_top]
        order = vcat(index_to_top, rest)

        cache.X[:, top:end] = cache.X[:, order]
        cache.X_norms[top:end] = cache.X_norms[order]
        cache.M_values[top:end] = cache.M_values[order]
        cache.F_values[top:end] = cache.F_values[order]
        cache.label[top:end] = cache.label[order]
        cache.succ[top:end] = cache.succ[order]

        if m > 0
            cache.G_values[:, top:end] = cache.G_values[:, order]
        end
    end
    return nothing
end

function match_point(cache::CacheUtility, x::Vector{Float64}, xnorm::Float64, m::Int)
    f = Inf
    f_obj = Inf
    g = fill(Inf, m)
    index_to_move = 0

    index_norms = findall(abs.(cache.X_norms .- xnorm) .<= cache.tol_match)
    if isempty(index_norms)
        return false, x, f, f_obj, g, index_to_move
    end

    X_temp = cache.X[:, index_norms]
    nX = size(X_temp, 2)
    index = [j for j in 1:nX if maximum(abs.(X_temp[:, j] .- x)) <= cache.tol_match]
    match = !isempty(index)

    if match
        j0 = index_norms[index[1]]
        x = X_temp[:, index[1]]
        f = cache.M_values[j0]
        f_obj = cache.F_values[j0]
        if m > 0
            g = cache.G_values[:, j0]
        end
        index_to_move = j0
    end
    return match, x, f, f_obj, g, index_to_move
end

# =========================================================================
# Penalty_Utility
# =========================================================================

mutable struct PenaltyUtility
    slacks::BitVector
    rho_ex::Vector{Float64}
    rho_in::Float64
    const_log::Float64
    const_ex::Float64
    v::Float64
    gmin::Float64
    x_central_path::Vector{Float64}

    interior::Bool
    exterior::Bool

    strict_update::Bool
    get_best_after_update::Bool
end

PenaltyUtility() = PenaltyUtility(BitVector(), Float64[], Inf, 1.0, 1.0, 2.0, Inf, Float64[], false, false, false, true)

function initialize_penalty!(penalty::PenaltyUtility, cache::CacheUtility, problem::Problem,
                              x0::Vector{Float64}, f_obj_0::Float64, g_0::Vector{Float64})
    if problem.m <= 0
        return f_obj_0
    end

    if problem.p == 0
        penalty.slacks = BitVector(g_0 .> -1e-15)
    else
        head = g_0[1:end-problem.p] .> -1e-15
        penalty.slacks = BitVector(vcat(head, trues(problem.p)))
    end

    penalty.gmin = minimum(abs.(g_0[.!penalty.slacks]))

    if sum(penalty.slacks) > 0
        penalty.exterior = true
        penalty.rho_ex = fill(0.1, sum(penalty.slacks))
        magn_f = floor(log10(abs(f_obj_0)))
        penalty.const_ex = 10.0^max(0.0, magn_f)
    end
    if sum(.!penalty.slacks) > 0
        penalty.interior = true
        penalty.rho_in = 0.1
    end

    f_m_0 = evaluate_penalty(penalty, f_obj_0, g_0)
    initialize_cache!(cache, problem, x0, f_obj_0, g_0, f_m_0)
    return f_m_0
end

function evaluate_penalty(penalty::PenaltyUtility, f_obj::Float64, g::Vector{Float64})
    f = f_obj
    g_in = g[.!penalty.slacks]
    g_ext = g[penalty.slacks]

    if penalty.interior
        viol_in = sum(max.(0.0, g_in))
        if viol_in >= 1e-20 || f_obj >= 1e20
            f = Inf
        else
            f = f - penalty.const_log * penalty.rho_in * sum(log.(-g_in))
        end
    end

    if penalty.exterior
        f = f + penalty.const_ex * sum((1.0 ./ penalty.rho_ex) .* (max.(0.0, g_ext) .^ penalty.v))
    end
    return f
end

function evaluate_penalty_cache!(penalty::PenaltyUtility, cache::CacheUtility, switch_pen::Bool)
    active = cache.label .== 1
    F = cache.F_values[active]
    G_in = cache.G_values[.!penalty.slacks, active]
    G_ext = cache.G_values[penalty.slacks, active]

    if switch_pen
        G_in_max = isempty(G_in) ? Float64[] : vec(maximum(G_in, dims=1))
        index_unf = G_in_max .>= 0
        F2 = copy(F)
        F2[index_unf] .= 1e20
        cache.M_values[active] = F2
        cache.label[active] = Int.(.!index_unf)

        active = cache.label .== 1
        F = cache.F_values[active]
        G_in = cache.G_values[.!penalty.slacks, active]
        G_ext = cache.G_values[penalty.slacks, active]
    end

    if penalty.interior
        F = F .- penalty.const_log .* penalty.rho_in .* vec(sum(log.(-G_in), dims=1))
    end
    if penalty.exterior
        F = F .+ penalty.const_ex .* vec(((1.0 ./ penalty.rho_ex)') * (max.(0.0, G_ext) .^ penalty.v))
    end
    cache.M_values[active] = F
    return nothing
end

# Quadratic model container: f(x) = x'Hf*x + gf'*x + F0 ; g_i(x) = x'HG[i]*x + gG[i]'*x + G0[i]
struct QuadH
    f::Matrix{Float64}
    G::Vector{Matrix{Float64}}
end

struct LinG
    f::Vector{Float64}
    G::Vector{Vector{Float64}}
end

function evaluate_penalty_model(penalty::PenaltyUtility, cache::CacheUtility, x::Vector{Float64},
                                 H::Union{QuadH,Nothing}, g::LinG, linear::Bool)
    m = length(g.G)
    gmodel = Vector{Float64}(undef, m)
    delta_g = Vector{Vector{Float64}}(undef, m)

    if !linear
        f = x' * H.f * x + g.f' * x + cache.F_values[1]
        delta_f = H.f * x + g.f
        for i in 1:m
            gmodel[i] = x' * H.G[i] * x + g.G[i]' * x + cache.G_values[i, 1]
            delta_g[i] = H.G[i]' * x + g.G[i]
        end
    else
        f = g.f' * x + cache.F_values[1]
        delta_f = copy(g.f)
        for i in 1:m
            gmodel[i] = g.G[i]' * x + cache.G_values[i, 1]
            delta_g[i] = copy(g.G[i])
        end
    end

    slacks = penalty.slacks
    g_in = gmodel[.!slacks]
    g_ext = gmodel[slacks]

    if penalty.interior
        viol_in = sum(max.(0.0, g_in))
        if viol_in > 0
            f = 1e30
            delta_f = fill(1e30, length(delta_f))
        else
            idx_in = findall(.!slacks)
            s = zeros(length(delta_f))
            for (k, i) in enumerate(idx_in)
                s .+= (1.0 / g_in[k]) .* delta_g[i]
            end
            f = f - penalty.const_log * penalty.rho_in * sum(log.(-g_in))
            delta_f = delta_f .- penalty.const_log * penalty.rho_in .* s
        end
    end

    if penalty.exterior
        idx_ext = findall(slacks)
        s = zeros(length(delta_f))
        for (k, i) in enumerate(idx_ext)
            s .+= (penalty.v / penalty.rho_ex[k]) * max(0.0, g_ext[k])^(penalty.v - 1) .* delta_g[i]
        end
        f = f + penalty.const_ex * sum((1.0 ./ penalty.rho_ex) .* (max.(0.0, g_ext) .^ penalty.v))
        delta_f = delta_f .+ penalty.const_ex .* s
    end

    return f, delta_f
end

function update_penalty_parameter!(penalty::PenaltyUtility, alg, cache::CacheUtility, problem::Problem, success::Bool)
    update_penalty = false

    if penalty.interior && !success
        criterion = min(1e2 * penalty.rho_in^alg.params.beta_par, 1e10 * penalty.gmin^2)
        if alg.alfa <= criterion
            penalty.rho_in = penalty.rho_in / alg.params.zeta_par
            update_penalty = true
        end
    end

    if penalty.exterior && ((!success && ((penalty.strict_update && update_penalty) || !penalty.strict_update)) || !penalty.interior)
        if alg.alfa <= alg.params.zeta_par * maximum(penalty.rho_ex .^ alg.params.beta_par)
            maxi = maximum(penalty.rho_ex)
            index = penalty.rho_ex .== maxi
            penalty.rho_ex[index] .= penalty.rho_ex[index] ./ alg.params.zeta_par
            update_penalty = true
        end
    end

    if update_penalty
        evaluate_penalty_cache!(penalty, cache, false)
        alg.f_current = cache.M_values[1]

        if penalty.get_best_after_update
            idx = argmin(cache.M_values)
            alg.f_current = cache.M_values[idx]
            alg.x_current = cache.X[:, idx]
            alg.f_obj_current = cache.F_values[idx]
            alg.g_current = cache.G_values[:, idx]

            n_cache = size(cache.X, 2)
            new_order = vcat(idx, [i for i in 1:n_cache if i != idx])

            cache.X = cache.X[:, new_order]
            cache.X_norms = cache.X_norms[new_order]
            cache.M_values = cache.M_values[new_order]
            cache.F_values = cache.F_values[new_order]
            cache.G_values = cache.G_values[:, new_order]
            cache.label = cache.label[new_order]
            cache.succ = cache.succ[new_order]
        end
    end

    return update_penalty
end

function switch_penalty!(penalty::PenaltyUtility, alg, problem::Problem, cache::CacheUtility)
    switch_to_log = false

    for i in 1:problem.m
        if penalty.slacks[i] && i <= problem.m - problem.p
            if alg.g_current[i] < -1e-15
                index_to_rem_vec = findall(penalty.slacks)
                index_to_rem = findfirst(==(i), index_to_rem_vec)
                penalty.slacks[i] = false
                penalty.rho_ex = penalty.rho_ex[1:end .!= index_to_rem]
                switch_to_log = true
            end
        end
    end

    if switch_to_log
        if sum(penalty.slacks) == 0
            penalty.exterior = false
        end
        evaluate_penalty_cache!(penalty, cache, true)
        alg.f_current = cache.M_values[1]
    end
    return nothing
end

# =========================================================================
# Gen_class
# =========================================================================

mutable struct GenClass
    tol_degconst::Float64
    pss::Int
    dense::Bool
    nD::Int
    max_D::Float64
    max_D_old::Float64
    L::Int
    eps_active::Float64
end

function GenClass(pss::Int, n::Int, alfa::Float64, epsilon_ini::Float64)
    GenClass(1e-3, pss, true, 0, 0.0, 0.0, 0, min(epsilon_ini, 1e2 * alfa))
end

function gen!(g::GenClass, problem::Problem, x::Vector{Float64})
    n = problem.n
    local D::Matrix{Float64}

    if !g.dense
        if g.pss == 0
            D = hcat(-ones(n, 1), Matrix{Float64}(I, n, n))
        elseif g.pss == 1
            D = hcat(Matrix{Float64}(I, n, n), -Matrix{Float64}(I, n, n))
        elseif g.pss == 2
            D = hcat(ones(n, 1), -ones(n, 1), Matrix{Float64}(I, n, n), -Matrix{Float64}(I, n, n))
        else
            Daux = Matrix{Float64}(I, n, n) * (1.0 / (n + 1)) .+ (zeros(n, n) .- 1.0 / n)
            C = cholesky(Symmetric(Daux)).U
            D = hcat(sum(C, dims=1)', -Matrix(C))
        end
    else
        a = randn(n)
        a = a / norm(a)
        H = Matrix{Float64}(I, n, n) - 2 * (a * a')
        Daux = zeros(n, n)
        for i in 1:n
            Daux[:, i] = H[:, i] / norm(H[:, i])
        end
        D = hcat(Daux, -Daux)
    end

    g.nD = size(D, 2)
    g.max_D = maximum(sqrt.(vec(sum(D .^ 2, dims=1))))
    return D
end

function order_D!(g::GenClass, alg, D::Matrix{Float64}, gvec::Vector{Float64}, H::Matrix{Float64}, poised::Int)
    if alg.params.order_option in (1, 5)
        di = poised == 2 ? -(1.0 ./ diag(H)) .* gvec : -gvec

        colnorms = sqrt.(vec(sum(D .^ 2, dims=1)))
        di_cosines = (di' * D)[1, :] ./ (colnorms .* norm(di))
        index = sortperm(-di_cosines)
        D = D[:, index]
        di_cosines = di_cosines[index]

        alg.poll_step.col_index = 1

        if alg.params.pruning != 0
            if alg.params.pruning == 1
                D = D[:, 1:1]
            else
                D = D[:, findall(di_cosines .> 0)]
            end
            g.nD = size(D, 2)
        end
    end
    return D
end

# =========================================================================
# Search_Step
# =========================================================================

mutable struct SearchStep
    sigma::Float64
    tol_Delta::Float64
    search_subspace::Bool
    use_unfeasible::Bool
    sing_models::Bool
    only_succ::Bool

    H_old::QuadH
    g_old::LinG
    active_index_old::BitVector
    H_calc::Int
end

SearchStep() = SearchStep(2.0, 1e-12, true, true, true, false,
                           QuadH(zeros(1, 1), Matrix{Float64}[]), LinG(zeros(1), Vector{Float64}[]),
                           trues(1), 0)

function quad_frob(ss::SearchStep, alg, X::Matrix{Float64}, Fvals)
    tol_svd = eps()
    n, m = size(X)

    F_values, G_values = Fvals
    m_G = size(G_values, 1)

    quad = 0
    H = QuadH(zeros(n, n), [zeros(n, n) for _ in 1:m_G])
    g = LinG(zeros(n), [zeros(n) for _ in 1:m_G])

    if m <= n + 1
        quad = 0
        if alg.params.always && ss.H_calc != 0
            quad = 2
        end
        return quad, H, g
    end

    quad = 1
    Y = X
    Y_values_G = G_values
    Y_values = F_values

    m_Y = size(Y, 2)
    Y = Y .- Y[:, 1]

    b = vcat(Y_values, zeros(n + 1))
    b_G = zeros(m_Y + n + 1, m_G)
    for c in 1:m_G
        b_G[1:m_Y, c] = Y_values_G[c, :]
    end

    A = ((Y' * Y) .^ 2) ./ 2
    W = [A ones(m_Y, 1) Y'; ones(1, m_Y) zeros(1, n + 1); Y zeros(n, n + 1)]

    F = svd(W)
    Sdiag = copy(F.S)
    Sdiag[Sdiag .< tol_svd] .= tol_svd
    Sinv = Diagonal(1.0 ./ Sdiag)
    lambda = F.V * Sinv * F.U' * b

    lambda_G = zeros(length(b), m_G)
    for c in 1:m_G
        lambda_G[:, c] = F.V * Sinv * F.U' * b_G[:, c]
    end

    gf = lambda[m_Y+2:m_Y+n+1]
    Hf = zeros(n, n)
    for j in 1:m_Y
        Hf .+= lambda[j] .* (Y[:, j] * Y[:, j]')
    end

    HG = Vector{Matrix{Float64}}(undef, m_G)
    gG = Vector{Vector{Float64}}(undef, m_G)
    for c in 1:m_G
        gG[c] = lambda_G[m_Y+2:m_Y+n+1, c]
        Hc = zeros(n, n)
        for j in 1:m_Y
            Hc .+= lambda_G[j, c] .* (Y[:, j] * Y[:, j]')
        end
        HG[c] = Hc
    end

    return quad, QuadH(Hf, HG), LinG(gf, gG)
end

function search_step!(ss::SearchStep, alg, cache::CacheUtility, problem::Problem, p_max::Function)
    n = problem.n
    d_trust = zeros(n)
    model_dec = 0.0
    ball_distance = 1.0

    if alg.iter == 0
        alg.gen.max_D_old = alg.gen.max_D
    end

    Delta = alg.alfa * ss.sigma * alg.gen.max_D_old
    if Delta < ss.tol_Delta
        Delta = ss.tol_Delta
    end

    if Delta <= cache.tol_match
        return d_trust, model_dec, ball_distance
    end

    active_index = trues(n)
    n_active = n

    if ss.only_succ
        cols = findall((cache.label .> 0) .& (cache.succ .> 0))
    elseif !ss.use_unfeasible
        cols = findall(cache.label .> 0)
    else
        cols = findall(cache.label .> -1)
    end
    X_aux = cache.X[:, cols]

    if size(X_aux, 2) > 1
        while true
            X_aux = X_aux[:, 1:min(size(X_aux, 2), p_max(n_active))]
            if ss.search_subspace
                Daux = X_aux .- alg.x_current
                norms = [norm(Daux[i, :]) for i in axes(Daux, 1)]
                active_index = norms .>= eps()
                if sum(active_index) == n_active || sum(active_index) == 1
                    n_active = sum(active_index)
                    break
                end
                n_active = sum(active_index)
            else
                break
            end
        end
    end

    npts = min(size(X_aux, 2), p_max(n_active))
    if ss.only_succ
        cols2 = findall((cache.label .> 0) .& (cache.succ .> 0))
    elseif !ss.use_unfeasible
        cols2 = findall(cache.label .> 0)
    else
        cols2 = findall(cache.label .> -1)
    end
    G_values_aux = cache.G_values[:, cols2][:, 1:npts]
    F_values_aux = cache.F_values[cols2][1:npts]
    M_values_aux = (F_values_aux, G_values_aux)

    if n_active > 1
        quad, H, g = quad_frob(ss, alg, X_aux[active_index, :], M_values_aux)

        if quad != 0
            if quad == 1
                ss.H_old = H
                ss.g_old = g
                ss.active_index_old = copy(active_index)
                ss.H_calc += 1
            elseif quad == 2
                H = ss.H_old
                g = ss.g_old
                active_index = copy(ss.active_index_old)
                n_active = sum(active_index)
            end

            lb = copy(problem.linear_domain.lb)
            ub = copy(problem.linear_domain.ub)
            if alg.scale_x
                lb[alg.scaling_mask] .= 0.0
                ub[alg.scaling_mask] .= 10.0
            end
            lb_solver = lb[active_index] .- alg.x_current[active_index]
            ub_solver = ub[active_index] .- alg.x_current[active_index]
            lb_solver = max.(lb_solver, -Delta)
            ub_solver = min.(ub_solver, Delta)
            fixed = problem.linear_domain.ub .== problem.linear_domain.lb
            ub_solver[fixed[active_index]] .= 0.0
            lb_solver[fixed[active_index]] .= 0.0

            xtrust, model_opt = subproblem_solver(alg, cache, H, g, n_active, lb_solver, ub_solver, false)

            model_dec = evaluate_penalty_model(alg.penalty, cache, zeros(n_active), H, g, false)[1] - model_opt
            ball_distance = (Delta - norm(xtrust)) / Delta

            d_trust[active_index] = xtrust
        end
    else
        d_trust[active_index] .= 2 * alg.alfa
        ball_distance = 0.0
    end

    return d_trust, model_dec, ball_distance
end

# =========================================================================
# subproblem_solver  (Julia replacement for fmincon + GlobalSearch)
# =========================================================================

function subproblem_solver(alg, cache::CacheUtility, H::Union{QuadH,Nothing}, g::LinG,
                            n_active::Int, lb::Vector{Float64}, ub::Vector{Float64}, linear::Bool;
                            n_starts::Int=4, rng::AbstractRNG=Random.default_rng())
    lb = min.(lb, 0.0)
    ub = max.(ub, 0.0)

    f_obj = y -> evaluate_penalty_model(alg.penalty, cache, y, H, g, linear)[1]
    function grad!(G, y)
        _, dP = evaluate_penalty_model(alg.penalty, cache, y, H, g, linear)
        G .= dP
        return G
    end

    x0 = zeros(n_active)
    best_x = x0
    best_f = f_obj(x0)

    starts = Vector{Vector{Float64}}()
    push!(starts, x0)
    for _ in 1:n_starts
        push!(starts, lb .+ rand(rng, n_active) .* (ub .- lb))
    end

    for xs in starts
        xs = clamp.(xs, lb, ub)
        try
            res = with_logger(NullLogger()) do
                Optim.optimize(f_obj, grad!, lb, ub, xs, Fminbox(LBFGS()),
                                Optim.Options(iterations=100, outer_iterations=10))
            end
            fval = Optim.minimum(res)
            if isfinite(fval) && fval < best_f
                best_f = fval
                best_x = Optim.minimizer(res)
            end
        catch
            continue
        end
    end

    return best_x, best_f
end

# =========================================================================
# Poll_Step
# =========================================================================

mutable struct PollStep
    first_flag_colD::Bool
    col_index::Int
    dir_past::Vector{Float64}
end

function PollStep(order_option::Int, n::Int)
    PollStep(order_option != 2, 1, zeros(n))
end

function poll_step!(ps::PollStep, alg, problem::Problem, cache::CacheUtility, D::Matrix{Float64})
    count_col = 0
    success = false
    xtemp = alg.x_current

    if !(alg.params.order_option in (4, 5, 8, 9)) || alg.iter == 0
        ps.col_index = 1
    else
        if ps.col_index == alg.gen.nD + 1
            ps.col_index = 1
        end
    end
    max_nD = alg.gen.nD

    if alg.params.order_option == 3
        D = D[:, randperm(alg.gen.nD)]
    end

    while !success && count_col < max_nD
        colD = D[1:problem.n, ps.col_index]
        xtemp = alg.x_current + alg.alfa * colD

        xtemp, success = evaluate_point!(alg, problem, cache, xtemp)

        ps.col_index += 1
        if ps.col_index == alg.gen.nD + 1
            ps.col_index = 1
        end
        count_col += 1
    end

    return xtemp, success
end

function mesh_proc(ps::PollStep, alg, success::Bool, poised_data::Tuple, ball_distance::Float64, succ_trust_linear::Bool)
    poised = poised_data[1]
    gvec = length(poised_data) >= 2 ? poised_data[2] : Float64[]
    H = length(poised_data) >= 3 ? poised_data[3] : zeros(0, 0)

    x = alg.x_prec
    xtemp = alg.x_current
    f = alg.f_prec
    ftemp = alg.f_current
    alfa = alg.alfa

    match_dir = false
    dir_new = (xtemp - x) ./ alfa
    if alg.params.mesh_option == 3
        if success && dir_new == ps.dir_past
            match_dir = true
        end
        ps.dir_past = dir_new
    end

    gamma1 = 0.25
    gamma2 = 0.75
    tol_rho = 1e-8

    if success
        if alg.params.mesh_option in (1, 2) && poised in (1, 2)
            deg_rho = false
            if poised == 2
                if abs(gvec' * (xtemp - x) + 0.5 * (xtemp - x)' * H * (xtemp - x)) <= tol_rho
                    deg_rho = true
                end
            else
                if abs(gvec' * (xtemp - x)) <= tol_rho
                    deg_rho = true
                end
            end

            if !deg_rho
                rho = if poised == 2
                    (ftemp - f) / (gvec' * (xtemp - x) + 0.5 * (xtemp - x)' * H * (xtemp - x))
                else
                    (ftemp - f) / (gvec' * (xtemp - x))
                end
                if rho > gamma2
                    alfa = alg.params.phi * alfa
                end
                if alg.params.mesh_option == 1 && rho <= gamma1
                    alfa = alg.params.theta * alfa
                end
            else
                if alg.params.mesh_option == 1
                    alfa = alg.params.theta * alfa
                end
            end
        else
            if (alg.params.mesh_option != 3 && norm(dir_new, 2) >= 1.0) || (alg.params.mesh_option == 3 && match_dir)
                alfa = alg.params.phi * alfa
            end
            if alg.params.phi == 1 && (alg.trust_succ == 1 || succ_trust_linear) && ball_distance <= 0.5
                alfa = alg.params.phi_search * alfa
            end
        end
        alfa = min(alg.params.alfa_max, alfa)
    else
        alfa = alg.params.theta * alfa
    end

    return alfa
end

# =========================================================================
# Simplex
# =========================================================================

mutable struct Simplex
    tol_hess::Float64
    lambda::Float64
    tol_degset::Float64
    sigma::Float64
    deriv_old::LinG
    active_index_old::BitVector

    simplex_subspace::Bool
    use_unfeasible::Bool
    sing_models::Bool
    bound_active::Bool
end

function Simplex(n::Int)
    Simplex(1e-3, 100.0, sqrt(eps()), 2.0, LinG(zeros(n), Vector{Float64}[]), trues(n),
            true, true, true, false)
end

function lambda_poised(sx::Simplex, alg, X::Matrix{Float64}, Fvalues, s_min::Int, s_max::Int, Delta::Float64)
    n, m = size(X)
    shessian = alg.params.shessian

    if (shessian != 2 && m < s_min) || (shessian == 2 && m < floor(Int, (s_min + 1) / 2))
        return 0, zeros(n, m), nothing
    end

    Sinitial = (X[:, 2:m] .- X[:, 1]) ./ Delta
    index_initial = findall(sqrt.(vec(sum(Sinitial .^ 2, dims=1))) .<= 1.0)

    poised = 0
    Y = zeros(n, m)
    Y_values = nothing
    index = Int[]

    test_poised = 0
    while test_poised != 2
        index = copy(index_initial)
        S = Sinitial[:, index]
        m_S = size(S, 2)

        if (shessian != 2 && m_S < s_min - 1) || (shessian == 2 && m_S < floor(Int, (s_min + 1) / 2) - 1)
            poised = 0
            Y = zeros(n, m)
            Y_values = nothing
            test_poised = 2
        else
            poised = 0
            if alg.params.economic != 0
                test_eco = 0
                while poised == 0 && test_eco <= 1
                    m_block = test_eco == 0 ? min(m_S, s_max - 1) : s_min - 1
                    if m_block >= s_min - 1
                        Ymat = shessian != 0 ? vcat(S[:, 1:m_block], 0.5 .* S[:, 1:m_block] .^ 2)' : S[:, 1:m_block]'
                        if alg.params.economic == 2
                            R = qr(Ymat).R
                            dR = diag(R)
                            Y_lambda = minimum(abs.(dR)) >= sx.tol_degset ? norm(1.0 ./ dR) : Inf
                        else
                            Y_lambda = cond(Ymat) / norm(Ymat)
                        end
                        if Y_lambda <= sx.lambda
                            poised = (shessian != 0 && test_eco == 0) ? 2 : 1
                            index = vcat(1, index[1:m_block] .+ 1)
                        end
                    end
                    test_eco += 1
                    if test_eco == 1 && shessian == 2
                        shessian = 0
                        s_min = floor(Int, (s_min + 1) / 2)
                        s_max = floor(Int, (s_max + 1) / 2)
                        test_eco = 0
                    end
                end
            end

            if poised == 0
                cont_S = 1
                cont_Y = 0
                Ymat = zeros(n * (shessian != 0 ? 2 : 1), 0)
                index_mask = ones(Int, m_S)

                while cont_S <= m_S && cont_Y < s_max - 1
                    aux = shessian != 0 ? vcat(S[:, cont_S], 0.5 .* S[:, cont_S] .^ 2) : S[:, cont_S]
                    Y_aux = hcat(Ymat, aux)'

                    if alg.params.economic == 2
                        R = qr(Y_aux).R
                        dR = diag(R)
                        Y_lambda = minimum(abs.(dR)) >= sx.tol_degset ? norm(1.0 ./ dR) : Inf
                    else
                        Y_lambda = cond(Y_aux) / norm(Y_aux)
                    end

                    if Y_lambda <= sx.lambda
                        poised = 1
                        cont_Y += 1
                        Ymat = hcat(Ymat, aux)
                    else
                        index_mask[cont_S] = 0
                    end
                    cont_S += 1
                end

                if cont_Y < s_min - 1
                    poised = 0
                    Y = zeros(n, m)
                    Y_values = nothing
                else
                    kept = findall(index_mask[1:cont_S-1] .== 1)
                    index = vcat(1, index[kept] .+ 1)
                end
            end

            if shessian == 2 && poised == 0 && test_poised == 0
                shessian = 0
                s_min = floor(Int, (s_min + 1) / 2)
                s_max = floor(Int, (s_max + 1) / 2)
                test_poised = 1
            else
                if alg.params.shessian != 0 && poised != 0 && test_poised == 0
                    poised = 2
                end
                test_poised = 2
            end
        end
    end

    if poised != 0
        Y = X[:, index]
        F_values_aux = Fvalues[1][index]
        G_values_aux = Fvalues[2][:, index]
        Y_values = (F_values_aux, G_values_aux)
    end

    return poised, Y, Y_values
end

function simplex_deriv(sx::Simplex, alg, S::AbstractMatrix{Float64}, invS::AbstractMatrix{Float64}, n::Int, nsimp::Int,
                        F_values::Vector{Float64}, sd_order::Int, active_index::BitVector)
    delta = F_values[2:nsimp] .- F_values[1]

    determined = false
    overdeterm = false
    underdeterm = false
    threshold = sd_order == 2 ? 2n + 1 : n + 1
    if nsimp == threshold
        determined = true
    elseif nsimp > threshold
        overdeterm = true
    else
        underdeterm = true
    end

    if determined
        deriv = S' \ delta
    else
        if underdeterm && !alg.params.min_norm
            delta = delta .- S' * sx.deriv_old.f[active_index]
        end
        deriv = invS * delta
        if underdeterm && !alg.params.min_norm
            deriv = sx.deriv_old.f[active_index] .+ deriv
        end
    end
    return deriv
end

function simplex_phase!(sx::Simplex, alg, problem::Problem, cache::CacheUtility, p_max::Function, s_min::Function, s_max::Function)
    n = problem.n
    H = zeros(n, n)
    gvec = zeros(n)

    active_index = trues(n)
    n_active = n
    if sx.bound_active
        box_distance = min.(alg.x_current .- problem.linear_domain.lb, problem.linear_domain.ub .- alg.x_current)
        active_index = box_distance .>= alg.alfa
        n_active = sum(active_index)
    end

    cols = sx.use_unfeasible ? findall(cache.label .> -1) : findall(cache.label .> 0)
    X_aux = cache.X[:, cols]

    if size(X_aux, 2) > 1
        while true
            X_aux = X_aux[:, 1:min(size(X_aux, 2), p_max(n_active))]
            if sx.simplex_subspace
                Daux = X_aux .- alg.x_current
                norms = [norm(Daux[i, :]) for i in axes(Daux, 1)]
                active_index = norms .>= eps()
                if sum(active_index) == n_active || sum(active_index) == 1
                    n_active = sum(active_index)
                    break
                end
                n_active = sum(active_index)
            else
                break
            end
        end
    else
        n_active = 0
    end

    poised = 0
    Y = zeros(n, 0)
    Delta = 0.0
    gvec_out = gvec
    H_out = H

    if n_active > 0
        cols2 = sx.use_unfeasible ? findall(cache.label .> -1) : findall(cache.label .> 0)
        G_values_aux = cache.G_values[:, cols2]
        F_values_aux = cache.F_values[cols2]
        npts = length(F_values_aux)
        G_values_aux = G_values_aux[:, 1:npts]
        F_values_aux = F_values_aux[1:npts]

        s_min_poised = s_min(n_active)
        s_max_poised = s_max(n_active)
        Delta = sx.sigma * alg.alfa * alg.gen.max_D_old

        poised, Yp, Yvals = lambda_poised(sx, alg, X_aux[active_index, :], (F_values_aux, G_values_aux), s_min_poised, s_max_poised, Delta)

        if poised != 0
            Y = Yp
            Y_values, Y_values_G = Yvals
            nsimp = size(Y, 2)
            S = Y[:, 2:nsimp] .- Y[:, 1]
            if poised == 2
                S = vcat(S, 0.5 .* S .^ 2)
            end
            Sinv = inv(S')

            deriv_active_f = simplex_deriv(sx, alg, S, Sinv, n, nsimp, vec(Y_values), poised, active_index)

            m_G = size(Y_values_G, 1)
            deriv_active_G = Vector{Vector{Float64}}(undef, m_G)
            for cons in 1:m_G
                deriv_active_G[cons] = simplex_deriv(sx, alg, S, Sinv, n, nsimp, Y_values_G[cons, :], poised, active_index)
            end

            lb = copy(problem.linear_domain.lb)
            ub = copy(problem.linear_domain.ub)
            if alg.scale_x
                lb[alg.scaling_mask] .= 0.0
                ub[alg.scaling_mask] .= 10.0
            end
            lb_solver = lb[active_index] .- alg.x_current[active_index]
            ub_solver = ub[active_index] .- alg.x_current[active_index]
            lb_solver = max.(lb_solver, -Delta)
            ub_solver = min.(ub_solver, Delta)
            fixed = problem.linear_domain.ub .== problem.linear_domain.lb
            ub_solver[fixed[active_index]] .= 0.0
            lb_solver[fixed[active_index]] .= 0.0

            gmodel = LinG(deriv_active_f, deriv_active_G)
            deriv_active, _ = subproblem_solver(alg, cache, nothing, gmodel, n_active, lb_solver, ub_solver, true)

            deriv = zeros(n)
            deriv[active_index] .+= deriv_active

            if !alg.params.min_norm
                sx.deriv_old = LinG(deriv, deriv_active_G)
                sx.active_index_old = copy(active_index)
            end

            gvec_out = deriv[1:n]
            if poised == 2
                aux_hess = deriv[n+1:2n]
                logic = abs.(aux_hess) .< sx.tol_hess
                aux_hess[logic] .+= sign.(aux_hess[logic]) .* sx.tol_hess
                logic0 = aux_hess .== 0
                aux_hess[logic0] .+= sx.tol_hess
                H_out = Diagonal(aux_hess) |> Matrix
            end
        end
    end

    return poised, gvec_out, H_out, Y, Delta
end

# =========================================================================
# Main algorithm  (Sid_Psm_class.m)
# =========================================================================

mutable struct SidPsmAlgorithm
    func_eval::Int
    iter::Int
    iter_suc::Int
    iter_uns::Int
    func_iter::Int
    store_fail_count::Int

    switch_strategy::Bool
    lin_proj::Bool
    constraint_satur::Float64
    scale_x::Bool
    scaling_mask::BitVector

    DFO_DATA::Vector{Float64}
    VIOL_DATA::Vector{Float64}

    x_current::Vector{Float64}
    f_current::Float64
    f_obj_current::Float64
    g_current::Vector{Float64}
    x_prec::Vector{Float64}
    f_prec::Float64
    f_obj_prec::Float64
    g_prec::Vector{Float64}
    alfa::Float64
    trust_succ::Int
    search_dec_fail::Int

    params::Parameters
    penalty::PenaltyUtility
    gen::GenClass
    search_step::SearchStep
    poll_step::PollStep
    simplex::Simplex
end

function SidPsmAlgorithm(problem::Problem)
    params = Parameters(problem.n)
    alfa0 = 1.0
    SidPsmAlgorithm(
        0, 0, 0, 0, 0, 0,
        false, false, -1e1, true, falses(problem.n),
        Float64[], Float64[],
        Float64[], Inf, Inf, Float64[], Float64[], Inf, Inf, Float64[], alfa0, 0, 0,
        params, PenaltyUtility(), GenClass(params.pss, problem.n, alfa0, params.epsilon_ini),
        SearchStep(), PollStep(params.order_option, problem.n), Simplex(problem.n),
    )
end

function initialization!(alg::SidPsmAlgorithm, problem::Problem, cache::CacheUtility)
    feas = true
    x0 = copy(problem.x0)

    if alg.scale_x
        lb = problem.linear_domain.lb
        ub = problem.linear_domain.ub
        alg.scaling_mask = (ub .< 1e20) .& (lb .> -1e20) .& (ub .> lb)
        x0[alg.scaling_mask] = ((problem.x0[alg.scaling_mask] .- lb[alg.scaling_mask]) ./
                                 (ub[alg.scaling_mask] .- lb[alg.scaling_mask])) .* 10
    end

    local f_obj_0::Float64
    local g_0::Vector{Float64}

    if problem.m == 0
        f_obj_0 = problem.func_f(problem.x0)
        g_0 = Float64[]
    else
        f_obj_0, g_0 = problem.bb(problem.x0)
    end
    alg.func_eval += 1

    if !isfinite(f_obj_0)
        feas = false
        f_m_0 = Inf
        println("Initial point provided is not in the domain of the objective function.")
        println("Please call sid_psm with a valid point.")
        return feas, f_obj_0, g_0, Inf, x0
    end

    local f_m_0::Float64
    if problem.m == 0
        initialize_cache!(cache, problem, x0, f_obj_0)
        f_m_0 = f_obj_0
        alg.DFO_DATA = [f_obj_0]
    else
        g_0 = max.(g_0, alg.constraint_satur)
        viol = sum(max.(0.0, g_0))
        alg.DFO_DATA = [f_obj_0]
        alg.VIOL_DATA = [viol]

        if !alg.params.penalty_approach
            f_m_0 = maximum(g_0) > 0 ? Inf : f_obj_0
            initialize_cache!(cache, problem, x0, f_obj_0, g_0, f_m_0)
        else
            f_m_0 = initialize_penalty!(alg.penalty, cache, problem, x0, f_obj_0, g_0)
        end

        if !isfinite(f_m_0)
            feas = false
            println("Initial point provided is not feasible.")
            return feas, f_obj_0, g_0, f_m_0, x0
        end
    end

    return feas, f_obj_0, g_0, f_m_0, x0
end

function evaluate_point!(alg::SidPsmAlgorithm, problem::Problem, cache::CacheUtility, x::Vector{Float64})
    if !isempty(alg.x_current) && sum(x .- alg.x_current) == 0
        return x, false
    end

    if alg.scale_x
        original_x = copy(x)
        original_x[alg.scaling_mask] = (x[alg.scaling_mask] ./ 10) .*
            (problem.linear_domain.ub[alg.scaling_mask] .- problem.linear_domain.lb[alg.scaling_mask]) .+
            problem.linear_domain.lb[alg.scaling_mask]
    else
        original_x = copy(x)
    end

    feas = linear_feas(problem.linear_domain, original_x)
    if alg.lin_proj
        x = max.(problem.linear_domain.lb, min.(problem.linear_domain.ub, x))
    end

    if !feas
        return x, false
    end

    xnorm = norm(x, 1)
    match, x, f, f_obj, g, index_match = match_point(cache, x, xnorm, problem.m)

    if !match
        alg.func_eval += 1

        if problem.m == 0
            f_obj = problem.func_f(original_x)
            g = Float64[]
        else
            f_obj, g = problem.bb(original_x)
        end
        f = f_obj

        if problem.m > 0
            g = max.(min.(1e10, g), alg.constraint_satur)
            viol = sum(max.(0.0, g))
            push!(alg.DFO_DATA, f_obj)
            push!(alg.VIOL_DATA, viol)

            if !alg.params.penalty_approach
                if viol > 0
                    f = Inf
                end
            else
                f = evaluate_penalty(alg.penalty, f_obj, g)
            end
        else
            g = Float64[]
            push!(alg.DFO_DATA, f_obj)
        end
    end

    success = false
    if f < alg.f_current - min(alg.params.gamma_par, alg.params.gamma_par * min(alg.alfa, 1.0)^2)
        alg.store_fail_count = 0
        success = true
        alg.x_prec = alg.x_current
        alg.f_prec = alg.f_current
        alg.f_obj_prec = alg.f_obj_current
        alg.x_current = x
        alg.f_current = f
        alg.f_obj_current = f_obj
        if problem.m > 0
            alg.g_prec = alg.g_current
            alg.g_current = g
            if alg.params.penalty_approach && alg.penalty.interior
                alg.penalty.gmin = minimum(abs.(alg.g_current[.!alg.penalty.slacks]))
            end
        end
    end

    if !match
        add_to_cache!(cache, problem, x, xnorm, f_obj, g, f, success)
    else
        move_top!(cache, index_match, problem.m, success)
    end

    return x, success
end

function minimize!(alg::SidPsmAlgorithm, problem::Problem; iprint::Int=0)
    cache = CacheUtility(alg.params.tol_alfa, problem.n, problem.m)

    if !linear_feas(problem.linear_domain, problem.x0)
        println("Initial point provided violates the linear constraints.")
        return alg.DFO_DATA, alg.VIOL_DATA, cache
    end

    feasible, f_obj_0, g_0, f_m_0, x0 = initialization!(alg, problem, cache)
    alg.f_obj_current = f_obj_0
    alg.f_current = f_m_0
    alg.f_obj_prec = alg.f_obj_current
    alg.f_prec = alg.f_current
    if problem.m > 0
        alg.g_current = g_0
        alg.g_prec = alg.g_current
        if alg.params.penalty_approach && alg.penalty.interior
            alg.penalty.gmin = minimum(abs.(alg.g_current[.!alg.penalty.slacks]))
        end
    end

    if !feasible
        return alg.DFO_DATA, alg.VIOL_DATA, cache
    end

    alg.x_current = x0
    alg.x_prec = x0

    local s_min::Function, s_max::Function, p_max::Function
    if alg.params.shessian != 0
        if alg.params.store_all
            s_min = n -> 2n + 1
            s_max = n -> 2n + 1
            p_max = n -> 8 * (n + 1)
        else
            s_min = n -> n
            s_max = n -> 2n + 1
            p_max = n -> 4 * (n + 1)
        end
    else
        if alg.params.store_all
            s_min = n -> n + 1
            s_max = n -> n + 1
            p_max = n -> (n + 1) * (n + 2)
        else
            s_min = n -> floor(Int, (n + 1) / 2)
            s_max = n -> n + 1
            p_max = n -> 2 * (n + 1)
        end
    end

    D = gen!(alg.gen, problem, alg.x_current)

    poised = 0
    halt = false
    alg.store_fail_count = 0

    while !halt
        alg.func_iter = 0
        success = false
        succ_trust_linear = false
        ball_distance = 1.0
        d_trust = zeros(problem.n)
        g_lin = Float64[]
        H_quad = zeros(0, 0)

        if alg.params.search_option != 0
            if alg.trust_succ == 0 && alg.search_dec_fail != 1
                d_trust, model_dec, ball_distance = search_step!(alg.search_step, alg, cache, problem, p_max)
                xtemp = alg.x_current + d_trust

                if sum(abs.(d_trust)) > 1e-2 * alg.alfa
                    xtemp, success = evaluate_point!(alg, problem, cache, xtemp)
                end

                if success
                    alg.search_dec_fail = model_dec <= 1e-2 * (alg.f_prec - alg.f_current) ? 1 : 0
                else
                    alg.search_dec_fail = 0
                end
            end

            if (!success && !alg.search_step.only_succ) || (alg.trust_succ != 0 && alg.search_dec_fail != 2)
                alg.search_step.only_succ = true
                d_trust, model_dec, ball_distance = search_step!(alg.search_step, alg, cache, problem, p_max)
                alg.search_step.only_succ = false
                xtemp = alg.x_current + d_trust

                if norm(d_trust) > 1e-2 * alg.alfa
                    xtemp, success = evaluate_point!(alg, problem, cache, xtemp)
                end

                if success
                    alg.trust_succ = 1
                    alg.search_dec_fail = model_dec <= 1e-2 * (alg.f_prec - alg.f_current) ? 2 : 0
                else
                    alg.trust_succ = 0
                    alg.search_dec_fail = 0
                end
            end

            if success && all(abs.(d_trust) .== 0)
                exp_success = true
                d_trust_exp = copy(d_trust)
                xtemp = alg.x_current + d_trust
                while exp_success && alg.func_eval <= alg.params.fevals_max
                    d_trust_exp = d_trust_exp .* 2
                    xtemp = xtemp .+ d_trust_exp
                    xtemp, exp_success = evaluate_point!(alg, problem, cache, xtemp)
                end
            end
        end
        if success
            alg.trust_succ = 1
        end

        poised, gvec, H, Y, Delta = simplex_phase!(alg.simplex, alg, problem, cache, p_max, s_min, s_max)
        g_lin = gvec
        H_quad = H

        if poised > 0
            xtemp = alg.x_current + gvec

            if sum(abs.(gvec)) > cache.tol_match
                xtemp, success = evaluate_point!(alg, problem, cache, xtemp)
            end

            if success
                ball_distance = (Delta - norm(gvec)) / Delta
                succ_trust_linear = true
            end

            if alg.params.stop_grad == 2
                nsimp = size(Y, 2)
                dir = Y[:, 2:nsimp] .- Y[:, 1]
                sg_dir_deriv = dir' * gvec
                if all(sg_dir_deriv .>= -alg.params.tol_grad)
                    halt = true
                end
            else
                if alg.params.stop_grad == 1 && maximum(abs.(Delta .* gvec)) <= alg.params.tol_grad
                    halt = true
                end
            end
        end

        if !success
            alg.gen.L += 1
            if alg.store_fail_count > 0
                alg.iter_uns += 1
            end

            if alg.gen.dense
                D = gen!(alg.gen, problem, alg.x_current)
            end

            if poised > 0
                D = order_D!(alg.gen, alg, D, g_lin, H_quad, poised)
            end

            _, success = poll_step!(alg.poll_step, alg, problem, cache, D)
        else
            alg.iter_suc += 1
        end

        alg.iter += 1

        poised_data = if !(alg.params.mesh_option in (0, 3))
            poised == 0 ? (poised,) : poised == 1 ? (poised, g_lin) : (poised, g_lin, H_quad)
        else
            (poised,)
        end
        alg.alfa = mesh_proc(alg.poll_step, alg, success, poised_data, ball_distance, succ_trust_linear)

        if alg.params.penalty_approach
            update_penalty = update_penalty_parameter!(alg.penalty, alg, cache, problem, success)

            if !update_penalty && alg.switch_strategy && success
                switch_penalty!(alg.penalty, alg, problem, cache)
            end

            if update_penalty && alg.params.centralpath_option
                d_central = alg.x_current - alg.penalty.x_central_path
                alg.penalty.x_central_path = alg.x_current
                xtemp = copy(alg.x_current)
                alfa_cp = 0.1
                cp_success = true
                if sum(abs.(d_central)) > cache.tol_match
                    while cp_success && alg.func_eval <= alg.params.fevals_max
                        xtemp = xtemp .+ alfa_cp .* d_central
                        alfa_cp *= 2
                        xtemp, cp_success = evaluate_point!(alg, problem, cache, xtemp)
                    end
                end
            end
        end

        if alg.params.stop_alfa && alg.alfa < alg.params.tol_alfa
            halt = true
        end
        if alg.params.stop_fevals && alg.func_eval >= alg.params.fevals_max
            halt = true
        end
        if alg.params.stop_iter && alg.iter >= alg.params.iter_max
            halt = true
        end

        if iprint > 0
            _printf_sidpsm(alg)
        end
    end

    s_data = length(alg.DFO_DATA)
    if s_data < alg.params.fevals_max
        append!(alg.DFO_DATA, fill(Inf, alg.params.fevals_max - s_data))
        if problem.m > 0
            append!(alg.VIOL_DATA, fill(Inf, alg.params.fevals_max - s_data))
        end
    elseif s_data > alg.params.fevals_max
        alg.DFO_DATA = alg.DFO_DATA[1:alg.params.fevals_max]
        if problem.m > 0
            alg.VIOL_DATA = alg.VIOL_DATA[1:alg.params.fevals_max]
        end
    end

    return alg.DFO_DATA, alg.VIOL_DATA, cache
end

function _printf_sidpsm(alg::SidPsmAlgorithm)
    println("feval=$(alg.func_eval)  f=$(alg.f_current)  f_obj=$(alg.f_obj_current)  alfa=$(alg.alfa)")
end

end # module
