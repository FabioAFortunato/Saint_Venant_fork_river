# ==============================================================================
# 1. Dependências
#
# Reaproveita de `ffjm2.jl` o avaliador do modelo quártico, a resolução do
# subproblema irrestrito por multi-start (`_ffjm2_model_direction`) e a
# atualização quase-Newton por modelo (`_ffjm2_update!`, que já sabe fazer
# `:bfgs`, `:sr1`, `:is_bfgs`, `:dw_model`, `:is_dw_model` e `:psb`).
# ==============================================================================

using ForwardDiff
using LinearAlgebra
using LineSearches
using Random

if !isdefined(@__MODULE__, :_ffjm2_model_direction)
    include("ffjm2.jl")
end

const FFJM2_ENSEMBLE_PURO_MODELS = (:bfgs, :sr1, :is_bfgs, :dw_model, :is_dw_model, :psb)

# ==============================================================================
# 2. Método externo FFJM2 puro (busca linear de Armijo) com conjunto de
# modelos quase-Newton
#
# Mesma ideia de `ffjm2_ensemble`/`ffjm2_ensemble_trust`: mantém um conjunto
# de Hessianas H[i] em paralelo, uma por modelo quase-Newton de `models`. A
# cada iteração externa k>0:
#
#   (a) resolve o subproblema quártico irrestrito (multi-start gaussiano em
#       torno de zero, ver `_ffjm2_model_direction`) uma vez para cada
#       conjunto de Hessianas, obtendo uma direção candidata por modelo;
#   (b) a direção usada é a de maior descida entre as candidatas, isto é, a
#       que minimiza ∇f(xₖ)ᵀd (mais negativa);
#   (c) essa direção passa pela mesma condição (2) + busca linear de Armijo
#       de `ffjm2` (bola pesada como alternativa antes de cair em -∇f(xₖ));
#   (d) o passo aceito atualiza todos os conjuntos de Hessianas, cada um com
#       sua própria fórmula, usando o mesmo par (s, y).
#
# Custa aproximadamente `length(models)` vezes o tempo de subproblema de
# `ffjm2` por iteração externa. É uma variante experimental para comparar o
# quanto os modelos concordam entre si (ver `last_model_directions` no
# retorno), não uma tentativa de ser mais rápida.
# ==============================================================================

"""
    ffjm2_ensemble_puro(F, x0; models=FFJM2_ENSEMBLE_PURO_MODELS, jacobian=nothing, kwargs...)

Variante de [`ffjm2`](@ref) que, em vez de um único `update`, mantém um
conjunto `models` (por padrão os 6 modelos de `_ffjm2_update!`: `:bfgs`,
`:sr1`, `:is_bfgs`, `:dw_model`, `:is_dw_model`, `:psb`) de Hessianas `H[i]`
em paralelo. A cada iteração, o subproblema quártico irrestrito (multi-start
gaussiano em torno de zero) é resolvido uma vez por modelo, e a direção usada
é a de maior descida entre as candidatas, ou seja, a que minimiza
`∇f(xₖ)ᵀd` (mais negativa).

A aceitação da direção (condição (2) do artigo + busca linear de Armijo, com
bola pesada como alternativa antes de cair em `-∇f(xₖ)`) segue exatamente
[`ffjm2`](@ref). Cada conjunto de Hessianas é atualizado com sua própria
fórmula (via `_ffjm2_update!`), usando o mesmo passo aceito e a mesma
diferença de Jacobiana para todos.

O retorno segue o mesmo formato de [`ffjm2`](@ref), exceto que `hessians` é
um `Dict{Symbol}` (uma entrada por modelo em vez de um único vetor) e há dois
campos extras: `last_model_directions` (`Dict{Symbol}` com a última direção
individual de cada modelo, antes da seleção) e `direction_source` (o modelo
vencedor na última iteração, ou `:gradient`/`:heavy_ball` se a condição (2)
rejeitou todos os candidatos).
"""
function ffjm2_ensemble_puro(
    F,
    x0::AbstractVector;
    models::Tuple = FFJM2_ENSEMBLE_PURO_MODELS,
    jacobian = nothing,
    maxiter::Integer = 1000,
    g_tol::Real = 1e-8,
    residual_rms_tol::Union{Nothing,Real} = 0.0,
    x_tol::Real = 0.0,
    f_rel_tol::Union{Nothing,Real} = 0.0,
    model_maxiter::Integer = 1000,
    model_g_tol::Real = 1e-6,
    model_multistart::Integer = 10,
    model_start_spread::Real = 1.0,
    model_seed::Integer = 1234,
    model_solver::Symbol = :ipopt,
    direction_theta::Real = 1e-4,
    direction_beta::Real = 1e-8,
    direction_delta::Real = Inf,
    heavy_ball_beta = 0.1,
    gamma_bar::Real = 10.0,
    initial_scaling::Bool = false,
    update_tol::Real = sqrt(eps(Float64)),
    linesearch = LineSearches.BackTracking(order = 3),
    callback = nothing,
    show_trace::Bool = false,
)
    # --------------------------------------------------------------------------
    # 2.1. Validação dos parâmetros
    # --------------------------------------------------------------------------
    start_time_ns = time_ns()
    !isempty(models) || throw(ArgumentError("models não pode ser vazio"))
    all(model -> model in (:bfgs, :sr1, :is_bfgs, :dw_model, :is_dw_model, :psb), models) ||
        throw(ArgumentError(
            "cada elemento de models deve ser :bfgs, :sr1, :is_bfgs, :dw_model, :is_dw_model ou :psb",
        ))
    gamma_bar > 1 || throw(ArgumentError("gamma_bar deve ser maior que 1"))
    model_solver in (:ipopt, :bfgs, :bobyqa, :mads) ||
        throw(ArgumentError("model_solver deve ser :ipopt, :bfgs, :bobyqa ou :mads"))
    maxiter >= 0 || throw(ArgumentError("maxiter deve ser não negativo"))
    model_multistart >= 1 ||
        throw(ArgumentError("model_multistart deve ser pelo menos 1"))
    isfinite(model_start_spread) && model_start_spread >= 0 ||
        throw(ArgumentError("model_start_spread deve ser finito e não negativo"))
    0 < direction_theta < 1 ||
        throw(ArgumentError("direction_theta deve pertencer a (0, 1)"))
    direction_beta > 0 ||
        throw(ArgumentError("direction_beta deve ser positivo"))
    direction_delta > 0 ||
        throw(ArgumentError("direction_delta deve ser positivo"))
    heavy_ball_beta === nothing || (isfinite(heavy_ball_beta) && heavy_ball_beta >= 0) ||
        throw(ArgumentError("heavy_ball_beta deve ser nothing ou não negativo"))
    x_tol >= 0 || throw(ArgumentError("x_tol deve ser não negativo"))
    f_rel_tol === nothing || f_rel_tol >= 0 ||
        throw(ArgumentError("f_rel_tol deve ser nothing ou não negativo"))

    # --------------------------------------------------------------------------
    # 2.2. Resíduos e Jacobiana no ponto inicial
    # --------------------------------------------------------------------------
    x = collect(float.(x0))
    dim = length(x)
    function_evaluations = Ref(0)
    gradient_evaluations = Ref(0)
    function_evaluation_time_seconds = Ref(0.0)
    gradient_evaluation_time_seconds = Ref(0.0)

    function raw_residual(x)
        values = F(x)
        values isa AbstractVector ||
            throw(ArgumentError("F(x) deve devolver um vetor de resíduos"))
        isempty(values) &&
            throw(ArgumentError("o vetor de resíduos não pode ser vazio"))
        R = mapreduce(typeof, promote_type, values)
        R <: Real ||
            throw(ArgumentError("todos os resíduos devem ser números reais"))
        return collect(R, values)
    end

    function residual(x)
        start_ns = time_ns()
        values = raw_residual(x)
        function_evaluations[] += 1
        function_evaluation_time_seconds[] += (time_ns() - start_ns) / 1e9
        return values
    end
    r = residual(x)

    initial_rms = norm(r) / sqrt(length(r))
    if show_trace
        println(
            "ffjm2_ensemble_puro iter 0: ",
            "f = $(0.5 * dot(r, r)), RMSD = $initial_rms (antes da Jacobiana)",
        )
    end

    jacobian_config = jacobian === nothing ?
        ForwardDiff.JacobianConfig(raw_residual, x, ForwardDiff.Chunk{dim}()) :
        nothing
    function jac(x)
        start_ns = time_ns()
        Jx = jacobian === nothing ?
            ForwardDiff.jacobian(raw_residual, x, jacobian_config) :
            Matrix(jacobian(x))
        gradient_evaluations[] += 1
        gradient_evaluation_time_seconds[] += (time_ns() - start_ns) / 1e9
        return Jx
    end

    J = jac(x)
    m, n = size(J)
    length(r) == m || throw(DimensionMismatch("F e sua Jacobiana são incompatíveis"))
    length(x) == n || throw(DimensionMismatch("x0 e a Jacobiana são incompatíveis"))
    all(isfinite, r) && all(isfinite, J) ||
        throw(ArgumentError("F(x0) e sua Jacobiana devem ser finitas"))

    T = promote_type(eltype(x), eltype(r), eltype(J), Float64)
    x = T.(x)
    r = T.(r)
    J = T.(J)
    H = Dict(model => [zeros(T, n, n) for _ in 1:m] for model in models)
    f = T(0.5) * dot(r, r)
    g = J' * r
    status = :maximum_iterations
    iterations = 0
    rejected_directions = 0
    model_solves = 0
    model_iterations = 0
    model_solve_time_seconds = 0.0
    alphas = T[]
    last_alpha = one(T)
    last_backtrackings = 0
    last_direction_rejected = false
    last_direction_norm = zero(T)
    last_direction_source = :model
    heavy_ball_used = 0
    previous_step = nothing
    last_model_directions = Dict{Symbol,Vector{T}}()

    # --------------------------------------------------------------------------
    # 2.3. Laço principal do método externo
    # --------------------------------------------------------------------------
    for k in 0:maxiter
        # 2.3.1. Critérios de parada no início da iteração.
        gnorm = norm(g)
        residual_rms = norm(r) / sqrt(length(r))
        if show_trace && k > 0
            println(
                "ffjm2_ensemble_puro iter $k: ",
                "f = $f, RMSD = $residual_rms, ",
                "α = $last_alpha, backtrackings = $last_backtrackings, ",
                "direção rejeitada = $last_direction_rejected, ",
                "fonte da direção = $last_direction_source",
            )
        end
        state = (; iteration = k, x = copy(x), value = f, residual = copy(r),
                 residual_rms, gradient = copy(g), gradient_norm = gnorm,
                 alpha = k == 0 ? nothing : last_alpha,
                 backtrackings = k == 0 ? 0 : last_backtrackings,
                 direction_rejected = k == 0 ? false : last_direction_rejected,
                 direction_norm = k == 0 ? nothing : last_direction_norm,
                 direction_source = last_direction_source)
        if callback !== nothing && callback(state) === true
            status = :callback
            iterations = k
            break
        end
        if residual_rms_tol !== nothing && residual_rms <= residual_rms_tol
            status = :residual_converged
            iterations = k
            break
        end
        if gnorm <= g_tol
            status = :gradient_converged
            iterations = k
            break
        end
        if k == maxiter
            iterations = k
            break
        end

        # 2.3.2. Direção: gradiente negativo em k=0; melhor descida entre os
        # modelos para k>0 (menor ∇f(xₖ)ᵀd entre as direções candidatas).
        direction_source = k == 0 ? :gradient : :model
        if k == 0
            p = -g
        else
            best_directional_derivative = T(Inf)
            best_direction = nothing
            for model in models
                model_result = _ffjm2_model_direction(
                    r,
                    J,
                    H[model],
                    last_direction_norm,
                    model_maxiter,
                    model_g_tol,
                    model_multistart,
                    model_start_spread,
                    model_seed + k,
                    model_solver,
                )
                model_solves += 1
                model_iterations += model_result.total_iterations
                model_solve_time_seconds += model_result.solve_time_seconds
                last_model_directions[model] = model_result.direction
                if show_trace
                    println(
                        "  Subproblema ($model): ", model_result.stop_reason,
                        " | Mₖ(x̄) = ", model_result.minimum,
                    )
                end
                dtg = dot(g, model_result.direction)
                if all(isfinite, model_result.direction) && dtg < best_directional_derivative
                    best_directional_derivative = dtg
                    best_direction = model_result.direction
                    direction_source = model
                end
            end
            if best_direction === nothing
                p = -g
                direction_source = :gradient
            else
                p = best_direction
            end
        end

        # 2.3.3. Aceitação da direção pela condição (2) do Algoritmo 2.1.
        gnorm = norm(g)
        direction_valid(d, dn) = all(isfinite, d) &&
            dot(g, d) <= -direction_theta * dn^2 * gnorm^2 &&
            direction_beta * gnorm <= dn <= direction_delta
        direction_rejected = !direction_valid(p, norm(p))
        if direction_rejected
            rejected_directions += 1
            p = -g
            direction_source = :gradient
            # Extensão fora do artigo: em vez de aceitar -g cegamente (como no
            # pseudocódigo da Seção 4), tenta antes a direção da bola pesada
            # (-g + β·passo_anterior), que também precisa satisfazer a
            # condição (2) — se falhar, cai em -g puro exatamente como antes.
            if heavy_ball_beta !== nothing && previous_step !== nothing
                d_heavy_ball = p .+ heavy_ball_beta .* previous_step
                if direction_valid(d_heavy_ball, norm(d_heavy_ball))
                    p = d_heavy_ball
                    direction_source = :heavy_ball
                    heavy_ball_used += 1
                end
            end
        end
        last_direction_rejected = direction_rejected
        last_direction_source = direction_source
        last_direction_norm = norm(p)

        # 2.3.4. Busca linear de Armijo na função objetivo verdadeira.
        line_search_evaluations = Ref(0)
        ϕ(α) = begin
            line_search_evaluations[] += 1
            rα = residual(x .+ α .* p)
            T(0.5) * dot(rα, rα)
        end
        dϕ0 = dot(g, p)
        α, fnew = try
            linesearch(ϕ, one(T), f, dϕ0)
        catch err
            if err isa LineSearches.LineSearchException
                status = :line_search_failed
                iterations = k
                break
            end
            rethrow()
        end
        last_alpha = α
        push!(alphas, α)
        last_backtrackings = max(line_search_evaluations[] - 1, 0)

        # 2.3.5. Avaliação verdadeira no ponto aceito pela busca linear.
        xnew = x .+ α .* p
        external_evaluation_start_ns = time_ns()
        rnew = T.(residual(xnew))
        fnew = T(0.5) * dot(rnew, rnew)
        external_evaluation_time_seconds =
            (time_ns() - external_evaluation_start_ns) / 1e9
        if show_trace
            println(
                "  Avaliação externa f(x): ",
                round(external_evaluation_time_seconds; digits = 6), " s",
            )
        end

        # 2.3.6. Atualização de cada conjunto de Hessianas, e do estado externo.
        Jnew = T.(jac(xnew))
        s = xnew - x
        if initial_scaling && k == 0
            alpha0 = norm(s) / gnorm
            for model in models
                for Hi in H[model]
                    Hi .= alpha0 .* Matrix{T}(I, n, n)
                end
            end
        end
        for model in models
            _ffjm2_update!(H[model], s, Jnew, J, model, update_tol, gamma_bar)
        end
        previous_step = s

        fold = f
        x, r, J, f = xnew, rnew, Jnew, T(fnew)
        g = J' * r
        iterations = k + 1
        if norm(s) <= x_tol * max(one(T), norm(x))
            status = :step_converged
            break
        end
        if f_rel_tol !== nothing &&
           abs(fold - f) <= f_rel_tol * max(one(T), abs(fold))
            status = :function_converged
            break
        end
    end

    # --------------------------------------------------------------------------
    # 2.4. Resultado do método externo
    # --------------------------------------------------------------------------
    converged = status in (
        :residual_converged,
        :gradient_converged,
        :step_converged,
        :function_converged,
    )
    total_function_evaluation_time_seconds = function_evaluation_time_seconds[]
    total_gradient_evaluation_time_seconds = gradient_evaluation_time_seconds[]
    mean_function_evaluation_time_seconds = function_evaluations[] == 0 ? 0.0 :
        total_function_evaluation_time_seconds / function_evaluations[]
    mean_gradient_evaluation_time_seconds = gradient_evaluations[] == 0 ? 0.0 :
        total_gradient_evaluation_time_seconds / gradient_evaluations[]
    return (;
        minimizer = x,
        minimum = f,
        residual = r,
        gradient = g,
        hessians = H,
        models,
        iterations,
        converged,
        status,
        execution_time_seconds = (time_ns() - start_time_ns) / 1e9,
        function_evaluations = function_evaluations[],
        gradient_evaluations = gradient_evaluations[],
        function_evaluation_time_seconds = mean_function_evaluation_time_seconds,
        gradient_evaluation_time_seconds = mean_gradient_evaluation_time_seconds,
        total_function_evaluation_time_seconds,
        total_gradient_evaluation_time_seconds,
        rejected_directions,
        heavy_ball_used,
        alphas,
        model_solver,
        model_solves,
        model_iterations,
        model_solve_time_seconds,
        last_model_directions,
        direction_source = last_direction_source,
    )
end
