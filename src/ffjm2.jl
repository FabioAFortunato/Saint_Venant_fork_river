# ==============================================================================
# 1. Dependências
# ==============================================================================

using ForwardDiff
using Ipopt
using LinearAlgebra
using LineSearches
using NLopt
using NOMAD
using OptimizationBase
using OptimizationLBFGSB
using Optim
using Random
import MathOptInterface as MOI

if !isdefined(@__MODULE__, :sr1_backtracking) ||
   !isdefined(@__MODULE__, :bfgs_backtracking)
    include("sr1_bfgs_backtracking.jl")
end

if !isdefined(@__MODULE__, :spg_box)
    include("teste2.jl")
end

struct GeometricBackTracking{T} <: LineSearches.AbstractLineSearch
    c1::T
    rho::T
    iterations::Int
    min_alpha::T
end

GeometricBackTracking(;
    c1::Real = 1e-4,
    rho::Real = 0.5,
    iterations::Integer = 1000,
    min_alpha::Real = 1e-12,
) = begin
    c1_value, rho_value, min_alpha_value =
        promote(float(c1), float(rho), float(min_alpha))
    GeometricBackTracking(c1_value, rho_value, Int(iterations), min_alpha_value)
end

function (linesearch::GeometricBackTracking)(
    objective,
    x::AbstractArray{T},
    direction::AbstractArray{T},
    initial_alpha,
    x_new::AbstractArray{T},
    initial_value,
    initial_slope,
    alpha_max = typemax(real(T)),
) where {T}
    phi, _ = LineSearches.make_ϕ_dϕ(objective, x_new, x, direction)
    alpha = min(initial_alpha, alpha_max)
    for _ in 0:linesearch.iterations
        value = phi(alpha)
        if isfinite(value) &&
           value <= initial_value + linesearch.c1 * alpha * initial_slope
            return alpha, value
        end
        alpha *= linesearch.rho
        if alpha < linesearch.min_alpha
            # `Optim.BFGS`'s `update_state!` aplica `state.x += state.alpha*state.s`
            # incondicionalmente, mesmo quando a busca linear lança esta exceção
            # (ele só lê `ex.alpha` de volta e segue em frente) — ver
            # `perform_linesearch!`/`update_state!` em Optim.jl. Devolver o
            # último `alpha` tentado (não nulo) faria esse passo, que nunca
            # satisfez Armijo, ser silenciosamente aceito como resultado final.
            # Lançar com `alpha = 0` torna esse passo forçado um no-op.
            throw(LineSearches.LineSearchException(
                "Backtracking geométrico atingiu o alpha mínimo sem satisfazer Armijo.",
                zero(alpha),
            ))
        end
    end
    throw(LineSearches.LineSearchException(
        "Backtracking geométrico atingiu o limite de iterações sem satisfazer Armijo.",
        zero(alpha),
    ))
end

# ==============================================================================
# 2. Método externo FFJM2
#
# Fluxo de uma iteração:
#   (a) verifica convergência;
#   (b) constrói e resolve o modelo quártico;
#   (c) valida a direção pela condição (2) do artigo;
#   (d) escolhe o tamanho do passo por Armijo;
#   (e) atualiza as Hessianas individuais por BFGS ou SR1.
# ==============================================================================

"""
    ffjm2(F, x0; update=:bfgs, jacobian=nothing, kwargs...)

Minimiza `1/2 * ||F(x)||²` pelo método descrito em (19)--(21) do
documento `teste1.pdf`. Uma matriz `H[i]` aproxima a Hessiana de cada
componente `F(x)[i]`; `update` pode ser `:bfgs` ou `:sr1`.
Cada subproblema do modelo é formulado diretamente na direção `d` e resolvido
por Ipopt com estratégia multi-start. O primeiro ponto inicial é o vetor nulo e
os demais são perturbações gaussianas reprodutíveis ao redor dele.
O Ipopt recebe a Hessiana exata do modelo do subproblema, construída a partir
das aproximações `H[i]` mantidas pelo método externo.
O minimizador do modelo (19) é diretamente a direção externa `dₖ` e deve
satisfazer a condição (2); caso contrário, usa-se `-∇f(xₖ)`.
Como esta interface não recebe um conjunto proibido `P`, a busca linear
corresponde ao caso `P = ∅`, com `t_first = 1`.

`F` deve receber um vetor e devolver diretamente um vetor de resíduos.
Se `jacobian` não for fornecida, a Jacobiana é calculada com
`ForwardDiff.jacobian`, usando `ForwardDiff.Chunk{dim}()` para diferenciar
todas as `dim` variáveis juntas.

Os critérios de parada padrão são deliberadamente moderados para resíduos
produzidos por simulações: RMSD menor que `residual_rms_tol`, norma do
gradiente dividida pelo número de variáveis menor que `g_tol`, passo pequeno
ou redução relativa pequena da função objetivo.

O retorno é um `NamedTuple` com os campos `minimizer`, `minimum`,
`residual`, `gradient`, `hessians`, `iterations`, `converged`, `status`,
`execution_time_seconds`, `function_evaluations`, `gradient_evaluations`,
`function_evaluation_time_seconds`, `gradient_evaluation_time_seconds`,
`total_function_evaluation_time_seconds`,
`total_gradient_evaluation_time_seconds` e `rejected_directions`. Os campos
de tempo sem o prefixo `total_` representam o custo médio de uma chamada.
"""

function ffjm2(
    F,
    x0::AbstractVector;
    update::Symbol = :bfgs,
    jacobian = nothing,
    maxiter::Integer = 200,
    g_tol::Real = 1e-2,
    residual_rms_tol::Union{Nothing,Real} = 0.0,
    x_tol::Real = 1e-6,
    f_rel_tol::Union{Nothing,Real} = 1e-6,
    model_maxiter::Integer = 1000,
    model_g_tol::Real = 1e-3,
    model_multistart::Integer = 1000,
    model_start_spread::Real = 1.0,
    model_seed::Integer = 1234,
    model_solver::Symbol = :ipopt,
    direction_theta::Real = 1e-4,
    direction_beta::Real = 1e-8,
    direction_delta::Real = Inf,
    update_tol::Real = sqrt(eps(Float64)),
    linesearch = LineSearches.BackTracking(order = 3),
    callback = nothing,
    show_trace::Bool = false,
)
    # --------------------------------------------------------------------------
    # 2.1. Validação dos parâmetros
    # --------------------------------------------------------------------------
    start_time_ns = time_ns()
    update in (:bfgs, :sr1) ||
        throw(ArgumentError("update deve ser :bfgs ou :sr1"))
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

        # Algumas simulações do projeto constroem `.erro` a partir de `[]`,
        # produzindo Vector{Any}. Recupera aqui o tipo numérico concreto,
        # tanto para Float64 quanto para ForwardDiff.Dual.
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

    # Mostra o RMSD da avaliação real antes da simulação com números Dual.
    initial_rms = norm(r) / sqrt(length(r))
    if show_trace
        println(
            "ffjm2 ($(uppercase(string(update)))) iter 0: ",
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
    H = [zeros(T, n, n) for _ in 1:m]
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

    # --------------------------------------------------------------------------
    # 2.3. Laço principal do método externo
    # --------------------------------------------------------------------------
    for k in 0:maxiter
        # 2.3.1. Critérios de parada no início da iteração.
        gnorm = norm(g)
        residual_rms = norm(r) / sqrt(length(r))
        if show_trace && k > 0
            println(
                "ffjm2 ($(uppercase(string(update)))) iter $k: ",
                "f = $f, RMSD = $residual_rms, ",
                "α = $last_alpha, backtrackings = $last_backtrackings, ",
                "direção rejeitada = $last_direction_rejected",
            )
        end
        state = (; iteration = k, x = copy(x), value = f, residual = copy(r),
                 residual_rms, gradient = copy(g), gradient_norm = gnorm,
                 alpha = k == 0 ? nothing : last_alpha,
                 backtrackings = k == 0 ? 0 : last_backtrackings,
                 direction_rejected = k == 0 ? false : last_direction_rejected,
                 direction_norm = k == 0 ? nothing : last_direction_norm)
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

        # 2.3.2. Direção: gradiente negativo em k=0; modelo quártico para k>0.
        model_result = nothing
        if k == 0
            p = -g
        else
            model_result = _ffjm2_model_direction(
                r,
                J,
                H,
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
            p = model_result.direction
        end

        # 2.3.3. Aceitação da direção pela condição (2) do Algoritmo 2.1.
        pnorm = norm(p)
        gnorm = norm(g)
        direction_rejected = !all(isfinite, p) || !(
            dot(g, p) <= -direction_theta * pnorm^2 * gnorm^2 &&
            direction_beta * gnorm <= pnorm <= direction_delta
        )
        if direction_rejected
            rejected_directions += 1
            p = -g
        end
        last_direction_rejected = direction_rejected
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
            print(
                "  Avaliação externa f(x): ",
                round(external_evaluation_time_seconds; digits = 6),
                " s",
            )
            if model_result !== nothing
                print(
                    " | Ipopt multi-start interno: ", model_result.stop_reason,
                    " | melhor início = ", model_result.best_start,
                    "/", model_result.starts,
                    " | iterações = ", model_result.iterations,
                    " | Mₖ(x̄) = ", model_result.minimum,
                    " | tempo total do subproblema = ",
                    round(model_result.solve_time_seconds; digits = 6),
                    " s",
                )
            end
            println()
        end
        # 2.3.6. Atualização das Hessianas individuais e do estado externo.
        Jnew = T.(jac(xnew))
        s = xnew - x
        _ffjm2_update!(H, s, Jnew, J, update, update_tol)

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
        alphas,
        model_solver,
        model_solves,
        model_iterations,
        model_solve_time_seconds,
    )
end

# ==============================================================================
# 3. Atualização quase-Newton das Hessianas dos resíduos
#
# Para cada resíduo f_i, H[i] aproxima ∇²f_i. A atualização utiliza
# s = x_{k+1} - x_k e y_i = ∇f_i(x_{k+1}) - ∇f_i(x_k).
# ==============================================================================

function _ffjm2_update!(H, s, Jnew, Jold, update, tol)
    ss = norm(s)
    for i in eachindex(H)
        Hi = H[i]
        y = vec(Jnew[i, :] - Jold[i, :])
        if update === :bfgs
            sy = dot(s, y)
            Hs = Hi * s
            sHs = dot(s, Hs)
            if abs(sy) > tol * max(one(sy), ss * norm(y))
                Hi .+= (y * y') / sy
            end
            if abs(sHs) > tol * max(one(sHs), ss * norm(Hs))
                Hi .-= (Hs * Hs') / sHs
            end
        else
            v = y - Hi * s
            vs = dot(v, s)
            if abs(vs) > tol * max(one(vs), norm(v) * ss)
                Hi .+= (v * v') / vs
            end
        end
        # Elimina assimetria de arredondamento acumulada.
        Hi .= (Hi .+ Hi') ./ 2
    end
    return H
end

# ==============================================================================
# 4. Modelo quártico e interface de derivadas para o Ipopt
#
# O Ipopt trabalha com a variável absoluta x. Internamente usamos d = x - xk:
#
#   q_i(x) = r_i + J_i*d + 1/2*d'*H_i*d
#   M_k(d) = 1/2 * Σ_i q_i(d)^2.
#
# O avaliador fornece ao Ipopt o valor, o gradiente e a Hessiana exata de M_k.
# ==============================================================================

struct _FFJM2IpoptEvaluator{T} <: MOI.AbstractNLPEvaluator
    r::Vector{T}
    J::Matrix{T}
    H::Vector{Matrix{T}}
end

# Recursos analíticos disponibilizados ao Ipopt.
MOI.features_available(::_FFJM2IpoptEvaluator) = [:Grad, :Hess]

function MOI.initialize(::_FFJM2IpoptEvaluator, requested_features)
    all(feature -> feature in (:Grad, :Hess), requested_features) ||
        throw(ArgumentError("recurso de derivada não suportado pelo subproblema"))
    return nothing
end

# Calcula simultaneamente q_i(d) e ∇q_i(d), armazenado nas linhas de A.
function _ffjm2_model_components(evaluator::_FFJM2IpoptEvaluator, d)
    T = promote_type(eltype(evaluator.r), eltype(d))
    q = similar(evaluator.r, T)
    A = similar(evaluator.J, T)
    for i in eachindex(evaluator.r)
        Hi_d = evaluator.H[i] * d
        q[i] = evaluator.r[i] + dot(view(evaluator.J, i, :), d) + dot(d, Hi_d) / 2
        view(A, i, :) .= view(evaluator.J, i, :) .+ Hi_d
    end
    return q, A
end

# Valor do modelo quártico M_k(d).
function MOI.eval_objective(evaluator::_FFJM2IpoptEvaluator, d)
    q, _ = _ffjm2_model_components(evaluator, d)
    return dot(q, q) / 2
end

# Gradiente exato: ∇M_k(d) = A' * q.
function MOI.eval_objective_gradient(evaluator::_FFJM2IpoptEvaluator, gradient, d)
    q, A = _ffjm2_model_components(evaluator, d)
    mul!(gradient, A', q)
    return nothing
end

# O Ipopt solicita apenas a parte triangular inferior da Hessiana.
function MOI.hessian_lagrangian_structure(evaluator::_FFJM2IpoptEvaluator)
    n = size(evaluator.J, 2)
    return [(i, j) for i in 1:n for j in 1:i]
end

# Hessiana exata: ∇²M_k(d) = A'A + Σ_i q_i(d)H_i.
function MOI.eval_hessian_lagrangian(
    evaluator::_FFJM2IpoptEvaluator,
    values,
    d,
    objective_factor,
    constraint_multipliers,
)
    isempty(constraint_multipliers) ||
        throw(ArgumentError("o subproblema interno deve ser irrestrito"))
    q, A = _ffjm2_model_components(evaluator, d)
    hessian = A' * A
    for i in eachindex(q)
        hessian .+= q[i] .* evaluator.H[i]
    end
    index = 1
    for i in axes(hessian, 1), j in 1:i
        values[index] = objective_factor * hessian[i, j]
        index += 1
    end
    return nothing
end

# ==============================================================================
# 5. Solução multi-start do subproblema quártico
#
# O primeiro Ipopt começa em d = 0. Os demais começam em perturbações
# gaussianas ao redor de zero. A melhor solução local é diretamente d_k.
# ==============================================================================

function _ffjm2_model_direction(
    r,
    J,
    H,
    maxiter,
    g_tol,
    multistart,
    start_spread,
    seed,
    solver,
)
    # 6.1. Congela os dados do modelo construído na iteração externa k.
    n = size(J, 2)
    evaluator = _FFJM2IpoptEvaluator(collect(r), Matrix(J), H)
    rng = MersenneTwister(seed)
    starts = Vector{Vector{eltype(r)}}(undef, Int(multistart))
    starts[1] = zeros(eltype(r), n)
    for i in 2:Int(multistart)
        starts[i] = starts[1] .+ start_spread .* randn(rng, eltype(r), n)
    end

    model_value(d) = MOI.eval_objective(evaluator, d)
    function model_gradient!(gradient, d)
        MOI.eval_objective_gradient(evaluator, gradient, d)
        return gradient
    end
    bound_radius = max(1.0e3, 100 * max(1.0, Float64(start_spread)))

    # 6.2. Resolve uma cópia do subproblema para cada ponto inicial.
    solve_start_ns = time_ns()
    results = map(starts) do initial
        if solver === :ipopt
            optimizer = Ipopt.Optimizer()
            MOI.set(optimizer, MOI.Silent(), true)
            MOI.set(optimizer, MOI.RawOptimizerAttribute("max_iter"), Int(maxiter))
            MOI.set(optimizer, MOI.RawOptimizerAttribute("tol"), Float64(g_tol))
            variables = MOI.add_variables(optimizer, n)
            MOI.set.(optimizer, MOI.VariablePrimalStart(), variables, initial)
            MOI.set(
                optimizer,
                MOI.NLPBlock(),
                MOI.NLPBlockData(MOI.NLPBoundsPair[], evaluator, true),
            )
            MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            MOI.optimize!(optimizer)
            minimizer = MOI.get.(optimizer, MOI.VariablePrimal(), variables)
            return (;
                minimizer,
                minimum = MOI.get(optimizer, MOI.ObjectiveValue()),
                iterations = MOI.get(optimizer, MOI.BarrierIterations()),
                status = MOI.get(optimizer, MOI.TerminationStatus()),
            )
        elseif solver === :bfgs
            options = Optim.Options(
                iterations = Int(maxiter),
                g_abstol = g_tol,
                show_trace = false,
            )
            result = Optim.optimize(model_value, model_gradient!, initial, Optim.BFGS(), options)
            return (;
                minimizer = copy(Optim.minimizer(result)),
                minimum = Optim.minimum(result),
                iterations = Optim.iterations(result),
                status = result.termination_code,
            )
        elseif solver === :bobyqa
            optimizer = NLopt.Opt(:LN_BOBYQA, n)
            optimizer.lower_bounds = fill(-bound_radius, n)
            optimizer.upper_bounds = fill(bound_radius, n)
            optimizer.xtol_abs = fill(max(Float64(g_tol), eps(Float64)), n)
            optimizer.maxeval = max(1, Int(maxiter))
            optimizer.min_objective = (d, _) -> model_value(d)
            minimum, minimizer, status = NLopt.optimize(optimizer, initial)
            return (;
                minimizer = collect(minimizer),
                minimum,
                iterations = optimizer.numevals,
                status,
            )
        else
            evaluations = Ref(0)
            best_value = Ref(Inf)
            best_point = copy(initial)
            objective = function (d)
                value = Float64(model_value(d))
                evaluations[] += 1
                if value < best_value[]
                    best_value[] = value
                    best_point .= d
                end
                return true, true, [value]
            end
            options = NOMAD.NomadOptions(
                display_degree = 0,
                max_bb_eval = max(1, Int(maxiter)),
            )
            problem = NOMAD.NomadProblem(
                n,
                1,
                ["OBJ"],
                objective,
                input_types = fill("R", n),
                lower_bound = fill(-bound_radius, n),
                upper_bound = fill(bound_radius, n),
                min_mesh_size = fill(max(Float64(g_tol), eps(Float64)), n),
                initial_mesh_size = fill(max(Float64(start_spread), 1e-2), n),
                options = options,
            )
            result = NOMAD.solve(problem, initial)
            minimizer = result.x_sol === nothing ? copy(best_point) : collect(result.x_sol)
            minimum = model_value(minimizer)
            return (; minimizer, minimum, iterations = evaluations[], status = result.status)
        end
    end
    # 6.3. Seleciona o menor valor encontrado entre as soluções locais.
    best_start = argmin(getproperty.(results, :minimum))
    result = results[best_start]
    total_iterations = sum(item.iterations for item in results)
    solve_time_seconds = (time_ns() - solve_start_ns) / 1e9

    return (
        direction = result.minimizer,
        model_minimizer = result.minimizer,
        stop_reason = string(result.status),
        iterations = result.iterations,
        total_iterations,
        minimum = result.minimum,
        solve_time_seconds,
        best_start,
        starts = length(starts),
        solver,
    )
end

# ==============================================================================
# 6. Rotina experimental de comparação
#
# Executa FFJM2-SR1 e três métodos quase-Newton de referência para
# uma única instância e grava as métricas em um arquivo TSV.
# ==============================================================================

"""
    comparar_ffjm2_bfgs(; kwargs...)

Compara `ffjm2(update=:sr1)`, `bfgs_puro_penalizado`, `bfgs_backtracking` e
`sr1_backtracking`. Salva SSE, RMSD, norma do gradiente da SSE, avaliações
externas, chamadas da função e do gradiente, tempos e solução em um arquivo
TSV.
"""
function comparar_ffjm2_bfgs(;
    simulation = sv_fork_assimilation,
    output = normpath(joinpath(@__DIR__, "..", "results", "comparacao_ffjm2_sr1_3.tsv")),
    ffjm2_options = (;),
    bfgs_options = (;),
    model_multistarts = (1, 10, 100, 1000),
    show_trace::Bool = true,
)
    multistarts = Int.(collect(model_multistarts))
    !isempty(multistarts) ||
        throw(ArgumentError("model_multistarts não pode ser vazio"))
    all(n -> n >= 1, multistarts) ||
        throw(ArgumentError("cada valor de model_multistarts deve ser pelo menos 1"))

    function raw_metrics(F_residual, x)
        residual = collect(F_residual(x))
        sse = sum(abs2, residual)
        rmsd = sqrt(sse / length(residual))
        dim = length(x)
        objective(z) = sum(abs2, F_residual(z))
        config = ForwardDiff.GradientConfig(
            objective,
            x,
            ForwardDiff.Chunk{dim}(),
        )
        gradient = ForwardDiff.gradient(objective, x, config)
        return (; sse, rmsd, gradient_norm = norm(gradient))
    end

    mkpath(dirname(output))
    rows = NamedTuple[]
    tend = 31.0
    initial = fill(0.09, 3)
    dim = length(initial)
    test = 1
    F_residual(x) = simulation(x, 0.0, tend, nothing).erro

    open(output, "w") do io
        write(
            io,
            "tend\tdimension\ttest\tmethod\tmodel_multistart\tsum_fi_squared\tRMSD\tgradient_norm\t",
            "external_evaluations\tfunction_evaluations\tgradient_evaluations\t",
            "function_evaluation_time_seconds\tgradient_evaluation_time_seconds\t",
            "total_function_evaluation_time_seconds\ttotal_gradient_evaluation_time_seconds\t",
            "execution_time_seconds\titerations\tconverged\t",
            "status\tx0\tsolution\n",
        )

        println(
            "\nComparação: tend = $tend, dimensão = $dim, ",
            "teste = $test, x0 = $initial, multi-starts = $multistarts",
        )

        for model_multistart in multistarts
            update = :sr1
            method_name = "ffjm2_sr1"
            println("\nExecutando $(uppercase(method_name)) com multi-start = $model_multistart")
            ff_options = merge(ffjm2_options, (; update, model_multistart, show_trace))
            external_evaluations = Ref(0)
            function counted_residual(x)
                external_evaluations[] += 1
                return F_residual(x)
            end
            result = ffjm2(counted_residual, initial; ff_options...)
            minimizer = copy(result.minimizer)
            metrics = raw_metrics(F_residual, minimizer)
            row = (;
                tend,
                dimension = dim,
                test,
                method = method_name,
                model_multistart,
                metrics...,
                external_evaluations = external_evaluations[],
                function_evaluations = result.function_evaluations,
                gradient_evaluations = result.gradient_evaluations,
                function_evaluation_time_seconds = result.function_evaluation_time_seconds,
                gradient_evaluation_time_seconds = result.gradient_evaluation_time_seconds,
                total_function_evaluation_time_seconds = result.total_function_evaluation_time_seconds,
                total_gradient_evaluation_time_seconds = result.total_gradient_evaluation_time_seconds,
                execution_time_seconds = result.execution_time_seconds,
                iterations = result.iterations,
                converged = result.converged,
                status = String(result.status),
                x0 = copy(initial),
                solution = minimizer,
            )
            push!(rows, row)
            _write_comparison_row(io, row)
            flush(io)
            println(
                "Avaliações externas $(uppercase(method_name)) (multi-start = $model_multistart) = ",
                external_evaluations[],
                " | avaliações de f = ", result.function_evaluations,
                " | chamadas do gradiente = ", result.gradient_evaluations,
            )
        end

        pure_options = merge((; show_trace), bfgs_options)
        pure_result = bfgs_puro_penalizado(initial; pure_options...)
        pure_x = copy(pure_result.minimizer)
        pure_metrics = raw_metrics(F_residual, pure_x)
        pure_row = (;
            tend,
            dimension = dim,
            test,
            method = "bfgs_puro_penalizado",
            model_multistart = 0,
            pure_metrics...,
            external_evaluations = pure_result.function_evaluations,
            function_evaluations = pure_result.function_evaluations,
            gradient_evaluations = pure_result.gradient_evaluations,
            function_evaluation_time_seconds = pure_result.function_evaluation_time_seconds,
            gradient_evaluation_time_seconds = pure_result.gradient_evaluation_time_seconds,
            total_function_evaluation_time_seconds = pure_result.total_function_evaluation_time_seconds,
            total_gradient_evaluation_time_seconds = pure_result.total_gradient_evaluation_time_seconds,
            execution_time_seconds = pure_result.execution_time_seconds,
            iterations = pure_result.iterations,
            converged = pure_result.converged,
            status = string(pure_result.status),
            x0 = copy(initial),
            solution = pure_x,
        )
        push!(rows, pure_row)
        _write_comparison_row(io, pure_row)
        flush(io)
        println(
            "Avaliações externas BFGS puro penalizado = ", pure_external_evaluations[],
            " | avaliações de f = ", pure_result.function_evaluations,
            " | chamadas do gradiente = ", pure_result.gradient_evaluations,
        )

        penalty_weight = get(pure_options, :penalty_weight, 1e6)
        lower = collect(float.(get(pure_options, :lower, zeros(dim))))
        upper = collect(float.(get(pure_options, :upper, fill(0.5, dim))))
        tol = get(pure_options, :g_tol, 1e-3)
        maxit = Int(get(pure_options, :maxiter, 100))
        maxnef = Int(get(pure_options, :f_calls_limit, 1000))

        for (method_name, solver) in (
            ("bfgs_backtracking", bfgs_backtracking),
            ("sr1_backtracking", sr1_backtracking),
        )
            external_evaluations = Ref(0)
            function_calls = Ref(0)
            gradient_calls = Ref(0)
            function_time = Ref(0.0)
            gradient_time = Ref(0.0)

            function scalar_objective(x)
                start_ns = time_ns()
                external_evaluations[] += 1
                residual = F_residual(x)
                penalty = zero(eltype(x))
                for i in eachindex(x)
                    below = max(zero(x[i]), lower[i] - x[i])
                    above = max(zero(x[i]), x[i] - upper[i])
                    penalty += below^2 + above^2
                end
                value = sum(abs2, residual) + penalty_weight * penalty
                function_calls[] += 1
                function_time[] += (time_ns() - start_ns) / 1e9
                return isnan(value) ? oftype(value, 1e26) : value
            end
            raw_objective(x) = begin
                external_evaluations[] += 1
                residual = F_residual(x)
                penalty = zero(eltype(x))
                for i in eachindex(x)
                    below = max(zero(x[i]), lower[i] - x[i])
                    above = max(zero(x[i]), x[i] - upper[i])
                    penalty += below^2 + above^2
                end
                sum(abs2, residual) + penalty_weight * penalty
            end
            gradient_config = ForwardDiff.GradientConfig(
                raw_objective,
                initial,
                ForwardDiff.Chunk{dim}(),
            )
            function scalar_gradient(x)
                start_ns = time_ns()
                gradient = ForwardDiff.gradient(raw_objective, x, gradient_config)
                gradient_calls[] += 1
                gradient_time[] += (time_ns() - start_ns) / 1e9
                return gradient
            end

            start_ns = time_ns()
            result = solver(
                scalar_objective, scalar_gradient, initial, tol, maxit, maxnef;
                show_trace,
            )
            execution_time_seconds = (time_ns() - start_ns) / 1e9
            minimizer = copy(result.x)
            metrics = raw_metrics(F_residual, minimizer)
            total_function_time = function_time[]
            total_gradient_time = gradient_time[]
            status = result.info == 0 ? :gradient_converged :
                result.info == 1 ? :maximum_iterations : :function_call_limit
            row = (;
                tend,
                dimension = dim,
                test,
                method = method_name,
                model_multistart = 0,
                metrics...,
                external_evaluations = external_evaluations[],
                function_evaluations = function_calls[],
                gradient_evaluations = gradient_calls[],
                function_evaluation_time_seconds = function_calls[] == 0 ? 0.0 : total_function_time / function_calls[],
                gradient_evaluation_time_seconds = gradient_calls[] == 0 ? 0.0 : total_gradient_time / gradient_calls[],
                total_function_evaluation_time_seconds = total_function_time,
                total_gradient_evaluation_time_seconds = total_gradient_time,
                execution_time_seconds,
                iterations = result.kon,
                converged = result.info == 0,
                status = String(status),
                x0 = copy(initial),
                solution = minimizer,
            )
            push!(rows, row)
            _write_comparison_row(io, row)
            flush(io)
            println(
                "Avaliações externas $(uppercase(method_name)) = ", external_evaluations[],
                " | avaliações de f = ", function_calls[],
                " | chamadas do gradiente = ", gradient_calls[],
            )
        end
    end

    println("Comparação salva em: $output")
    return (; rows, output)
end

function _write_comparison_row(io, row)
    # Mantém uma linha por execução para facilitar leitura por planilhas.
    values = (
        row.tend,
        row.dimension,
        row.test,
        row.method,
        row.model_multistart,
        row.sse,
        row.rmsd,
        row.gradient_norm,
        row.external_evaluations,
        row.function_evaluations,
        row.gradient_evaluations,
        row.function_evaluation_time_seconds,
        row.gradient_evaluation_time_seconds,
        row.total_function_evaluation_time_seconds,
        row.total_gradient_evaluation_time_seconds,
        row.execution_time_seconds,
        row.iterations,
        row.converged,
        row.status,
        repr(row.x0),
        repr(row.solution),
    )
    write(io, join(values, '\t'), '\n')
end

"""
    comparar_backtracking(; kwargs...)

Compara `Optim.BFGS()` (via [`bfgs_puro_penalizado`](@ref)) com quatro
buscas lineares: [`SimpleBackTracking`](@ref) (geométrico, `alpha=1`/
`rho=0.5`, sem interpolação), `LineSearches.BackTracking` de ordem 2 e ordem
3 (interpolação quadrática/cúbica) e `LineSearches.HagerZhang` — a busca
linear padrão do `Optim.BFGS()` quando nenhum `linesearch` é passado —, na
função objetivo penalizada `sv_objective`/`sv_gradient`. O CSV contém as
mesmas métricas de `comparar_ffjm2_bfgs`.
"""
function comparar_backtracking(;
    output = normpath(joinpath(@__DIR__, "..", "results", "comparacao_backtracking.csv")),
    tol::Real = 1e-3,
    maxit::Integer = 100,
    maxnef::Integer = 1000,
    penalty_weight::Real = 1e6,
    show_trace::Bool = true,
)
    tend = 31.0
    initial = fill(0.09, 10)
    dim = length(initial)
    lower = zeros(dim)
    upper = fill(0.5, dim)
    test = 1

    function raw_metrics(x)
        residual = collect(sv_residual(x))
        invalid_residual = any(value -> !isfinite(value) || abs(value) >= 1e27, residual)
        if invalid_residual
            return (; sse = NaN, rmsd = NaN, gradient_norm = NaN)
        end
        sse = sum(abs2, residual)
        isfinite(sse) || return (; sse = NaN, rmsd = NaN, gradient_norm = NaN)
        rmsd = sqrt(sse / length(residual))
        objective(z) = sum(abs2, sv_residual(z))
        config = ForwardDiff.GradientConfig(objective, x, ForwardDiff.Chunk{dim}())
        gradient = ForwardDiff.gradient(objective, x, config)
        return (; sse, rmsd, gradient_norm = norm(gradient))
    end

    mkpath(dirname(output))
    rows = NamedTuple[]
    header = (
        "tend", "dimension", "test", "method", "model_multistart",
        "sum_fi_squared", "RMSD", "gradient_norm", "external_evaluations",
        "function_evaluations", "gradient_evaluations",
        "function_evaluation_time_seconds", "gradient_evaluation_time_seconds",
        "total_function_evaluation_time_seconds",
        "total_gradient_evaluation_time_seconds", "execution_time_seconds",
        "iterations", "converged", "status", "x0", "solution",
    )

    open(output, "w") do io
        write(io, join(header, ','), '\n')
        for search in (
            # (name = "backtracking_simples", linesearch = SimpleBackTracking()),
            # (name = "backtracking_ordem_2", linesearch = LineSearches.BackTracking(order = 2)),
            # (name = "backtracking_ordem_3", linesearch = LineSearches.BackTracking(order = 3)),
            # (name = "padrao_hager_zhang", linesearch = LineSearches.HagerZhang()),
            (name = "backtracking_dinamico", linesearch = DynamicBackTracking()),
        )
            println("\nExecutando Optim.BFGS com $(search.name)")
            result = bfgs_puro_penalizado(
                initial;
                maxiter = maxit, f_calls_limit = maxnef, g_tol = tol,
                penalty_weight, lower, upper, linesearch = search.linesearch, show_trace,
            )
            metrics = raw_metrics(result.minimizer)
            row = (;
                tend, dimension = dim, test, method = search.name,
                model_multistart = 0, metrics...,
                external_evaluations = result.function_evaluations,
                function_evaluations = result.function_evaluations,
                gradient_evaluations = result.gradient_evaluations,
                function_evaluation_time_seconds = result.function_evaluation_time_seconds,
                gradient_evaluation_time_seconds = result.gradient_evaluation_time_seconds,
                total_function_evaluation_time_seconds = result.total_function_evaluation_time_seconds,
                total_gradient_evaluation_time_seconds = result.total_gradient_evaluation_time_seconds,
                execution_time_seconds = result.execution_time_seconds,
                iterations = result.iterations, converged = result.converged,
                status = string(result.status),
                x0 = copy(initial), solution = copy(result.minimizer),
            )
            push!(rows, row)
            _write_comparison_csv_row(io, row)
            flush(io)
        end
    end

    println("Comparação salva em: $output")
    return (; rows, output)
end

function _write_comparison_csv_row(io, row)
    values = (
        row.tend, row.dimension, row.test, row.method, row.model_multistart,
        row.sse, row.rmsd, row.gradient_norm, row.external_evaluations,
        row.function_evaluations, row.gradient_evaluations,
        row.function_evaluation_time_seconds, row.gradient_evaluation_time_seconds,
        row.total_function_evaluation_time_seconds,
        row.total_gradient_evaluation_time_seconds, row.execution_time_seconds,
        row.iterations, row.converged, row.status, repr(row.x0), repr(row.solution),
    )
    csv_field(value) = begin
        text = string(value)
        occursin(r"[,\"\r\n]", text) ? '"' * replace(text, '"' => "\"\"") * '"' : text
    end
    write(io, join(csv_field.(values), ','), '\n')
end

# ==============================================================================
# 6.1. Solver L-BFGS-B de referência
#
# Esta função não faz parte do FFJM2. Ela existe para produzir uma solução de
# comparação usando L-BFGS-B e limites de caixa nativos.
# ==============================================================================

"""
    bfgs_puro(x0; lower=zeros(length(x0)), upper=fill(0.5, length(x0)), kwargs...)

Minimiza a soma dos quadrados de [`sv_residual`](@ref) com limites de caixa
nativos, usando o solver
`OptimizationLBFGSB.LBFGSB()` e gradiente calculado explicitamente via
`ForwardDiff.Chunk{dim}()`. O retorno segue o contrato de `ffjm2` e também
preserva os aliases principais da solução SciML (`u`, `objective`, `stats` e
`retcode`). Como L-BFGS-B não armazena uma Hessiana completa, `hessians` é
`nothing`.

`evaluation_history` guarda, na ordem, todos os pontos pedidos pelo L-BFGS-B,
enquanto `alphas` guarda somente o passo aceito em cada evento `NEW_X`.
`extra_line_search_evaluations` informa o total de tentativas extras da busca
linear.
"""
function bfgs_puro(
    x0::AbstractVector;
    maxiter::Integer = 100,
    g_tol::Real = 1e-3,
    lower::AbstractVector = zeros(length(x0)),
    upper::AbstractVector = fill(0.5, length(x0)),
    show_trace::Bool = false,
)
    start_time_ns = time_ns()
    x = collect(float.(x0))
    dim = length(x)
    lb = collect(float.(lower))
    ub = collect(float.(upper))
    length(lb) == dim || throw(DimensionMismatch("lower e x0 devem ter o mesmo tamanho"))
    length(ub) == dim || throw(DimensionMismatch("upper e x0 devem ter o mesmo tamanho"))
    all(lb .< ub) || throw(ArgumentError("cada limite inferior deve ser menor que o superior"))

    function_evaluations = Ref(0)
    gradient_evaluations = Ref(0)
    function_evaluation_time_seconds = Ref(0.0)
    gradient_evaluation_time_seconds = Ref(0.0)
    latest_x = Ref{Any}(nothing)
    latest_residual = Ref{Any}(nothing)
    evaluation_history = NamedTuple[]

    raw_residual(x) = collect(sv_residual(x))

    # Sem penalidade de caixa: os limites já são impostos nativamente pelo
    # L-BFGS-B (`bounds`/`backend.nbd`/`backend.l`/`backend.u` abaixo). Usa a
    # mesma função objetivo compartilhada de `sr1_bfgs_backtracking.jl`
    # (`sv_objective_from_residual`/`sv_gradient!`) que `bfgs_puro_penalizado`
    # e `bfgs_puro_armijo`, só com `penalty_weight = 0` para desativar o termo
    # de penalidade — assim os três `bfgs_*` calculam objetivo e gradiente
    # exatamente da mesma forma, em vez de cada um reimplementar a soma de
    # quadrados saturada por conta própria.
    function objective(x, _)
        start_ns = time_ns()
        residual = raw_residual(x)
        value = sv_objective_from_residual(residual, x; penalty_weight = 0.0, lower, upper)
        function_evaluations[] += 1
        function_evaluation_time_seconds[] += (time_ns() - start_ns) / 1e9
        latest_x[] = copy(x)
        latest_residual[] = residual
        push!(evaluation_history, (;
            evaluation = function_evaluations[],
            x = copy(x),
            objective = value,
        ))
        return value
    end

    function grad!(G, x, _)
        start_ns = time_ns()
        sv_gradient!(G, x; penalty_weight = 0.0, lower, upper)
        gradient_evaluations[] += 1
        gradient_evaluation_time_seconds[] += (time_ns() - start_ns) / 1e9
        return G
    end

    backend_module = OptimizationLBFGSB.LBFGSBJL
    backend, bounds = backend_module._opt_bounds(dim, 10, lb, ub)
    backend_x = copy(x)
    backend_f = 0.0
    alphas = eltype(x)[]
    backend.task[1:5] = b"START"
    for i in eachindex(backend_x)
        backend.nbd[i] = bounds[1, i]
        backend.l[i] = bounds[2, i]
        backend.u[i] = bounds[3, i]
    end
    while true
        backend_module.setulb(
            dim, 10, backend_x, backend.l, backend.u, backend.nbd,
            backend_f, backend.g, 1e7, g_tol, backend.wa, backend.iwa,
            backend.task, -1, backend.csave, backend.lsave, backend.isave,
            backend.dsave,
        )
        if backend.task[1:2] == b"FG"
            backend_f = objective(backend_x, nothing)
            grad!(backend.g, backend_x, nothing)
        elseif backend.task[1:5] == b"NEW_X"
            push!(alphas, backend.dsave[14])
            if show_trace
                println(
                    "L-BFGS-B | iteração = ", backend.isave[30],
                    " | f = ", backend_f,
                    " | α = ", backend.dsave[14],
                    " | x = ", backend_x,
                )
            end
            if backend.isave[30] >= maxiter
                backend.task[1:43] = b"STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"
            end
        else
            break
        end
    end
    stop_reason = String(strip(String(backend.task)))
    retcode = OptimizationBase.deduce_retcode(stop_reason)
    stats = (;
        iterations = Int(backend.isave[30]),
        f_calls = function_evaluations[],
        g_calls = gradient_evaluations[],
    )
    result = (;
        u = copy(backend_x),
        objective = backend_f,
        original = backend,
        stats,
        retcode,
    )
    final_residual = if latest_x[] !== nothing && latest_x[] == result.u
        copy(latest_residual[])
    else
        # Normalmente a solução final já é a última avaliação. Este
        # fallback mantém o contrato mesmo se outro backend mudar esse comportamento.
        start_ns = time_ns()
        values = raw_residual(result.u)
        function_evaluations[] += 1
        function_evaluation_time_seconds[] += (time_ns() - start_ns) / 1e9
        values
    end
    final_gradient = copy(result.original.g)
    converged = result.retcode == OptimizationBase.ReturnCode.Success
    status = Symbol(string(result.retcode))
    total_function_evaluation_time_seconds = function_evaluation_time_seconds[]
    total_gradient_evaluation_time_seconds = gradient_evaluation_time_seconds[]
    mean_function_evaluation_time_seconds = function_evaluations[] == 0 ? 0.0 :
        total_function_evaluation_time_seconds / function_evaluations[]
    mean_gradient_evaluation_time_seconds = gradient_evaluations[] == 0 ? 0.0 :
        total_gradient_evaluation_time_seconds / gradient_evaluations[]
    execution_time_seconds = (time_ns() - start_time_ns) / 1e9
    if show_trace
        println(
            "L-BFGS-B finalizado | f = ", result.objective,
            " | iterações = ", result.stats.iterations,
            " | avaliações de f = ", function_evaluations[],
            " | avaliações de ∇f = ", gradient_evaluations[],
            " | retcode = ", result.retcode,
            " | x = ", result.u,
        )
    end
    return (;
        minimizer = result.u,
        minimum = result.objective,
        residual = final_residual,
        gradient = final_gradient,
        hessians = nothing,
        iterations = result.stats.iterations,
        converged,
        status,
        execution_time_seconds,
        rejected_directions = 0,
        alphas,
        # Histórico fiel de todas as tentativas da busca linear.
        evaluation_history,
        extra_line_search_evaluations = max(function_evaluations[] - length(alphas) - 1, 0),
        solution = result,
        # Aliases da interface Optimization.jl, mantidos por compatibilidade.
        u = result.u,
        objective = result.objective,
        stats = result.stats,
        retcode = result.retcode,
        function_evaluations = function_evaluations[],
        gradient_evaluations = gradient_evaluations[],
        residual_evaluations = function_evaluations[] + gradient_evaluations[],
        function_evaluation_time_seconds = mean_function_evaluation_time_seconds,
        gradient_evaluation_time_seconds = mean_gradient_evaluation_time_seconds,
        total_function_evaluation_time_seconds,
        total_gradient_evaluation_time_seconds,
    )
end

# ==============================================================================
# 6.2. BFGS do Optim com penalização externa de caixa
# ==============================================================================

"""
    bfgs_puro_penalizado(x0; kwargs...)

Minimiza a soma dos quadrados dos resíduos de [`sv_residual`](@ref) com
`Optim.BFGS()` e acrescenta uma penalidade quadrática quando alguma variável
viola `lower` ou `upper`. Interrompe a otimização quando o passo aceito é
menor que `alpha_min`. O retorno possui os mesmos campos de `bfgs_puro` para
permitir comparações diretas entre os dois métodos.
"""
function bfgs_puro_penalizado(
    x0::AbstractVector;
    maxiter::Integer = 100,
    f_calls_limit::Integer = 1000,
    g_calls_limit::Integer = 100,
    g_tol::Real = 1e-3,
    x_tol::Real = 0.0,
    alpha_min::Real = 1e-12,
    penalty_weight::Real = 1e6,
    lower::AbstractVector = zeros(length(x0)),
    upper::AbstractVector = fill(0.5, length(x0)),
    linesearch = nothing,
    show_trace::Bool = false,
)
    start_time_ns = time_ns()
    x = collect(float.(x0))
    dim = length(x)
    lb = collect(float.(lower))
    ub = collect(float.(upper))
    length(lb) == dim || throw(DimensionMismatch("lower e x0 devem ter o mesmo tamanho"))
    length(ub) == dim || throw(DimensionMismatch("upper e x0 devem ter o mesmo tamanho"))
    all(lb .< ub) || throw(ArgumentError("cada limite inferior deve ser menor que o superior"))
    alpha_min >= 0 || throw(ArgumentError("alpha_min deve ser não negativo"))
    penalty_weight > 0 || throw(ArgumentError("penalty_weight deve ser positivo"))

    function_evaluations = Ref(0)
    gradient_evaluations = Ref(0)
    function_evaluation_time_seconds = Ref(0.0)
    gradient_evaluation_time_seconds = Ref(0.0)
    latest_x = Ref{Any}(nothing)
    latest_residual = Ref{Any}(nothing)
    latest_gradient_x = Ref{Any}(nothing)
    latest_gradient = Ref{Any}(nothing)

    # `pen_objective` é a única definição da função objetivo penalizada desta
    # otimização: `objective` (valor, reaproveitando o resíduo já calculado)
    # e `grad!` (gradiente, via `ForwardDiff` diferenciando `pen_objective`
    # diretamente) chamam exatamente a mesma fórmula
    # (`sv_objective_from_residual`, `sr1_bfgs_backtracking.jl`), em vez de
    # `grad!` passar por uma segunda closure equivalente construída dentro de
    # `sv_gradient!`.
    raw_residual(x) = sv_fork_assimilation(x, 0.0, 31.0, nothing).erro
    pen_objective(x) = sum(abs2, raw_residual(x)) + sv_box_penalty(x, lb, ub, penalty_weight)

    function objective(x)
        start_ns = time_ns()
        residual = raw_residual(x)
        value = pen_objective(x)
        function_evaluations[] += 1
        function_evaluation_time_seconds[] += (time_ns() - start_ns) / 1e9
        latest_x[] = copy(x)
        latest_residual[] = residual
        return (isfinite(value)) ? value : oftype(value, Inf)
    end

    config = ForwardDiff.GradientConfig(pen_objective, x, ForwardDiff.Chunk{dim}())
    function grad!(G, x)
        start_ns = time_ns()
        ForwardDiff.gradient!(G, pen_objective, x, config)
        gradient_evaluations[] += 1
        gradient_evaluation_time_seconds[] += (time_ns() - start_ns) / 1e9
        latest_gradient_x[] = copy(x)
        latest_gradient[] = copy(G)
        return G
    end

    callback_calls = Ref(0)
    alpha_too_small = Ref(false)
    stop_on_small_alpha = function (state)
        callback_calls[] += 1
        if callback_calls[] > 1 && state.alpha < alpha_min
            alpha_too_small[] = true
            return true
        end
        return false
    end

    options = Optim.Options(
        iterations = Int(maxiter),
        f_calls_limit = Int(f_calls_limit),
        g_calls_limit = Int(g_calls_limit),
        g_abstol = g_tol,
        x_abstol = x_tol,
        show_trace = show_trace,
        store_trace = true,
        extended_trace = true,
        callback = stop_on_small_alpha,
    )
    method = linesearch === nothing ? Optim.BFGS(linesearch = LineSearches.BackTracking(order = 2)) : Optim.BFGS(; linesearch)
    result = Optim.optimize(objective, grad!, x, method, options)
    minimizer = copy(Optim.minimizer(result))
    final_residual = if latest_x[] !== nothing && latest_x[] == minimizer
        copy(latest_residual[])
    else
        start_ns = time_ns()
        values = raw_residual(minimizer)
        function_evaluations[] += 1
        function_evaluation_time_seconds[] += (time_ns() - start_ns) / 1e9
        values
    end
    final_gradient = if latest_gradient_x[] !== nothing && latest_gradient_x[] == minimizer
        copy(latest_gradient[])
    else
        # Fallback raro: o Optim normalmente termina no último ponto onde
        # solicitou o gradiente.
        G = similar(minimizer)
        grad!(G, minimizer)
    end

    total_function_evaluation_time_seconds = function_evaluation_time_seconds[]
    total_gradient_evaluation_time_seconds = gradient_evaluation_time_seconds[]
    mean_function_evaluation_time_seconds = function_evaluations[] == 0 ? 0.0 :
        total_function_evaluation_time_seconds / function_evaluations[]
    mean_gradient_evaluation_time_seconds = gradient_evaluations[] == 0 ? 0.0 :
        total_gradient_evaluation_time_seconds / gradient_evaluations[]
    converged = !alpha_too_small[] && Optim.converged(result)
    status = alpha_too_small[] ? :alpha_too_small : Symbol(string(result.termination_code))
    execution_time_seconds = (time_ns() - start_time_ns) / 1e9
    alphas = [
        state.metadata["Current step size"]
        for state in Optim.trace(result) if state.iteration > 0
    ]

    return (;
        minimizer,
        minimum = Optim.minimum(result),
        residual = final_residual,
        gradient = final_gradient,
        hessians = nothing,
        iterations = Optim.iterations(result),
        converged,
        status,
        execution_time_seconds,
        rejected_directions = 0,
        alphas,
        solution = result,
        u = minimizer,
        objective = Optim.minimum(result),
        stats = result,
        retcode = result.termination_code,
        function_evaluations = function_evaluations[],
        gradient_evaluations = gradient_evaluations[],
        residual_evaluations = function_evaluations[] + gradient_evaluations[],
        function_evaluation_time_seconds = mean_function_evaluation_time_seconds,
        gradient_evaluation_time_seconds = mean_gradient_evaluation_time_seconds,
        total_function_evaluation_time_seconds,
        total_gradient_evaluation_time_seconds,
    )
end

# ==============================================================================
# 6.3. BFGS explícito com busca linear somente de Armijo
# ==============================================================================

"""
    bfgs_puro_armijo(x0; kwargs...)

Executa BFGS sobre [`sv_residual`](@ref) com penalidade externa de caixa e
busca linear apenas de Armijo,
com backtracking por interpolação cúbica e sem condição de Wolfe. A direção
quase-Newton `-B⁻¹∇f` somente é aceita se
satisfizer o mesmo critério de direção usado por `ffjm2`; caso contrário,
utiliza `-∇f`. O campo `alphas` do retorno contém, na ordem, o passo aceito
em cada iteração.
"""
function bfgs_puro_armijo(
    x0::AbstractVector;
    maxiter::Integer = 100,
    f_calls_limit::Integer = 200,
    g_calls_limit::Integer = 100,
    g_tol::Real = 1e-3,
    x_tol::Real = 0.0,
    f_rel_tol::Union{Nothing,Real} = 1e-8,
    penalty_weight::Real = 1e6,
    lower::AbstractVector = zeros(length(x0)),
    upper::AbstractVector = fill(0.5, length(x0)),
    armijo_c1::Real = 1e-4,
    backtracking_factor::Real = 0.5,
    backtracking_min_factor::Real = 0.1,
    max_backtrackings::Integer = 50,
    direction_theta::Real = 1e-4,
    direction_beta::Real = 1e-8,
    direction_delta::Real = Inf,
    update_tol::Real = sqrt(eps(Float64)),
    show_trace::Bool = false,
)
    start_time_ns = time_ns()
    x = collect(float.(x0))
    n = length(x)
    lb = collect(float.(lower))
    ub = collect(float.(upper))
    length(lb) == n || throw(DimensionMismatch("lower e x0 devem ter o mesmo tamanho"))
    length(ub) == n || throw(DimensionMismatch("upper e x0 devem ter o mesmo tamanho"))
    all(lb .< ub) || throw(ArgumentError("cada limite inferior deve ser menor que o superior"))
    penalty_weight > 0 || throw(ArgumentError("penalty_weight deve ser positivo"))
    0 < armijo_c1 < 1 || throw(ArgumentError("armijo_c1 deve pertencer a (0, 1)"))
    0 < backtracking_factor < 1 ||
        throw(ArgumentError("backtracking_factor deve pertencer a (0, 1)"))
    0 < backtracking_min_factor <= backtracking_factor ||
        throw(ArgumentError(
            "backtracking_min_factor deve pertencer a (0, backtracking_factor]",
        ))
    0 < direction_theta < 1 ||
        throw(ArgumentError("direction_theta deve pertencer a (0, 1)"))
    direction_beta > 0 || throw(ArgumentError("direction_beta deve ser positivo"))
    direction_delta > 0 || throw(ArgumentError("direction_delta deve ser positivo"))

    function_evaluations = Ref(0)
    gradient_evaluations = Ref(0)
    function_time = Ref(0.0)
    gradient_time = Ref(0.0)
    latest_x = Ref{Any}(nothing)
    latest_residual = Ref{Any}(nothing)

    # Objetivo e gradiente compartilhados com `bfgs_puro`/`bfgs_puro_penalizado`
    # via `sv_objective_from_residual`/`sv_gradient!` (`sr1_bfgs_backtracking.jl`).
    raw_residual(z) = collect(sv_residual(z))
    function objective(z)
        start_ns = time_ns()
        residual = raw_residual(z)
        value = sv_objective_from_residual(residual, z; penalty_weight, lower = lb, upper = ub)
        function_evaluations[] += 1
        function_time[] += (time_ns() - start_ns) / 1e9
        latest_x[] = copy(z)
        latest_residual[] = residual
        return value
    end

    function gradient!(G, z)
        start_ns = time_ns()
        sv_gradient!(G, z; penalty_weight, lower = lb, upper = ub)
        gradient_evaluations[] += 1
        gradient_time[] += (time_ns() - start_ns) / 1e9
        return G
    end

    B = Matrix{eltype(x)}(I, n, n)
    f = objective(x)
    g = similar(x)
    gradient!(g, x)
    iterations = 0
    rejected_directions = 0
    alphas = eltype(x)[]
    status = :maximum_iterations
    armijo_linesearch = LineSearches.BackTracking(
        c_1 = armijo_c1,
        ρ_hi = backtracking_factor,
        ρ_lo = backtracking_min_factor,
        iterations = Int(max_backtrackings),
        order = 3,
    )

    for k in 1:Int(maxiter)
        if norm(g) <= g_tol
            status = :gradient_converged
            break
        end
        if function_evaluations[] >= f_calls_limit
            status = :function_call_limit
            break
        end
        if gradient_evaluations[] >= g_calls_limit
            status = :gradient_call_limit
            break
        end

        p = try
            -(B \ g)
        catch
            fill!(similar(g), NaN)
        end
        pnorm = norm(p)
        gnorm = norm(g)
        valid_direction = all(isfinite, p) &&
            dot(g, p) <= -direction_theta * pnorm^2 * gnorm^2 &&
            direction_beta * gnorm <= pnorm <= direction_delta
        if !valid_direction
            rejected_directions += 1
            p = -g
        end

        directional_derivative = dot(g, p)
        line_search_evaluations = Ref(0)
        ϕ(α) = begin
            function_evaluations[] < f_calls_limit ||
                throw(LineSearches.LineSearchException(
                    "limite de avaliações da função atingido", α,
                ))
            line_search_evaluations[] += 1
            trial_x = x .+ α .* p
            trial_f = objective(trial_x)
            trial_f
        end
        α, ftrial = try
            armijo_linesearch(ϕ, one(eltype(x)), f, directional_derivative)
        catch err
            if err isa LineSearches.LineSearchException
                status = function_evaluations[] >= f_calls_limit ?
                    :function_call_limit : :line_search_failed
                break
            end
            rethrow()
        end
        if !isfinite(ftrial)
            status = :line_search_failed
            break
        end
        xtrial = x .+ α .* p
        backtrackings = max(line_search_evaluations[] - 1, 0)
        push!(alphas, α)

        gtrial = similar(g)
        gradient!(gtrial, xtrial)
        s = xtrial - x
        y = gtrial - g
        sy = dot(s, y)
        Bs = B * s
        sBs = dot(s, Bs)
        if sy > update_tol * max(one(sy), norm(s) * norm(y)) &&
           sBs > update_tol * max(one(sBs), norm(s) * norm(Bs))
            B .+= (y * y') / sy - (Bs * Bs') / sBs
            B .= (B .+ B') ./ 2
        end

        fold = f
        x, f, g = xtrial, ftrial, gtrial
        iterations = k
        if show_trace
            println(
                "BFGS-Armijo iter ", k,
                " | f = ", f,
                " | ||g|| = ", norm(g),
                " | α = ", α,
                " | backtrackings = ", backtrackings,
                " | direção rejeitada = ", !valid_direction,
            )
        end
        if norm(s) <= x_tol * max(one(eltype(x)), norm(x))
            status = :step_converged
            break
        end
        if f_rel_tol !== nothing &&
           abs(fold - f) <= f_rel_tol * max(one(f), abs(fold))
            status = :function_converged
            break
        end
    end
    if status == :maximum_iterations && norm(g) <= g_tol
        status = :gradient_converged
    end

    converged = status in (:gradient_converged, :step_converged, :function_converged)
    final_residual = if latest_x[] !== nothing && latest_x[] == x
        copy(latest_residual[])
    else
        start_ns = time_ns()
        values = raw_residual(x)
        function_evaluations[] += 1
        function_time[] += (time_ns() - start_ns) / 1e9
        values
    end
    total_function_evaluation_time_seconds = function_time[]
    total_gradient_evaluation_time_seconds = gradient_time[]
    mean_function_evaluation_time_seconds = function_evaluations[] == 0 ? 0.0 :
        function_time[] / function_evaluations[]
    mean_gradient_evaluation_time_seconds = gradient_evaluations[] == 0 ? 0.0 :
        gradient_time[] / gradient_evaluations[]
    execution_time_seconds = (time_ns() - start_time_ns) / 1e9
    stats = (; iterations, f_calls = function_evaluations[],
             g_calls = gradient_evaluations[])
    solution = (; minimizer = copy(x), minimum = f, iterations, status)

    return (;
        minimizer = copy(x),
        minimum = f,
        residual = final_residual,
        gradient = copy(g),
        hessians = nothing,
        iterations,
        converged,
        status,
        execution_time_seconds,
        rejected_directions,
        alphas,
        residual_evaluations = function_evaluations[] + gradient_evaluations[],
        solution,
        u = copy(x),
        objective = f,
        stats,
        retcode = status,
        function_evaluations = function_evaluations[],
        gradient_evaluations = gradient_evaluations[],
        function_evaluation_time_seconds = mean_function_evaluation_time_seconds,
        gradient_evaluation_time_seconds = mean_gradient_evaluation_time_seconds,
        total_function_evaluation_time_seconds,
        total_gradient_evaluation_time_seconds,
    )
end

# ==============================================================================
# 7. Exemplo de resíduos e ponto inicial do problema Saint-Venant
# ==============================================================================

F(x) = sv_fork_assimilation(
    x,
    0.0,
    31.0,
    nothing,
).erro

x0 = fill(0.09, 3)

"""
    comparar_buscas_lineares_bfgs_penalizado(; kwargs...)

Executa `bfgs_puro_penalizado` para o problema Saint-Venant com dimensões 3
e 10. Compara a busca linear padrão do `Optim.BFGS`, backtracking geométrico
de ordem 1 e os backtrackings interpolados de ordens 2 e 3. Todas as
execuções param quando um passo aceito é menor que `alpha_min`.

Os resultados são escritos em CSV e também devolvidos no campo `rows`.
"""
function comparar_buscas_lineares_bfgs_penalizado(;
    output = normpath(joinpath(
        @__DIR__, "..", "results", "comparacao_buscas_lineares_bfgs_penalizado.csv",
    )),
    dimensions = (3, 10),
    alpha_min::Real = 1e-12,
    show_trace::Bool = true,
    bfgs_options = (;),
)
    alpha_min > 0 || throw(ArgumentError("alpha_min deve ser positivo"))
    dims = Int.(collect(dimensions))
    !isempty(dims) || throw(ArgumentError("dimensions não pode ser vazio"))
    all(>(0), dims) || throw(ArgumentError("as dimensões devem ser positivas"))

    searches = (
        (name = "padrao_hager_zhang", linesearch = nothing),
        (name = "backtracking_ordem_1", linesearch = GeometricBackTracking(; min_alpha = alpha_min)),
        (name = "backtracking_ordem_2", linesearch = LineSearches.BackTracking(; order = 2)),
        (name = "backtracking_ordem_3", linesearch = LineSearches.BackTracking(; order = 3)),
    )
    csv_field(value) = begin
        text = value isa AbstractVector ? repr(collect(value)) : string(value)
        occursin(r"[,\"\n\r]", text) ? "\"$(replace(text, '\"' => "\"\""))\"" : text
    end

    mkpath(dirname(output))
    rows = NamedTuple[]
    open(output, "w") do io
        header = (
            "dimension", "line_search", "f_x", "RMSD", "gradient_norm",
            "function_evaluations", "gradient_evaluations", "execution_time_seconds",
            "iterations", "convergence_type", "converged", "minimizer",
        )
        write(io, join(header, ','), '\n')

        for dimension in dims, search in searches
            initial = fill(0.09, dimension)
            options = merge(
                bfgs_options,
                (; alpha_min, show_trace, linesearch = search.linesearch),
            )
            println(
                "\nBFGS penalizado | dimensão = $dimension",
                " | busca linear = $(search.name)",
            )
            result = bfgs_puro_penalizado(initial; options...)
            residual = collect(sv_residual(result.minimizer))
            rmsd = norm(residual) / sqrt(length(residual))
            row = (;
                dimension,
                line_search = search.name,
                f_x = result.minimum,
                RMSD = rmsd,
                gradient_norm = norm(result.gradient),
                function_evaluations = result.function_evaluations,
                gradient_evaluations = result.gradient_evaluations,
                execution_time_seconds = result.execution_time_seconds,
                iterations = result.iterations,
                convergence_type = string(result.status),
                converged = result.converged,
                minimizer = copy(result.minimizer),
            )
            push!(rows, row)
            write(io, join(csv_field.(values(row)), ','), '\n')
            flush(io)
        end
    end

    println("Comparação de buscas lineares salva em: $output")
    return (; rows, output)
end

"""
    comparar_solvers_subproblema_ffjm2(; kwargs...)

Compara Ipopt, BFGS, BOBYQA e MADS na solução do subproblema interno do
`ffjm2`, usando exatamente os mesmos pontos iniciais aleatórios para cada
solver. Por padrão, executa FFJM2-SR1 nas dimensões 3 e 10, com multi-starts
de 1, 10 e 100 pontos, e escreve os resultados em CSV.
"""
function comparar_solvers_subproblema_ffjm2(;
    simulation = sv_fork_assimilation,
    output = normpath(joinpath(
        @__DIR__, "..", "results", "comparacao_solvers_subproblema_ffjm2_bfgs.csv",
    )),
    dimensions = (3, 10),
    model_multistarts = (1, 10, 100),
    model_solvers = (:ipopt, :bfgs, :bobyqa, :mads),
    update::Symbol = :bfgs,
    show_trace::Bool = true,
    ffjm2_options = (;),
)
    dims = Int.(collect(dimensions))
    multistarts = Int.(collect(model_multistarts))
    solvers = Symbol.(collect(model_solvers))
    !isempty(dims) && all(>(0), dims) ||
        throw(ArgumentError("dimensions deve conter apenas valores positivos"))
    !isempty(multistarts) && all(>(0), multistarts) ||
        throw(ArgumentError("model_multistarts deve conter apenas valores positivos"))
    !isempty(solvers) && all(s -> s in (:ipopt, :bfgs, :bobyqa, :mads), solvers) ||
        throw(ArgumentError("model_solvers contém um solver inválido"))
    update in (:bfgs, :sr1) || throw(ArgumentError("update deve ser :bfgs ou :sr1"))

    csv_field(value) = begin
        text = value isa AbstractVector ? repr(collect(value)) : string(value)
        occursin(r"[,\"\n\r]", text) ? "\"$(replace(text, '\"' => "\"\""))\"" : text
    end
    header = (
        "dimension", "model_solver", "model_multistart", "update", "f_x", "RMSD",
        "gradient_norm", "function_evaluations", "gradient_evaluations",
        "execution_time_seconds", "iterations", "convergence_type", "converged",
        "model_solves", "model_iterations", "model_solve_time_seconds", "minimizer",
    )

    mkpath(dirname(output))
    rows = NamedTuple[]
    open(output, "w") do io
        write(io, join(header, ','), '\n')
        for dimension in dims, solver in solvers, multistart in multistarts
            initial = fill(0.09, dimension)
            residual_function(x) = simulation(x, 0.0, 31.0, nothing).erro
            options = merge(
                ffjm2_options,
                (;
                    update,
                    model_solver = solver,
                    model_multistart = multistart,
                    show_trace,
                ),
            )
            println(
                "\nFFJM2-$(uppercase(string(update)))",
                " | dimensão = $dimension",
                " | solver interno = $(uppercase(string(solver)))",
                " | multi-start = $multistart",
            )
            result = ffjm2(residual_function, initial; options...)
            rmsd = norm(result.residual) / sqrt(length(result.residual))
            row = (;
                dimension,
                model_solver = string(solver),
                model_multistart = multistart,
                update = string(update),
                f_x = result.minimum,
                RMSD = rmsd,
                gradient_norm = norm(result.gradient),
                function_evaluations = result.function_evaluations,
                gradient_evaluations = result.gradient_evaluations,
                execution_time_seconds = result.execution_time_seconds,
                iterations = result.iterations,
                convergence_type = string(result.status),
                converged = result.converged,
                model_solves = result.model_solves,
                model_iterations = result.model_iterations,
                model_solve_time_seconds = result.model_solve_time_seconds,
                minimizer = copy(result.minimizer),
            )
            push!(rows, row)
            write(io, join(csv_field.(values(row)), ','), '\n')
            flush(io)
        end
    end

    println("Comparação dos solvers do subproblema salva em: $output")
    return (; rows, output)
end

# ==============================================================================
# 8. Benchmark: custo computacional de F_residual e de suas derivadas por
#    ForwardDiff, em função da dimensão de entrada
# ==============================================================================

"""
    benchmark_forwarddiff_sv(; kwargs...)

Mede o custo computacional (em segundos) de avaliar

    F_residual(x) = sum(abs2, simulation(x, 0.0, tend, nothing).erro)

e de suas derivadas por `ForwardDiff`, em função da dimensão de
`x = fill(0.09, dim)`.

Para cada `dim` em `gradient_dims` (padrão `(1, 3, 10, 50, 100)`), mede o
tempo de uma avaliação de `F_residual` e o tempo do gradiente
(`ForwardDiff.gradient`, com `ForwardDiff.Chunk{dim}()`, mesma convenção
usada no restante do arquivo). Para cada `dim` em `hessian_dims` (padrão
`(1, 3, 100)` — bem menor que `gradient_dims`, pois o custo da Hessiana cresce
muito mais rápido que o do gradiente), mede também o tempo da Hessiana
(`ForwardDiff.hessian`). Diferente do gradiente, a Hessiana usa o chunk
*padrão* do ForwardDiff (não `Chunk{dim}()`): a Hessiana é forward-sobre-forward,
e forçar um chunk do tamanho de `dim` faria o código gerado (e o tempo de
compilação) crescer com `dim²`, o que é proibitivo para `dim = 100`.

Cada combinação (dimensão, métrica) é avaliada uma única vez (`@elapsed`),
sem repetições — cada avaliação de `simulation` já custa segundos, e
repetições tornariam o benchmark caro demais, especialmente para a Hessiana
em `dim = 100`.

Os resultados são escritos incrementalmente (uma linha por vez, com `flush`)
em `output`, um CSV com colunas `dim`, `metric` (`"f"`, `"gradient"` ou
`"hessian"`), `seconds` e `value` — `value` é o próprio `F_residual(x)` para
`metric = "f"`, e a norma (euclidiana para o gradiente, de Frobenius para a
Hessiana) nas demais linhas. Também são devolvidos no campo `rows`.
"""
function benchmark_forwarddiff_sv(;
    simulation = sv_fork_assimilation,
    tend::Real = 31.0,
    gradient_dims = (1, 3, 10, 50, 100),
    hessian_dims = (1, 3, 100),
    output = normpath(joinpath(@__DIR__, "..", "results", "benchmark_forwarddiff_sv.csv")),
    show_trace::Bool = true,
)
    grad_dims = Int.(collect(gradient_dims))
    hess_dims = Int.(collect(hessian_dims))
    !isempty(grad_dims) && all(>(0), grad_dims) ||
        throw(ArgumentError("gradient_dims deve conter apenas valores positivos"))
    !isempty(hess_dims) && all(>(0), hess_dims) ||
        throw(ArgumentError("hessian_dims deve conter apenas valores positivos"))

    function F_residual(x)
        residual = simulation(x, 0.0, tend, nothing).erro
        return sum(abs2, residual)
    end

    mkpath(dirname(output))
    rows = NamedTuple[]

    open(output, "w") do io
        write(io, "dim,metric,seconds,value\n")
        function record!(dim, metric, seconds, value)
            row = (; dim, metric, seconds, value)
            push!(rows, row)
            write(io, join((row.dim, row.metric, row.seconds, row.value), ','), '\n')
            flush(io)
            show_trace && println(
                "dim = $dim | $metric | $(round(seconds; digits = 3)) s | valor = $value",
            )
            return row
        end

        for dim in grad_dims
            x = fill(0.09, dim)

            t_f = @elapsed (f = F_residual(x))
            record!(dim, "f", t_f, f)

            config = ForwardDiff.GradientConfig(F_residual, x, ForwardDiff.Chunk{dim}())
            t_g = @elapsed (g = ForwardDiff.gradient(F_residual, x, config))
            record!(dim, "gradient", t_g, norm(g))
        end

        for dim in hess_dims
            x = fill(0.09, dim)
            hconfig = ForwardDiff.HessianConfig(F_residual, x)
            t_h = @elapsed (H = ForwardDiff.hessian(F_residual, x, hconfig))
            record!(dim, "hessian", t_h, norm(H))
        end
    end

    println("Benchmark de derivadas por ForwardDiff salvo em: $output")
    return (; rows, output)
end

# ==============================================================================
# 9. Comparação de BFGS, SPG, BOBYQA e MADS na função objetivo penalizada
# ==============================================================================

"""
    comparar_bfgs_spg_bobyqa_mads(; kwargs...)

Compara BFGS (`Optim.BFGS`, mesma penalização/backtracking de
[`bfgs_puro_penalizado`](@ref)), SPG ([`spg_box`](@ref), `teste2.jl` —
gradiente espectral projetado próprio, com salvaguarda contra falsa
convergência em pontos saturados, `info = 4`), BOBYQA (`NLopt.jl`,
`:LN_BOBYQA`) e MADS (`NOMAD.jl`) na minimização de `sv_objective` (problema
fixo de [`sv_residual`](@ref), tend=31) para cada `dimensions`. O chute
inicial é sempre `fill(0.09, dim)`, com caixa `[0, 0.5]^dim` e o mesmo
`penalty_weight` para todos os solvers.

BOBYQA e MADS não usam gradiente: `gradient_evaluations` é sempre `0` para
eles. `gradient_norm` ainda é calculado (via `sv_gradient`, fora da contagem
de avaliações) no minimizador final de todo método, para permitir comparação
com BFGS/SPG.

Os resultados são escritos incrementalmente em `output` (CSV) e também
devolvidos no campo `rows`.
"""
function comparar_bfgs_spg_bobyqa_mads(;
    output = normpath(joinpath(
        @__DIR__, "..", "results", "comparacao_bfgs_spg_bobyqa_mads.csv",
    )),
    dimensions = (3, 10),
    penalty_weight::Real = 1e6,
    maxiter::Integer = 100,
    f_calls_limit::Integer = 1000,
    g_calls_limit::Integer = 100,
    g_tol::Real = 1e-3,
    rhobeg::Real = 0.005,
    rhoend::Real = 1e-6,
    show_trace::Bool = true,
)
    dims = Int.(collect(dimensions))
    !isempty(dims) && all(>(0), dims) ||
        throw(ArgumentError("dimensions deve conter apenas valores positivos"))

    csv_field(value) = begin
        text = value isa AbstractVector ? repr(collect(value)) : string(value)
        occursin(r"[,\"\n\r]", text) ? "\"$(replace(text, '\"' => "\"\""))\"" : text
    end
    header = (
        "method", "dimension", "RMSD", "gradient_norm",
        "execution_time_seconds", "function_evaluations", "gradient_evaluations",
        "f_x", "minimizer",
    )

    mkpath(dirname(output))
    rows = NamedTuple[]
    open(output, "w") do io
        write(io, join(header, ','), '\n')

        for dim in dims
            x0 = fill(0.09, dim)
            lower = zeros(dim)
            upper = fill(0.5, dim)

            # Objetivo/gradiente locais, autocontidos (sem passar pelas
            # funções compartilhadas sv_objective/sv_gradient!/sv_residual) —
            # usados por SPG, BOBYQA e MADS abaixo. BFGS continua à parte,
            # via bfgs_puro_penalizado (tem seu próprio aparato equivalente).
            function_evaluations = Ref(0)
            gradient_evaluations = Ref(0)
            function_evaluation_time_seconds = Ref(0.0)
            gradient_evaluation_time_seconds = Ref(0.0)
            latest_x = Ref{Any}(nothing)
            latest_residual = Ref{Any}(nothing)
            latest_gradient_x = Ref{Any}(nothing)
            latest_gradient = Ref{Any}(nothing)

            raw_residual(x) = sv_fork_assimilation(x, 0.0, 15.0, nothing).erro
            pen_objective(x) = sum(abs2, raw_residual(x)) + sv_box_penalty(x, lower, upper, penalty_weight)

            function objective(x)
                start_ns = time_ns()
                residual = raw_residual(x)
                value = pen_objective(x)
                function_evaluations[] += 1
                function_evaluation_time_seconds[] += (time_ns() - start_ns) / 1e9
                latest_x[] = copy(x)
                latest_residual[] = residual
                return (isfinite(value)) ? value : oftype(value, Inf)
            end

            config = ForwardDiff.GradientConfig(pen_objective, x0, ForwardDiff.Chunk{dim}())
            function grad!(G, x)
                start_ns = time_ns()
                ForwardDiff.gradient!(G, pen_objective, x, config)
                gradient_evaluations[] += 1
                gradient_evaluation_time_seconds[] += (time_ns() - start_ns) / 1e9
                latest_gradient_x[] = copy(x)
                latest_gradient[] = copy(G)
                return G
            end

            function record!(
                method, minimizer, f_x, execution_time_seconds,
                function_evaluations, gradient_evaluations,
            )
                residual = collect(raw_residual(minimizer))
                rmsd = norm(residual) / sqrt(length(residual))
                gradient_buffer = similar(minimizer, Float64)
                grad!(gradient_buffer, minimizer)
                gradient_norm = norm(gradient_buffer)
                row = (;
                    method, dimension = dim, RMSD = rmsd, gradient_norm,
                    execution_time_seconds, function_evaluations, gradient_evaluations,
                    f_x, minimizer = copy(minimizer),
                )
                push!(rows, row)
                write(io, join(csv_field.(values(row)), ','), '\n')
                flush(io)
                return row
            end

            println("\ndimensão = $dim")

            # BFGS: reaproveita bfgs_puro_penalizado (mesma penalização/backtracking).
            println("Executando BFGS")
            result_bfgs = bfgs_puro_penalizado(
                x0; maxiter, f_calls_limit, g_calls_limit, g_tol,
                penalty_weight, lower, upper, show_trace,
            )
            record!(
                "BFGS", result_bfgs.minimizer, result_bfgs.minimum,
                result_bfgs.execution_time_seconds, result_bfgs.function_evaluations,
                result_bfgs.gradient_evaluations,
            )

            # SPG (spg_box, teste2.jl): gradiente espectral projetado próprio,
            # com salvaguarda contra falsa convergência em pontos saturados
            # (info = 4) — ver docstring de spg_box.
            println("Executando SPG")
            gradient_spg(x) = begin
                G = similar(x, Float64)
                grad!(G, x)
                G
            end
            start_spg_ns = time_ns()
            result_spg = spg_box(
                objective, gradient_spg, x0, lower, upper;
                gtol = g_tol, maxit = maxiter, maxnef = f_calls_limit, show_trace,
            )
            execution_time_spg = (time_ns() - start_spg_ns) / 1e9
            record!(
                "SPG", result_spg.x, result_spg.fval, execution_time_spg,
                result_spg.nef, result_spg.neg,
            )

            # BOBYQA (NLopt.jl, livre de derivada): caixa nativa via lower/upper_bounds.
            println("Executando BOBYQA")
            function_evaluations_bobyqa = Ref(0)
            optimizer = NLopt.Opt(:LN_BOBYQA, dim)
            optimizer.lower_bounds = lower
            optimizer.upper_bounds = upper
            optimizer.initial_step = fill(Float64(rhobeg), dim)
            optimizer.xtol_abs = fill(Float64(rhoend), dim)
            optimizer.maxeval = Int(f_calls_limit)
            optimizer.min_objective = (x, grad) -> begin
                function_evaluations_bobyqa[] += 1
                return objective(x)
            end
            start_bobyqa_ns = time_ns()
            f_bobyqa, x_bobyqa, status_bobyqa = NLopt.optimize(optimizer, x0)
            execution_time_bobyqa = (time_ns() - start_bobyqa_ns) / 1e9
            record!(
                "BOBYQA", x_bobyqa, f_bobyqa, execution_time_bobyqa,
                function_evaluations_bobyqa[], 0,
            )

            # MADS (NOMAD.jl, livre de derivada): caixa nativa via lower_bound/upper_bound.
            println("Executando MADS")
            function_evaluations_mads = Ref(0)
            best_value_mads = Ref(Inf)
            best_point_mads = copy(x0)
            objective_mads = function (x)
                value = Float64(objective(x))
                function_evaluations_mads[] += 1
                if value < best_value_mads[]
                    best_value_mads[] = value
                    best_point_mads .= x
                end
                return true, true, [value]
            end
            options_mads = NOMAD.NomadOptions(
                display_degree = 0, max_bb_eval = Int(f_calls_limit),
            )
            problem_mads = NOMAD.NomadProblem(
                dim, 1, ["OBJ"], objective_mads,
                input_types = fill("R", dim),
                lower_bound = lower, upper_bound = upper,
                min_mesh_size = fill(Float64(rhoend), dim),
                initial_mesh_size = fill(Float64(rhobeg), dim),
                options = options_mads,
            )
            start_mads_ns = time_ns()
            result_mads = NOMAD.solve(problem_mads, x0)
            execution_time_mads = (time_ns() - start_mads_ns) / 1e9
            x_mads = result_mads.x_sol === nothing ?
                copy(best_point_mads) : collect(result_mads.x_sol)
            f_mads = Float64(objective(x_mads))
            record!(
                "MADS", x_mads, f_mads, execution_time_mads,
                function_evaluations_mads[], 0,
            )
        end
    end

    println("Comparação BFGS/SPG/BOBYQA/MADS salva em: $output")
    return (; rows, output)
end

# ==============================================================================
# 10. Backtracking simples (linesearch geométrico minimalista)
# ==============================================================================

"""
    SimpleBackTracking(; alpha=1.0, rho=0.5, c1=1e-4, iterations=1000, min_alpha=1e-12)

Backtracking geométrico minimalista, compatível com a interface de
`LineSearches.AbstractLineSearch` (mesmo protocolo de [`GeometricBackTracking`](@ref)),
para uso direto em `Optim.BFGS(linesearch = SimpleBackTracking())`.

Diferente de [`GeometricBackTracking`](@ref) — que parte do `initial_alpha`
que o próprio `Optim.BFGS` sugere a cada iteração —, este backtracking
sempre recomeça do `alpha` fixo configurado no construtor (`1.0` por
padrão), ignorando o `initial_alpha` recebido. A cada tentativa reprovada,
multiplica o passo por `rho` (`0.5` por padrão) até satisfazer a condição de
Armijo (parâmetro `c1`) ou até `min_alpha`/`iterations` ser atingido, caso em
que lança `LineSearches.LineSearchException` (mesmo tratamento de
`GeometricBackTracking`, necessário porque `Optim.BFGS` aplica o passo
incondicionalmente ao capturar essa exceção).
"""
struct SimpleBackTracking{T} <: LineSearches.AbstractLineSearch
    alpha::T
    rho::T
    c1::T
    iterations::Int
    min_alpha::T
end

SimpleBackTracking(;
    alpha::Real = 1.0,
    rho::Real = 0.5,
    c1::Real = 1e-4,
    iterations::Integer = 1000,
    min_alpha::Real = 1e-12,
) = begin
    alpha_value, rho_value, c1_value, min_alpha_value =
        promote(float(alpha), float(rho), float(c1), float(min_alpha))
    SimpleBackTracking(alpha_value, rho_value, c1_value, Int(iterations), min_alpha_value)
end

function (linesearch::SimpleBackTracking)(
    objective,
    x::AbstractArray{T},
    direction::AbstractArray{T},
    _initial_alpha,
    x_new::AbstractArray{T},
    initial_value,
    initial_slope,
    alpha_max = typemax(real(T)),
) where {T}
    phi, _ = LineSearches.make_ϕ_dϕ(objective, x_new, x, direction)
    alpha = min(linesearch.alpha, alpha_max)
    for _ in 0:linesearch.iterations
        value = phi(alpha)
        if isfinite(value) &&
           value <= initial_value + linesearch.c1 * alpha * initial_slope
            return alpha, value
        end
        alpha *= linesearch.rho
        if alpha < linesearch.min_alpha
            throw(LineSearches.LineSearchException(
                "Backtracking simples atingiu o alpha mínimo sem satisfazer Armijo.",
                zero(alpha),
            ))
        end
    end
    throw(LineSearches.LineSearchException(
        "Backtracking simples atingiu o limite de iterações sem satisfazer Armijo.",
        zero(alpha),
    ))
end

# ==============================================================================
# 11. Backtracking com passo inicial dinâmico
# ==============================================================================

"""
    DynamicBackTracking(; c=1e-4, p=2, rho=0.5, iterations=1000, min_alpha=1e-12)

Backtracking com interpolação quadrática (mesmo princípio de
`LineSearches.BackTracking(order = 2)` e de
[`quadratic_step_alpha`](@ref)/`quadratic_backtracking`,
`sr1_bfgs_backtracking.jl`), compatível com a interface de
`LineSearches.AbstractLineSearch` (mesmo protocolo de
[`GeometricBackTracking`](@ref)/[`SimpleBackTracking`](@ref)), com passo
inicial dinâmico

    alpha0_k = min(1, 1 / (1 + c*||g_k||^p))

em vez do `alpha` fixo de `SimpleBackTracking` ou do `initial_alpha` sugerido
pelo próprio `Optim.BFGS` (`GeometricBackTracking`). `g_k` é o gradiente no
ponto corrente (obtido de `objective`, o `NLSolversBase.OnceDifferentiable`
que o Optim passa à busca linear) e `c` é a mesma constante de Armijo usada
no teste de decréscimo suficiente `phi(alpha) <= phi(0) + c*alpha*phi'(0)`.
Passos maiores só quando `‖g_k‖` é pequena (perto da solução); quando
`‖g_k‖` é grande, `alpha0_k` encolhe, evitando passos unitários exagerados
longe do ótimo.

A cada falha do teste de Armijo, o próximo `alpha` vem da parábola ancorada
em `phi(0)`, `phi'(0)` e no último `phi(alpha)` tentado
([`quadratic_step_alpha`](@ref), limitada a `[0.1, 0.5] .* alpha` — mesma
salvaguarda do `LineSearches.jl`), em vez de simplesmente multiplicar por
`rho`; cai para o backtracking geométrico simples (`alpha *= rho`) quando a
parábola não é utilizável (curvatura não positiva, `phi(alpha)` não finita,
ou resultado não finito). A busca falha (lança
`LineSearches.LineSearchException`, mesmo tratamento de
`GeometricBackTracking`/`SimpleBackTracking`) se `alpha` cair abaixo de
`min_alpha` ou após `iterations` tentativas.
"""
struct DynamicBackTracking{T} <: LineSearches.AbstractLineSearch
    c::T
    p::T
    rho::T
    iterations::Int
    min_alpha::T
end

DynamicBackTracking(;
    c::Real = 1e-4,
    p::Real = 2,
    rho::Real = 0.5,
    iterations::Integer = 1000,
    min_alpha::Real = 1e-12,
) = begin
    c_value, p_value, rho_value, min_alpha_value =
        promote(float(c), float(p), float(rho), float(min_alpha))
    DynamicBackTracking(c_value, p_value, rho_value, Int(iterations), min_alpha_value)
end

function (linesearch::DynamicBackTracking)(
    objective,
    x::AbstractArray{T},
    direction::AbstractArray{T},
    initial_alpha,
    x_new::AbstractArray{T},
    initial_value,
    initial_slope,
    alpha_max = typemax(real(T)),
) where {T}
    phi, _ = LineSearches.make_ϕ_dϕ(objective, x_new, x, direction)

    # O Optim sempre embrulha o `OnceDifferentiable` em um `ManifoldObjective`
    # (mesmo com a manifold trivial `Flat`) antes de chamar a busca linear.
    # `gradient(...)` sem reavaliar exige o objeto interno.
    inner = objective isa Optim.ManifoldObjective ? objective.inner_obj : objective
    gnorm = norm(LineSearches.NLSolversBase.gradient(inner))
    alpha0 = min(one(linesearch.c), 1 / (1 + linesearch.c * gnorm^linesearch.p))
    alpha = min(alpha0, alpha_max)

    for _ in 0:linesearch.iterations
        value = phi(alpha)
        if isfinite(value) &&
           value <= initial_value + linesearch.c * alpha * initial_slope
            return alpha, value
        end
        # Interpolação quadrática (ordem 2): ancora a parábola em phi(0),
        # phi'(0) e no phi(alpha) que acabou de falhar o teste de Armijo, e
        # usa o minimizador dela como próximo alpha — só cai para o
        # backtracking geométrico (`alpha *= rho`) quando essa parábola não é
        # utilizável.
        alpha_quad = isfinite(value) ?
            quadratic_step_alpha(alpha, initial_value, initial_slope, value) :
            nothing
        alpha = alpha_quad === nothing ? alpha * linesearch.rho : alpha_quad
        if alpha < linesearch.min_alpha
            # `Optim.BFGS`'s `update_state!` aplica `state.x += state.alpha*state.s`
            # incondicionalmente, mesmo quando a busca linear lança esta exceção
            # (ele só lê `ex.alpha` de volta e segue em frente) — ver
            # `perform_linesearch!`/`update_state!` em Optim.jl. Se aqui
            # devolvêssemos o último `alpha` tentado (não nulo), um passo que
            # nunca satisfez Armijo seria silenciosamente aceito como resultado
            # final. Lançar com `alpha = 0` torna esse passo forçado um no-op,
            # deixando o Optim parar no último ponto de fato válido.
            throw(LineSearches.LineSearchException(
                "Backtracking dinâmico atingiu o alpha mínimo sem satisfazer Armijo.",
                zero(alpha),
            ))
        end
    end
    throw(LineSearches.LineSearchException(
        "Backtracking dinâmico atingiu o limite de iterações sem satisfazer Armijo.",
        zero(alpha),
    ))
end
function bfgs_original(
    x0::AbstractVector;
    maxiter::Integer = 100,
    f_calls_limit::Integer = 200,
    g_calls_limit::Integer = 100,
    g_tol::Real = 1e-3,
    x_tol::Real = 0.0,
    alpha_min::Real = 1e-12,
    penalty_weight::Real = 1e6,
    lower::AbstractVector = zeros(length(x0)),
    upper::AbstractVector = fill(0.5, length(x0)),
    show_trace::Bool = false,
)   

    start_time_ns = time_ns()
    x = collect(float.(x0))
    dim = length(x)
    lb = collect(float.(lower))
    ub = collect(float.(upper))
    length(lb) == dim || throw(DimensionMismatch("lower e x0 devem ter o mesmo tamanho"))
    length(ub) == dim || throw(DimensionMismatch("upper e x0 devem ter o mesmo tamanho"))
    all(lb .< ub) || throw(ArgumentError("cada limite inferior deve ser menor que o superior"))
    alpha_min >= 0 || throw(ArgumentError("alpha_min deve ser não negativo"))
    penalty_weight > 0 || throw(ArgumentError("penalty_weight deve ser positivo"))

    function_evaluations = Ref(0)
    gradient_evaluations = Ref(0)
    function_evaluation_time_seconds = Ref(0.0)
    gradient_evaluation_time_seconds = Ref(0.0)
    latest_x = Ref{Any}(nothing)
    latest_residual = Ref{Any}(nothing)
    latest_gradient_x = Ref{Any}(nothing)
    latest_gradient = Ref{Any}(nothing)

    raw_residual(x) = collect(sv_fork_assimilation(x, 0.0, 31.0, nothing))
    # function box_penalty(x)
    #     penalty = zero(eltype(x))
    #     for i in eachindex(x)
    #         below = max(zero(x[i]), lb[i] - x[i])
    #         above = max(zero(x[i]), x[i] - ub[i])
    #         penalty += below^2 + above^2
    #     end
    #     return penalty_weight * penalty
    # end
    function raw_objective(x)
        sse = sum(abs2, raw_residual(x))
        value = sse #+ box_penalty(x)
        return isnan(value) ? oftype(value, 1e26) : value
    end
    function objective(x)
        start_ns = time_ns()
        residual = raw_residual(x)
        sse = sum(abs2, residual)
        value = sse# + box_penalty(x)
        value = isnan(value) ? oftype(value, 1e26) : value
        function_evaluations[] += 1
        function_evaluation_time_seconds[] += (time_ns() - start_ns) / 1e9
        latest_x[] = copy(x)
        latest_residual[] = residual
        return value
    end

    config = ForwardDiff.GradientConfig(
        raw_objective,
        x,
        ForwardDiff.Chunk{dim}(),
    )
    function grad!(G, x)
        start_ns = time_ns()
        ForwardDiff.gradient!(G, raw_objective, x, config)
        gradient_evaluations[] += 1
        gradient_evaluation_time_seconds[] += (time_ns() - start_ns) / 1e9
        latest_gradient_x[] = copy(x)
        latest_gradient[] = copy(G)
        return G
    end

    callback_calls = Ref(0)
    alpha_too_small = Ref(false)
    stop_on_small_alpha = function (state)
        callback_calls[] += 1
        if callback_calls[] > 1 && state.alpha < alpha_min
            alpha_too_small[] = true
            return true
        end
        return false
    end

    options = Optim.Options(
        iterations = Int(maxiter),
        f_calls_limit = Int(f_calls_limit),
        g_calls_limit = Int(g_calls_limit),
        g_abstol = g_tol,
        x_abstol = x_tol,
        show_trace = show_trace,
        store_trace = true,
        extended_trace = true,
        callback = stop_on_small_alpha,
    )
    result = Optim.optimize(objective, grad!, x, Optim.BFGS(), options)
    minimizer = copy(Optim.minimizer(result))
    final_residual = if latest_x[] !== nothing && latest_x[] == minimizer
        copy(latest_residual[])
    else
        start_ns = time_ns()
        values = raw_residual(minimizer)
        function_evaluations[] += 1
        function_evaluation_time_seconds[] += (time_ns() - start_ns) / 1e9
        values
    end
    final_gradient = if latest_gradient_x[] !== nothing && latest_gradient_x[] == minimizer
        copy(latest_gradient[])
    else
        # Fallback raro: o Optim normalmente termina no último ponto onde
        # solicitou o gradiente.
        G = similar(minimizer)
        grad!(G, minimizer)
    end

    total_function_evaluation_time_seconds = function_evaluation_time_seconds[]
    total_gradient_evaluation_time_seconds = gradient_evaluation_time_seconds[]
    mean_function_evaluation_time_seconds = function_evaluations[] == 0 ? 0.0 :
        total_function_evaluation_time_seconds / function_evaluations[]
    mean_gradient_evaluation_time_seconds = gradient_evaluations[] == 0 ? 0.0 :
        total_gradient_evaluation_time_seconds / gradient_evaluations[]
    converged = !alpha_too_small[] && Optim.converged(result)
    status = alpha_too_small[] ? :alpha_too_small : Symbol(string(result.termination_code))
    execution_time_seconds = (time_ns() - start_time_ns) / 1e9
    alphas = [
        state.metadata["Current step size"]
        for state in Optim.trace(result) if state.iteration > 0
    ]

    return (;
        minimizer,
        minimum = Optim.minimum(result),
        residual = final_residual,
        gradient = final_gradient,
        hessians = nothing,
        iterations = Optim.iterations(result),
        converged,
        status,
        execution_time_seconds,
        rejected_directions = 0,
        alphas,
        solution = result,
        u = minimizer,
        objective = Optim.minimum(result),
        stats = result,
        retcode = result.termination_code,
        function_evaluations = function_evaluations[],
        gradient_evaluations = gradient_evaluations[],
        residual_evaluations = function_evaluations[] + gradient_evaluations[],
        function_evaluation_time_seconds = mean_function_evaluation_time_seconds,
        gradient_evaluation_time_seconds = mean_gradient_evaluation_time_seconds,
        total_function_evaluation_time_seconds,
        total_gradient_evaluation_time_seconds,
    )
end