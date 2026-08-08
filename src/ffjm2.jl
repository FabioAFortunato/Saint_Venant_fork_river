# ==============================================================================
# 1. Dependências
# ==============================================================================

using ForwardDiff
using Ipopt
using LinearAlgebra
using LineSearches
using OptimizationBase
using OptimizationLBFGSB
using Optim
using Random
import MathOptInterface as MOI

if !isdefined(@__MODULE__, :sr1_backtracking) ||
   !isdefined(@__MODULE__, :bfgs_backtracking)
    include("sr1_bfgs_backtracking.jl")
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
        relative_gnorm = gnorm / size(x, 1)
        residual_rms = norm(r) / sqrt(length(r))
        if show_trace && k > 0
            println(
                "ffjm2 ($(uppercase(string(update)))) iter $k: ",
                "f = $f, RMSD = $residual_rms, ",
                "||d|| = $last_direction_norm, ||g||/n = $relative_gnorm, ",
                "α = $last_alpha, backtrackings = $last_backtrackings, ",
                "direção rejeitada = $last_direction_rejected",
            )
        end
        state = (; iteration = k, x = copy(x), value = f, residual = copy(r),
                 residual_rms, gradient = copy(g), gradient_norm = gnorm,
                 relative_gradient_norm = relative_gnorm,
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
        if relative_gnorm <= g_tol
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
            )
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

    # 6.2. Resolve uma cópia irrestrita do subproblema para cada ponto inicial.
    solve_start_ns = time_ns()
    results = map(starts) do initial
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
    end
    # 6.3. Seleciona o menor valor encontrado entre as soluções locais.
    best_start = argmin(getproperty.(results, :minimum))
    result = results[best_start]
    solve_time_seconds = (time_ns() - solve_start_ns) / 1e9

    return (
        direction = result.minimizer,
        model_minimizer = result.minimizer,
        stop_reason = string(result.status),
        iterations = result.iterations,
        minimum = result.minimum,
        solve_time_seconds,
        best_start,
        starts = length(starts),
    )
end

# ==============================================================================
# 6. Rotina experimental de comparação
#
# Executa as variantes FFJM2 e três métodos quase-Newton de referência para
# uma única instância e grava as métricas em um arquivo TSV.
# ==============================================================================

"""
    comparar_ffjm2_bfgs(; kwargs...)

Compara `ffjm2` com atualizações BFGS e SR1, `bfgs_puro_penalizado`,
`bfgs_backtracking` e `sr1_backtracking`. Salva SSE, RMSD, norma do gradiente
da SSE, avaliações externas, chamadas da função e do gradiente, tempos e
solução em um arquivo TSV.
"""
function comparar_ffjm2_bfgs(;
    simulation = sv_fork_assimilation,
    output = normpath(joinpath(@__DIR__, "..", "results", "comparacao_ffjm2_bfgs.tsv")),
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
    initial = fill(0.09, 10)
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

        for update in (:bfgs, :sr1), model_multistart in multistarts
            method_name = "ffjm2_$(update)"
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
        pure_external_evaluations = Ref(0)
        function F_bfgs_puro(x)
            pure_external_evaluations[] += 1
            return F_residual(x)
        end
        pure_result = bfgs_puro_penalizado(F_bfgs_puro, initial; pure_options...)
        pure_x = copy(pure_result.minimizer)
        pure_metrics = raw_metrics(F_residual, pure_x)
        pure_row = (;
            tend,
            dimension = dim,
            test,
            method = "bfgs_puro_penalizado",
            model_multistart = 0,
            pure_metrics...,
            external_evaluations = pure_external_evaluations[],
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
        maxnef = Int(get(pure_options, :f_calls_limit, 200))

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
                # for i in eachindex(x)
                #     below = max(zero(x[i]), lower[i] - x[i])
                #     above = max(zero(x[i]), x[i] - upper[i])
                #     penalty += below^2 + above^2
                # end
                value = sum(abs2, residual) #+ penalty_weight * penalty
                function_calls[] += 1
                function_time[] += (time_ns() - start_ns) / 1e9
                return isnan(value) ? oftype(value, 1e26) : value
            end
            raw_objective(x) = begin
                external_evaluations[] += 1
                residual = F_residual(x)
                #penalty = zero(eltype(x))
                # for i in eachindex(x)
                #     below = max(zero(x[i]), lower[i] - x[i])
                #     above = max(zero(x[i]), x[i] - upper[i])
                #     penalty += below^2 + above^2
                # end
                sum(abs2, residual)# + penalty_weight * penalty
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

# ==============================================================================
# 6.1. Solver L-BFGS-B de referência
#
# Esta função não faz parte do FFJM2. Ela existe para produzir uma solução de
# comparação usando L-BFGS-B e limites de caixa nativos.
# ==============================================================================

"""
    bfgs_puro(F, x0; lower=zeros(length(x0)), upper=fill(0.5, length(x0)), kwargs...)

Minimiza a soma dos quadrados com limites de caixa nativos, usando o solver
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
    F,
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

    raw_residual(x) = collect(F(x))

    function raw_objective(x)
        sse = sum(abs2, raw_residual(x))
        return isnan(sse) ? oftype(sse, 1e26) : sse
    end

    function objective(x, _)
        start_ns = time_ns()
        residual = raw_residual(x)
        sse = sum(abs2, residual)
        value = isnan(sse) ? oftype(sse, 1e26) : sse
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

    config = ForwardDiff.GradientConfig(
        raw_objective,
        x,
        ForwardDiff.Chunk{dim}(),
    )
    function grad!(G, x, _)
        start_ns = time_ns()
        ForwardDiff.gradient!(G, raw_objective, x, config)
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
    bfgs_puro_penalizado(F, x0; kwargs...)

Minimiza a soma dos quadrados dos resíduos com `Optim.BFGS()` e acrescenta
uma penalidade quadrática quando alguma variável viola `lower` ou `upper`.
Interrompe a otimização quando o passo aceito é menor que `alpha_min`.
O retorno possui os mesmos campos de `bfgs_puro` para permitir comparações
diretas entre os dois métodos.
"""
function bfgs_puro_penalizado(
    F,
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

    raw_residual(x) = collect(F(x))
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

# ==============================================================================
# 6.3. BFGS explícito com busca linear somente de Armijo
# ==============================================================================

"""
    bfgs_puro_armijo(F, x0; kwargs...)

Executa BFGS com penalidade externa de caixa e busca linear apenas de Armijo,
com backtracking por interpolação cúbica e sem condição de Wolfe. A direção
quase-Newton `-B⁻¹∇f` somente é aceita se
satisfizer o mesmo critério de direção usado por `ffjm2`; caso contrário,
utiliza `-∇f`. O campo `alphas` do retorno contém, na ordem, o passo aceito
em cada iteração.
"""
function bfgs_puro_armijo(
    F,
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

    raw_residual(z) = collect(F(z))
    function box_penalty(z)
        value = zero(eltype(z))
        for i in eachindex(z)
            below = max(zero(z[i]), lb[i] - z[i])
            above = max(zero(z[i]), z[i] - ub[i])
            value += below^2 + above^2
        end
        return penalty_weight * value
    end
    function raw_objective(z)
        value = sum(abs2, raw_residual(z)) + box_penalty(z)
        return isnan(value) ? oftype(value, 1e26) : value
    end
    function objective(z)
        start_ns = time_ns()
        residual = raw_residual(z)
        value = sum(abs2, residual) + box_penalty(z)
        value = isnan(value) ? oftype(value, 1e26) : value
        function_evaluations[] += 1
        function_time[] += (time_ns() - start_ns) / 1e9
        latest_x[] = copy(z)
        latest_residual[] = residual
        return value
    end

    config = ForwardDiff.GradientConfig(raw_objective, x, ForwardDiff.Chunk{n}())
    function gradient!(G, z)
        start_ns = time_ns()
        ForwardDiff.gradient!(G, raw_objective, z, config)
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
