# ==============================================================================
# 1. Dependências
#
# Reaproveita de `ffjm2.jl` o avaliador do modelo quártico
# (`_FFJM2IpoptEvaluator` e as funções `MOI.eval_*` associadas) e a
# atualização quase-Newton das Hessianas individuais (`_ffjm2_update!`); o
# modelo em si não muda, só a forma como o subproblema é resolvido e como o
# passo é aceito.
# ==============================================================================

using ForwardDiff
using Ipopt
using LinearAlgebra
using NLopt
using NOMAD
using Optim
using Random
import MathOptInterface as MOI

if !isdefined(@__MODULE__, :_FFJM2IpoptEvaluator) ||
   !isdefined(@__MODULE__, :_ffjm2_update!)
    include("ffjm2.jl")
end

# ==============================================================================
# 2. Método externo FFJM2 com região de confiança
#
# Fluxo de uma iteração:
#   (a) verifica convergência;
#   (b) resolve o modelo quártico com ‖d‖∞ ≤ Δₖ (multi-start dentro da
#       mesma caixa);
#   (c) aceita ou rejeita o passo pela razão entre redução real e prevista,
#       encolhendo Δₖ e repetindo (b) enquanto o passo for rejeitado;
#   (d) expande, mantém ou já deixa Δₖ encolhido para a próxima iteração;
#   (e) atualiza as Hessianas individuais por BFGS ou SR1.
# ==============================================================================

"""
    ffjm2_trust(F, x0; update=:bfgs, jacobian=nothing, kwargs...)

Variante de [`ffjm2`](@ref) que troca a busca linear de Armijo (passo
(2.3.4) do artigo, condição (2) incluída) por uma região de confiança
clássica (Nocedal & Wright, cap. 4) em torno do modelo quártico (19)--(21).
O subproblema é resolvido com a restrição de caixa ‖d‖∞ ≤ Δₖ, e o
multi-start amostra os pontos iniciais uniformemente dentro dessa mesma
caixa (em vez de perturbações gaussianas sem limite). O passo dₖ é aceito
ou rejeitado pela razão

    ρₖ = (f(xₖ) - f(xₖ + dₖ)) / (Mₖ(0) - Mₖ(dₖ))

entre a redução real da função verdadeira e a redução prevista pelo
modelo (`Mₖ(0) = f(xₖ)` sempre, pois `qᵢ(0) = rᵢ`). Se `ρₖ < trust_region_eta1`,
o passo é rejeitado, `Δₖ` é multiplicado por `trust_region_shrink` e o
subproblema é resolvido de novo com o raio menor; isso se repete até um
passo ser aceito ou até `Δₖ` cair abaixo de `trust_region_min` (ou o número
de tentativas passar de `trust_region_max_shrinks`), quando a iteração
externa para com `status = :trust_region_stalled`. Se o passo é aceito e
`ρₖ ≥ trust_region_eta2` com `dₖ` na borda da região, `Δₖ` é expandido por
`trust_region_expand` (limitado por `trust_region_max`) para a próxima
iteração; caso contrário `Δₖ` é mantido.

Como a globalização passa a ser inteiramente feita pela região de
confiança, a condição (2) do artigo e a extensão de bola pesada de
[`ffjm2`](@ref) não se aplicam aqui.

`H[i]`, a Jacobiana e a atualização quase-Newton das Hessianas individuais
seguem exatamente [`ffjm2`](@ref) (`update` aceita `:bfgs`, `:sr1`,
`:is_bfgs`, `:dw_model`, `:is_dw_model` ou `:psb` — ver `_ffjm2_update!` em
`ffjm2.jl` para a derivação de cada fórmula; `gamma_bar` controla o
intervalo de escala usado por `:is_bfgs`/`:is_dw_model`; `initial_scaling`
ativa a escala inicial de Shanno & Phua na primeiríssima atualização, ver
[`ffjm2`](@ref)).

`F` deve receber um vetor e devolver diretamente um vetor de resíduos. Se
`jacobian` não for fornecida, a Jacobiana é calculada com
`ForwardDiff.jacobian`.

O retorno segue o mesmo formato de [`ffjm2`](@ref), com `trust_region_radii`
(raio Δₖ ao final de cada iteração aceita) no lugar de `alphas`, e
`rejected_directions` contando passos rejeitados pelo teste de razão (em
vez de direções rejeitadas pela condição (2)).
"""
function ffjm2_trust(
    F,
    x0::AbstractVector;
    update::Symbol = :bfgs,
    jacobian = nothing,
    maxiter::Integer = 1000,
    g_tol::Real = 1e-8,
    residual_rms_tol::Union{Nothing,Real} = 0.0,
    model_maxiter::Integer = 1000,
    model_g_tol::Real = 1e-6,
    model_multistart::Integer = 10,
    model_seed::Integer = 1234,
    model_solver::Symbol = :ipopt,
    trust_region_initial::Real = 1.0,
    trust_region_min::Real = 1e-12,
    trust_region_max::Real = 1e6,
    trust_region_eta1::Real = 0.1,
    trust_region_eta2::Real = 0.75,
    trust_region_shrink::Real = 0.5,
    trust_region_expand::Real = 2.0,
    trust_region_max_shrinks::Integer = 30,
    gamma_bar::Real = 10.0,
    initial_scaling::Bool = true,
    update_tol::Real = sqrt(eps(Float64)),
    callback = nothing,
    show_trace::Bool = false,
)
    # --------------------------------------------------------------------------
    # 2.1. Validação dos parâmetros
    # --------------------------------------------------------------------------
    start_time_ns = time_ns()
    update in (:bfgs, :sr1, :is_bfgs, :dw_model, :is_dw_model, :psb) ||
        throw(ArgumentError("update deve ser :bfgs, :sr1, :is_bfgs, :dw_model, :is_dw_model ou :psb"))
    gamma_bar > 1 || throw(ArgumentError("gamma_bar deve ser maior que 1"))
    model_solver in (:ipopt, :bfgs, :bobyqa, :mads) ||
        throw(ArgumentError("model_solver deve ser :ipopt, :bfgs, :bobyqa ou :mads"))
    maxiter >= 0 || throw(ArgumentError("maxiter deve ser não negativo"))
    model_multistart >= 1 ||
        throw(ArgumentError("model_multistart deve ser pelo menos 1"))
    isfinite(trust_region_initial) && trust_region_initial > 0 ||
        throw(ArgumentError("trust_region_initial deve ser finito e positivo"))
    isfinite(trust_region_min) && trust_region_min > 0 ||
        throw(ArgumentError("trust_region_min deve ser finito e positivo"))
    isfinite(trust_region_max) && trust_region_max >= trust_region_min ||
        throw(ArgumentError("trust_region_max deve ser finito e >= trust_region_min"))
    0 < trust_region_eta1 < trust_region_eta2 < 1 ||
        throw(ArgumentError("é preciso 0 < trust_region_eta1 < trust_region_eta2 < 1"))
    0 < trust_region_shrink < 1 ||
        throw(ArgumentError("trust_region_shrink deve pertencer a (0, 1)"))
    trust_region_expand > 1 ||
        throw(ArgumentError("trust_region_expand deve ser maior que 1"))
    trust_region_max_shrinks >= 1 ||
        throw(ArgumentError("trust_region_max_shrinks deve ser pelo menos 1"))

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
            "ffjm2_trust ($(uppercase(string(update)))) iter 0: ",
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
    trust_region_radii = T[]
    Δ = T(trust_region_initial)
    # O subproblema só é resolvido até `model_g_tol` (padrão 1e-6); exigir
    # ‖d‖∞ a menos de 1e-8 de Δ para considerar o passo "na borda" era mais
    # apertado do que a própria precisão do solver, então a condição quase
    # nunca disparava e Δ só encolhia, nunca crescia de volta (ver mgh19).
    boundary_tol = T(max(model_g_tol, 1e-8))
    last_ratio = T(NaN)
    last_shrinks = 0
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
                "ffjm2_trust ($(uppercase(string(update)))) iter $k: ",
                "f = $f, RMSD = $residual_rms, ",
                "Δ = $Δ, ρ = $last_ratio, encolhimentos = $last_shrinks, ",
                "‖d‖ = $last_direction_norm",
            )
        end
        state = (; iteration = k, x = copy(x), value = f, residual = copy(r),
                 residual_rms, gradient = copy(g), gradient_norm = gnorm,
                 trust_region_radius = Δ,
                 ratio = k == 0 ? nothing : last_ratio,
                 shrinks = k == 0 ? 0 : last_shrinks,
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

        # 2.3.2. Subproblema com região de confiança: resolve, testa a razão
        # ρₖ e encolhe Δₖ enquanto o passo for rejeitado.
        shrinks = 0
        accepted = false
        d = zeros(T, n)
        xnew = x
        rnew = r
        fnew = f
        model_result = nothing
        ratio = T(NaN)
        while true
            model_result = _ffjm2_trust_model_direction(
                r,
                J,
                H,
                Δ,
                model_maxiter,
                model_g_tol,
                model_multistart,
                model_seed + k,
                model_solver,
            )
            model_solves += 1
            model_iterations += model_result.total_iterations
            model_solve_time_seconds += model_result.solve_time_seconds
            d = model_result.direction
            predicted_reduction = f - model_result.minimum

            xnew = x .+ d
            rnew = T.(residual(xnew))
            fnew = T(0.5) * dot(rnew, rnew)
            actual_reduction = f - fnew

            ratio = predicted_reduction > 0 ?
                actual_reduction / predicted_reduction :
                (actual_reduction > 0 ? T(Inf) : T(-Inf))

            if show_trace
                println(
                    "  Subproblema TR: Δ = $Δ | Ipopt multi-start interno: ",
                    model_result.stop_reason,
                    " | melhor início = ", model_result.best_start,
                    "/", model_result.starts,
                    " | Mₖ(x̄) = ", model_result.minimum,
                    " | red. prevista = ", predicted_reduction,
                    " | red. real = ", actual_reduction,
                    " | ρ = ", ratio,
                    " | tempo do subproblema = ",
                    round(model_result.solve_time_seconds; digits = 6), " s",
                )
            end

            if ratio >= trust_region_eta1
                accepted = true
                if ratio >= trust_region_eta2 && norm(d, Inf) >= Δ * (1 - boundary_tol)
                    Δ = min(Δ * T(trust_region_expand), T(trust_region_max))
                end
                break
            end

            rejected_directions += 1
            shrinks += 1
            Δ *= T(trust_region_shrink)
            if Δ < trust_region_min || shrinks >= trust_region_max_shrinks
                break
            end
        end
        last_ratio = ratio
        last_shrinks = shrinks
        last_direction_norm = norm(d)
        push!(trust_region_radii, Δ)

        if !accepted
            status = :trust_region_stalled
            iterations = k
            break
        end

        # 2.3.3. Atualização das Hessianas individuais e do estado externo.
        Jnew = T.(jac(xnew))
        s = xnew - x
        if initial_scaling && k == 0
            alpha0 = norm(s) / gnorm
            for Hi in H
                Hi .= alpha0 .* Matrix{T}(I, n, n)
            end
        end
        _ffjm2_update!(H, s, Jnew, J, update, update_tol, gamma_bar)

        x, r, J, f = xnew, rnew, Jnew, T(fnew)
        g = J' * r
        iterations = k + 1
    end

    # --------------------------------------------------------------------------
    # 2.4. Resultado do método externo
    # --------------------------------------------------------------------------
    converged = status in (:residual_converged, :gradient_converged)
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
        trust_region_radii,
        model_solver,
        model_solves,
        model_iterations,
        model_solve_time_seconds,
    )
end

# ==============================================================================
# 3. Multi-start do subproblema quártico dentro da região de confiança
#
# Reaproveita o avaliador `_FFJM2IpoptEvaluator` de `ffjm2.jl` (o modelo
# Mₖ(d) não muda); a região de confiança entra como restrição de caixa
# ‖d‖∞ ≤ Δ nas variáveis do subproblema, igual para todo `solver`, e como
# raio de amostragem do multi-start (uniforme em [-Δ, Δ]ⁿ, em vez das
# perturbações gaussianas sem limite de `ffjm2`).
# ==============================================================================

function _ffjm2_trust_model_direction(
    r,
    J,
    H,
    Δ,
    maxiter,
    g_tol,
    multistart,
    seed,
    solver,
)
    n = size(J, 2)
    evaluator = _FFJM2IpoptEvaluator(collect(r), Matrix(J), H)
    rng = MersenneTwister(seed)
    Δf = Float64(Δ)
    starts = Vector{Vector{eltype(r)}}(undef, Int(multistart))
    starts[1] = zeros(eltype(r), n)
    for i in 2:Int(multistart)
        starts[i] = Δ .* (2 .* rand(rng, eltype(r), n) .- 1)
    end

    model_value(d) = MOI.eval_objective(evaluator, d)
    function model_gradient!(gradient, d)
        MOI.eval_objective_gradient(evaluator, gradient, d)
        return gradient
    end

    # 3.1. Resolve uma cópia do subproblema, restrita a ‖d‖∞ ≤ Δ, para cada
    # ponto inicial.
    solve_start_ns = time_ns()
    results = map(starts) do initial
        if solver === :ipopt
            optimizer = Ipopt.Optimizer()
            MOI.set(optimizer, MOI.Silent(), true)
            MOI.set(optimizer, MOI.RawOptimizerAttribute("max_iter"), Int(maxiter))
            MOI.set(optimizer, MOI.RawOptimizerAttribute("tol"), Float64(g_tol))
            variables = MOI.add_variables(optimizer, n)
            MOI.set.(optimizer, MOI.VariablePrimalStart(), variables, initial)
            MOI.add_constraint.(optimizer, variables, MOI.Interval(-Δf, Δf))
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
            lower = fill(-Δf, n)
            upper = fill(Δf, n)
            options = Optim.Options(
                iterations = Int(maxiter),
                g_abstol = g_tol,
                show_trace = false,
            )
            result = Optim.optimize(
                model_value, model_gradient!, lower, upper, initial,
                Optim.Fminbox(Optim.BFGS()), options,
            )
            return (;
                minimizer = copy(Optim.minimizer(result)),
                minimum = Optim.minimum(result),
                iterations = Optim.iterations(result),
                status = result.termination_code,
            )
        elseif solver === :bobyqa
            optimizer = NLopt.Opt(:LN_BOBYQA, n)
            optimizer.lower_bounds = fill(-Δf, n)
            optimizer.upper_bounds = fill(Δf, n)
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
                lower_bound = fill(-Δf, n),
                upper_bound = fill(Δf, n),
                min_mesh_size = fill(max(Float64(g_tol), eps(Float64)), n),
                initial_mesh_size = fill(max(Δf, 1e-2), n),
                options = options,
            )
            result = NOMAD.solve(problem, initial)
            minimizer = result.x_sol === nothing ? copy(best_point) : collect(result.x_sol)
            minimum = model_value(minimizer)
            return (; minimizer, minimum, iterations = evaluations[], status = result.status)
        end
    end
    # 3.2. Seleciona o menor valor encontrado entre as soluções locais.
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
