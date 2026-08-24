# ==============================================================================
# 1. Dependências
#
# Reaproveita de `ffjm2_trust.jl` (que já inclui `ffjm2.jl`) o avaliador do
# modelo quártico, a resolução do subproblema restrito a uma caixa
# (`_ffjm2_trust_model_direction`) e a atualização quase-Newton por modelo
# (`_ffjm2_update!`, que já sabe fazer `:bfgs`, `:sr1`, `:is_bfgs`,
# `:dw_model`, `:is_dw_model` e `:psb`).
# ==============================================================================

using ForwardDiff
using LinearAlgebra
using Random
import MathOptInterface as MOI

if !isdefined(@__MODULE__, :_ffjm2_trust_model_direction)
    include("ffjm2_trust.jl")
end

const FFJM2_ENSEMBLE_TRUST_MODELS = (:bfgs, :sr1, :is_bfgs, :dw_model, :is_dw_model, :psb)

# ==============================================================================
# 2. Método externo FFJM2 com região de confiança e conjunto de modelos
# quase-Newton
#
# Mesma ideia de `ffjm2_ensemble` (conjunto de Hessianas H[i] em paralelo,
# uma por modelo quase-Newton de `models`; a cada iteração externa, a
# direção usada é a de maior descida entre os candidatos, isto é, a que
# minimiza ∇f(xₖ)ᵀd), só que embutida no laço de região de confiança de
# `ffjm2_trust` em vez da busca linear de Armijo de `ffjm2_box`:
#
#   (a) resolve o subproblema quártico em caixa ‖d‖∞≤Δₖ uma vez para cada
#       conjunto de Hessianas, obtendo uma direção candidata por modelo;
#   (b) a direção usada é a de maior descida entre as candidatas;
#   (c) essa direção passa pelo teste de razão ρₖ = (f(xₖ)-f(xₖ+dₖ)) /
#       (Mₖ(0)-Mₖ(dₖ)) de `ffjm2_trust` — se rejeitada, Δₖ encolhe e (a)-(b)
#       se repetem com o raio menor, até um passo ser aceito ou Δₖ cair
#       abaixo de `trust_region_min`;
#   (d) o passo aceito atualiza todos os conjuntos de Hessianas, cada um com
#       sua própria fórmula, usando o mesmo par (s, y).
#
# Custa aproximadamente `length(models)` vezes o tempo de subproblema de
# `ffjm2_trust` por tentativa de raio (cada encolhimento de Δₖ resolve de
# novo para todos os modelos). É uma variante experimental para comparar o
# quanto os modelos concordam entre si (ver `last_model_directions` no
# retorno), não uma tentativa de ser mais rápida.
# ==============================================================================

"""
    ffjm2_ensemble_trust(F, x0; models=FFJM2_ENSEMBLE_TRUST_MODELS, jacobian=nothing, kwargs...)

Variante de [`ffjm2_trust`](@ref) que, em vez de um único `update`, mantém um
conjunto `models` (por padrão os 6 modelos de `_ffjm2_update!`: `:bfgs`,
`:sr1`, `:is_bfgs`, `:dw_model`, `:is_dw_model`, `:psb`) de Hessianas `H[i]`
em paralelo. A cada tentativa de raio da região de confiança, o subproblema
quártico em caixa `‖d‖∞≤Δₖ` é resolvido uma vez por modelo, e a direção usada
é a de maior descida entre as candidatas, ou seja, a que minimiza
`∇f(xₖ)ᵀd` (mais negativa).

A aceitação do passo usa o teste de razão `ρₖ` de [`ffjm2_trust`](@ref)
(`ρₖ < trust_region_eta1` rejeita e encolhe `Δₖ *= trust_region_shrink`,
repetindo para todos os modelos). Mas o **crescimento** de `Δₖ` é diferente:
como a direção aceita é a de maior descida entre vários modelos (não o
minimizador de caixa de um único modelo), ela raramente satura a borda
`‖d‖∞=Δₖ` — a regra clássica de Nocedal–Wright (crescer só quando o passo
bate na borda) quase nunca dispara aqui, e `Δₖ` fica preso pequeno mesmo com
`ρₖ` excelente. Em vez disso, `Δₖ` da próxima iteração é recalculado como
`min(max(‖sₖ₋₁‖, trust_region_floor), trust_region_max)` — o mesmo piso de
[`ffjm2_box`](@ref)/[`ffjm2_ensemble`](@ref), que nunca deixa a caixa
efetiva do subproblema ficar menor que `trust_region_floor`, dando aos
modelos espaço para propor passos maiores mesmo depois de uma sequência de
passos pequenos.

Cada conjunto de Hessianas é atualizado com sua própria fórmula (via
`_ffjm2_update!`), usando o mesmo passo aceito e a mesma diferença de
Jacobiana para todos.

O retorno segue o mesmo formato de [`ffjm2_trust`](@ref), exceto que
`hessians` é um `Dict{Symbol}` (uma entrada por modelo em vez de um único
vetor) e há dois campos extras: `last_model_directions` (`Dict{Symbol}` com a
última direção individual de cada modelo, antes da seleção) e
`direction_source` (o modelo vencedor na última iteração aceita).
"""
function ffjm2_ensemble_trust(
    F,
    x0::AbstractVector;
    models::Tuple = FFJM2_ENSEMBLE_TRUST_MODELS,
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
    trust_region_floor::Real = 1.0,
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
    !isempty(models) || throw(ArgumentError("models não pode ser vazio"))
    all(model -> model in (:bfgs, :sr1, :is_bfgs, :dw_model, :is_dw_model, :psb), models) ||
        throw(ArgumentError(
            "cada elemento de models deve ser :bfgs, :sr1, :is_bfgs, :dw_model, :is_dw_model ou :psb",
        ))
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
    isfinite(trust_region_floor) && trust_region_floor > 0 ||
        throw(ArgumentError("trust_region_floor deve ser finito e positivo"))
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
            "ffjm2_ensemble_trust iter 0: ",
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
    trust_region_radii = T[]
    Δ = T(trust_region_initial)
    last_ratio = T(NaN)
    last_shrinks = 0
    last_direction_norm = zero(T)
    last_direction_source = :model
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
                "ffjm2_ensemble_trust iter $k: ",
                "f = $f, RMSD = $residual_rms, ",
                "Δ = $Δ, ρ = $last_ratio, encolhimentos = $last_shrinks, ",
                "‖d‖ = $last_direction_norm, fonte da direção = $last_direction_source",
            )
        end
        state = (; iteration = k, x = copy(x), value = f, residual = copy(r),
                 residual_rms, gradient = copy(g), gradient_norm = gnorm,
                 trust_region_radius = Δ,
                 ratio = k == 0 ? nothing : last_ratio,
                 shrinks = k == 0 ? 0 : last_shrinks,
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

        # 2.3.2. Subproblema com região de confiança: resolve uma vez por
        # modelo, seleciona a direção de maior descida e testa a razão ρₖ,
        # encolhendo Δₖ (e repetindo para todos os modelos) enquanto o passo
        # for rejeitado.
        shrinks = 0
        accepted = false
        d = zeros(T, n)
        direction_source = :model
        xnew = x
        rnew = r
        fnew = f
        best_model_result = nothing
        ratio = T(NaN)
        while true
            best_directional_derivative = T(Inf)
            best_direction = nothing
            for model in models
                model_result = _ffjm2_trust_model_direction(
                    r,
                    J,
                    H[model],
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
                last_model_directions[model] = model_result.direction
                if show_trace
                    println(
                        "  Subproblema TR ($model): Δ = $Δ | ", model_result.stop_reason,
                        " | Mₖ(x̄) = ", model_result.minimum,
                    )
                end
                dtg = dot(g, model_result.direction)
                if all(isfinite, model_result.direction) && dtg < best_directional_derivative
                    best_directional_derivative = dtg
                    best_direction = model_result.direction
                    best_model_result = model_result
                    direction_source = model
                end
            end
            d = best_direction === nothing ? -g : best_direction
            predicted_reduction = f - (best_model_result === nothing ? f : best_model_result.minimum)

            xnew = x .+ d
            rnew = T.(residual(xnew))
            fnew = T(0.5) * dot(rnew, rnew)
            actual_reduction = f - fnew

            ratio = predicted_reduction > 0 ?
                actual_reduction / predicted_reduction :
                (actual_reduction > 0 ? T(Inf) : T(-Inf))

            if show_trace
                println(
                    "  TR: Δ = $Δ | fonte = $direction_source | red. prevista = ",
                    predicted_reduction, " | red. real = ", actual_reduction,
                    " | ρ = ", ratio,
                )
            end

            if ratio >= trust_region_eta1
                accepted = true
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
        last_direction_source = direction_source

        if !accepted
            push!(trust_region_radii, Δ)
            status = :trust_region_stalled
            iterations = k
            break
        end

        # 2.3.3. Atualização de cada conjunto de Hessianas, e do estado externo.
        Jnew = T.(jac(xnew))
        s = xnew - x
        # Em vez de só crescer Δ quando o passo aceito satura a caixa (regra
        # de Nocedal-Wright, frágil aqui — ver docstring), usa o mesmo piso
        # de `ffjm2_box`/`ffjm2_ensemble`: a caixa da próxima iteração nunca
        # fica menor que `trust_region_floor`, dando aos modelos espaço para
        # propor passos maiores mesmo após uma sequência de passos pequenos.
        Δ = min(max(norm(s), T(trust_region_floor)), T(trust_region_max))
        push!(trust_region_radii, Δ)
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
        trust_region_radii,
        model_solver,
        model_solves,
        model_iterations,
        model_solve_time_seconds,
        last_model_directions,
        direction_source = last_direction_source,
    )
end
