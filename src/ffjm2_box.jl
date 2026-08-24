# ==============================================================================
# 1. Dependências
#
# Reaproveita de `ffjm2_trust.jl` (que já inclui `ffjm2.jl`) o avaliador do
# modelo quártico, a atualização quase-Newton das Hessianas e a resolução do
# subproblema restrito a uma caixa (`_ffjm2_trust_model_direction`). O que
# muda aqui é como a caixa `Δ` é definida a cada iteração e como o passo é
# aceito.
# ==============================================================================

using ForwardDiff
using LinearAlgebra
using LineSearches
using Random
import MathOptInterface as MOI

if !isdefined(@__MODULE__, :_ffjm2_trust_model_direction)
    include("ffjm2_trust.jl")
end

# ==============================================================================
# 2. Método externo FFJM2 com subproblema em caixa e globalização por Armijo
#
# Híbrido entre `ffjm2` (busca linear de Armijo, condição (2), bola pesada) e
# `ffjm2_trust` (subproblema restrito a uma caixa ‖d‖∞ ≤ Δ, multi-start
# dentro dela): a região aqui só disciplina onde o modelo é resolvido e
# amostrado — quem decide aceitar ou rejeitar o passo continua sendo a
# condição (2) do artigo mais a busca linear, exatamente como em `ffjm2`. Não
# há teste de razão nem laço de encolher-e-resolver-de-novo: o subproblema é
# resolvido uma única vez por iteração.
#
# Δₖ = max(‖sₖ₋₁‖, trust_region_floor): a caixa acompanha o tamanho do
# último passo aceito sₖ₋₁ = xₖ - xₖ₋₁, com um piso `trust_region_floor` para
# não colapsar antes da hora — mesma ideia (e mesma lição do caso mgh10, ver
# `project_mgh10_hard_case` na memória) de `model_start_spread`/
# `last_direction_norm` em `ffjm2.jl`, só que aqui vira de fato um limite
# rígido do subproblema (via bounds no Ipopt/Fminbox/BOBYQA/NOMAD), não
# apenas o desvio da amostra gaussiana do multi-start.
#
# Fluxo de uma iteração:
#   (a) verifica convergência;
#   (b) resolve o modelo quártico com ‖d‖∞ ≤ Δₖ (multi-start dentro da
#       mesma caixa);
#   (c) valida a direção pela condição (2) do artigo;
#   (d) escolhe o tamanho do passo por Armijo;
#   (e) atualiza as Hessianas individuais por BFGS ou SR1.
# ==============================================================================

"""
    ffjm2_box(F, x0; update=:bfgs, jacobian=nothing, kwargs...)

Variante híbrida de [`ffjm2`](@ref) e [`ffjm2_trust`](@ref): o subproblema
quártico (19)--(21) é resolvido restrito à caixa `‖d‖∞ ≤ Δₖ`, com o
multi-start amostrado uniformemente dentro dela (mecanismo de
[`ffjm2_trust`](@ref), via `_ffjm2_trust_model_direction`), mas a
globalização volta a ser inteiramente feita pela condição (2) do artigo e
pela busca linear de Armijo (mecanismo de [`ffjm2`](@ref)) — sem teste de
razão nem laço de encolher/resolver de novo.

`Δₖ = max(‖sₖ₋₁‖, trust_region_floor)` acompanha o tamanho do último passo
aceito `sₖ₋₁` (não a direção pré-busca-linear), com piso
`trust_region_floor` (padrão `1.0`) para não colapsar. Se a direção do
modelo não satisfizer a condição (2), tenta-se a direção da bola pesada
`-∇f(xₖ) + heavy_ball_beta*sₖ₋₁` (extensão fora do artigo, desligável com
`heavy_ball_beta = nothing`) e, por fim, `-∇f(xₖ)` puro — exatamente como em
[`ffjm2`](@ref).

`H[i]` e a Jacobiana seguem exatamente [`ffjm2`](@ref). A atualização
quase-Newton aceita três `update` além de `:bfgs` e `:sr1`:

  * `:is_bfgs` — BFGS auto-escalado com teste de intervalo de Lukšan &
    Spedicato (2000, `_research/IS_BFGS.pdf`) — escala a Hessiana direta
    `H[i]` por `1/γ` a cada atualização, com `γ = (sᵀH[i]s)/(yᵀs)` testado
    no intervalo `[1/gamma_bar, gamma_bar]` (fora dele, `γ=1`, caindo de
    volta ao BFGS comum para aquela iteração/resíduo). `gamma_bar` (padrão
    `10.0`) controla esse intervalo.
  * `:dw_model` — a atualização da classe Broyden do Teorema 4.5(iii) de
    Dennis & Wolkowicz (1993, `_research/DW_model.pdf`), descrita no artigo
    como a de melhor desempenho numérico entre as testadas: atualização
    fraca de Greenstadt seguida de DFP, hereditariamente definida positiva.
  * `:is_dw_model` — combina os dois: aplica o mesmo `γ` (e mesmo teste de
    intervalo) do `:is_bfgs` à parte da atualização herdada do BFGS comum
    dentro do `:dw_model`, mantendo o termo extra do `:dw_model` sem
    escalar. Generaliza estritamente os outros dois (ver comentário de
    `_ffjm2_box_update!`).
  * `:psb` — Powell-Symmetric-Broyden (Powell, 1970), atualização simétrica
    de posto 2 clássica; assim como `:sr1`, satisfaz a equação secante mas
    não preserva definição positiva (ao contrário de `:bfgs`).

`initial_scaling` (padrão `false`) ativa a escala inicial de Shanno & Phua
(1978, `_research/initial_scale.pdf`) na primeiríssima atualização de cada
`H[i]`, ver [`ffjm2`](@ref).

O retorno segue o mesmo formato de [`ffjm2`](@ref), com `box_radii` (raio
`Δₖ` usado em cada iteração a partir de `k=1`) no lugar de `model
start_spread`.
"""
function ffjm2_box(
    F,
    x0::AbstractVector;
    update::Symbol = :sr1,
    jacobian = nothing,
    maxiter::Integer = 1000,
    g_tol::Real = 1e-8,
    residual_rms_tol::Union{Nothing,Real} = 0.0,
    model_maxiter::Integer = 1000,
    model_g_tol::Real = 1e-6,
    model_multistart::Integer = 10,
    model_seed::Integer = 1234,
    model_solver::Symbol = :ipopt,
    trust_region_floor::Real = 1.0,
    direction_theta::Real = 1e-4,
    direction_beta::Real = 1e-8,
    direction_delta::Real = Inf,
    heavy_ball_beta = 0.5,
    gamma_bar::Real = 10.0,
    initial_scaling::Bool = true,
    update_tol::Real = sqrt(eps(Float64)),
    linesearch = LineSearches.BackTracking(),
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
    isfinite(trust_region_floor) && trust_region_floor > 0 ||
        throw(ArgumentError("trust_region_floor deve ser finito e positivo"))
    0 < direction_theta < 1 ||
        throw(ArgumentError("direction_theta deve pertencer a (0, 1)"))
    direction_beta > 0 ||
        throw(ArgumentError("direction_beta deve ser positivo"))
    direction_delta > 0 ||
        throw(ArgumentError("direction_delta deve ser positivo"))
    heavy_ball_beta === nothing || (isfinite(heavy_ball_beta) && heavy_ball_beta >= 0) ||
        throw(ArgumentError("heavy_ball_beta deve ser nothing ou não negativo"))

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
            "ffjm2_box ($(uppercase(string(update)))) iter 0: ",
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
    box_radii = T[]
    last_alpha = one(T)
    last_backtrackings = 0
    last_direction_rejected = false
    last_direction_norm = zero(T)
    last_direction_source = :model
    last_step_norm = zero(T)
    heavy_ball_used = 0
    previous_step = nothing

    # --------------------------------------------------------------------------
    # 2.3. Laço principal do método externo
    # --------------------------------------------------------------------------
    for k in 0:maxiter
        # 2.3.1. Critérios de parada no início da iteração.
        gnorm = norm(g)
        residual_rms = norm(r) / sqrt(length(r))
        if show_trace && k > 0
            println(
                "ffjm2_box ($(uppercase(string(update)))) iter $k: ",
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

        # 2.3.2. Direção: gradiente negativo em k=0; modelo em caixa para k>0.
        model_result = nothing
        direction_source = k == 0 ? :gradient : :model
        if k == 0
            p = -g
        else
            Δ = max(last_step_norm, T(trust_region_floor))
            push!(box_radii, Δ)
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
            p = model_result.direction
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
            print(
                "  Avaliação externa f(x): ",
                round(external_evaluation_time_seconds; digits = 6),
                " s",
            )
            if model_result !== nothing
                print(
                    " | Δ = ", box_radii[end],
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
        if initial_scaling && k == 0
            alpha0 = norm(s) / gnorm
            for Hi in H
                Hi .= alpha0 .* Matrix{T}(I, n, n)
            end
        end
        _ffjm2_box_update!(H, s, Jnew, J, update, update_tol, gamma_bar)
        previous_step = s
        last_step_norm = norm(s)

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
        heavy_ball_used,
        alphas,
        box_radii,
        model_solver,
        model_solves,
        model_iterations,
        model_solve_time_seconds,
    )
end

# ==============================================================================
# 3. Atualização quase-Newton das Hessianas dos resíduos
#    (IS-BFGS e DW model, além de BFGS/SR1 comuns)
#
# Igual a `_ffjm2_update!` de `ffjm2.jl` para `:bfgs`/`:sr1`; acrescenta:
#
# `:is_bfgs` — BFGS auto-escalado com teste de intervalo (Lukšan & Spedicato,
# 2000, `_research/IS_BFGS.pdf`, eqs. (2.7), (2.9) e (2.17)). O artigo escala
# a Hessiana *inversa* `H`; `H[i]` aqui é a Hessiana *direta* de cada
# resíduo (usada em `qᵢ(d)=rᵢ+Jᵢd+½d'H[i]d`), então usamos a fórmula dual
# (2.17), que escala `B=H⁻¹` em vez de `H`. Com `ρ=1` e a escolha (2.9)
# `γ=c/b` (que dispensa calcular `H⁻¹` explicitamente, pois `c=s'H⁻¹s=s'Bs`
# já é a própria Hessiana direta aplicada a `s`):
#
#   b = y's,  c = s'Bs
#   γ = c/b testado no intervalo [1/gamma_bar, gamma_bar]; fora dele, γ=1
#   B₊ = (1/γ)·[B − (Bs)(Bs)'/c] + yy'/b
#
# (verificado à mão: para qualquer γ>0, B₊s=y — a equação secante direta —
# continua satisfeita; γ só afeta o condicionamento, não a correção).
#
# `:dw_model` — a atualização da classe Broyden do Teorema 4.5(iii) de
# Dennis & Wolkowicz (1993, `_research/DW_model.pdf`): "atualização fraca de
# Greenstadt direta seguida de uma atualização DFP", hereditariamente
# definida positiva mas não necessariamente na classe convexa — descrita no
# artigo (conjugada com escalonamento de Oren-Luenberger só na 1ª iteração)
# como a que teve melhor desempenho numérico entre as testadas. Usamos a
# versão que opera na Hessiana direta `B` (dual da (iv), que opera em `H`),
# pelo mesmo motivo do `:is_bfgs`: dispensa inverter `H[i]`. Fórmula
# (`φ=1-(b/c)` na eq. (3.1), simplificada; verificado à mão que B₊s=y):
#
#   b = y's,  c = s'Bs,  w = y/b − Bs/c
#   B₊ = B − (Bs)(Bs)'/c + yy'/b + b·ww'
#
# Não implementamos o escalonamento na 1ª iteração do artigo: `H[i]` aqui
# começa em zero (não numa identidade escalada como no artigo), e escalar a
# matriz nula não tem efeito — o guard de `c≈0` abaixo já cobre esse caso
# inicial, caindo de volta em só acumular `yy'/b` (igual ao `:bfgs` comum
# no mesmo caso).
#
# `:is_dw_model` — combina os dois: aplica o `γ` de escala do `:is_bfgs`
# (mesmo `γ=c/b`, mesmo teste de intervalo) à parte `[B − (Bs)(Bs)'/c]` do
# `:dw_model`, mantendo `yy'/b` e o termo extra `b·ww'` sem escalar:
#
#   B₊ = (1/γ)·[B − (Bs)(Bs)'/c] + yy'/b + b·ww'
#
# (verificado à mão: B₊s=y para qualquer γ>0, mesma prova de sempre). É uma
# generalização estrita dos outros dois: com o termo `b·ww'` removido, é
# exatamente `:is_bfgs`; com `γ=1`, é exatamente `:dw_model`.
#
# `:psb` — Powell-Symmetric-Broyden (Powell, 1970; ver também Nocedal &
# Wright, eq. 6.22), atualização clássica de posto 2, não específica de
# nenhum dos dois artigos acima:
#
#   r = y − Bs
#   B₊ = B + (rs' + sr')/(s's) − (s'r/(s's)²)·ss'
#
# (verificado à mão: B₊s=y). Ao contrário de `:bfgs`/`:is_bfgs`/`:dw_model`/
# `:is_dw_model`, não preserva definição positiva — mesma categoria de
# `:sr1` nesse sentido.
# ==============================================================================

function _ffjm2_box_update!(H, s, Jnew, Jold, update, tol, gamma_bar)
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
        elseif update === :sr1
            v = y - Hi * s
            vs = dot(v, s)
            if abs(vs) > tol * max(one(vs), norm(v) * ss)
                Hi .+= (v * v') / vs
            end
        elseif update === :is_bfgs
            sy = dot(s, y)
            Bs = Hi * s
            sBs = dot(s, Bs)
            curvature_ok = abs(sy) > tol * max(one(sy), ss * norm(y))
            if abs(sBs) > tol * max(one(sBs), ss * norm(Bs))
                γ = curvature_ok ? sBs / sy : one(sBs)
                inv_γ = (one(γ) / gamma_bar <= γ <= gamma_bar) ? one(γ) / γ : one(γ)
                Hi .= inv_γ .* (Hi .- (Bs * Bs') ./ sBs)
            end
            if curvature_ok
                Hi .+= (y * y') / sy
            end
        elseif update === :dw_model
            sy = dot(s, y)
            Bs = Hi * s
            sBs = dot(s, Bs)
            curvature_ok = abs(sy) > tol * max(one(sy), ss * norm(y))
            correction_ok = abs(sBs) > tol * max(one(sBs), ss * norm(Bs))
            if correction_ok
                Hi .-= (Bs * Bs') ./ sBs
            end
            if curvature_ok
                Hi .+= (y * y') ./ sy
            end
            if curvature_ok && correction_ok
                w = y ./ sy .- Bs ./ sBs
                Hi .+= sy .* (w * w')
            end
        elseif update === :is_dw_model
            sy = dot(s, y)
            Bs = Hi * s
            sBs = dot(s, Bs)
            curvature_ok = abs(sy) > tol * max(one(sy), ss * norm(y))
            correction_ok = abs(sBs) > tol * max(one(sBs), ss * norm(Bs))
            if correction_ok
                γ = curvature_ok ? sBs / sy : one(sBs)
                inv_γ = (one(γ) / gamma_bar <= γ <= gamma_bar) ? one(γ) / γ : one(γ)
                Hi .= inv_γ .* (Hi .- (Bs * Bs') ./ sBs)
            end
            if curvature_ok
                Hi .+= (y * y') ./ sy
            end
            if curvature_ok && correction_ok
                w = y ./ sy .- Bs ./ sBs
                Hi .+= sy .* (w * w')
            end
        else # :psb
            Bs = Hi * s
            resid = y .- Bs
            if ss > tol * max(one(ss), ss)
                s2 = ss^2
                rs = dot(resid, s)
                Hi .+= (resid * s' .+ s * resid') ./ s2 .- (rs / s2^2) .* (s * s')
            end
        end
        # Elimina assimetria de arredondamento acumulada.
        Hi .= (Hi .+ Hi') ./ 2
    end
    return H
end
