# ==============================================================================
# Saint-Venant com dados pré-gerados (experimento gêmeo / twin experiment)
#
# `sv_fork_dados_pregerados` roda a simulação direta uma vez com um vetor de
# Manning "verdadeiro" (`ng_verdadeiro`) e grava a série temporal de z na
# saída (x=3256) e no ponto interior (x=751), devolvendo interpoladores
# prontos para uso (mesma ideia de `zfinal_fake`/`zmeio_fake` em `sv_fork.jl`,
# só que como função, sem depender do global `const fake` nem de um arquivo
# `dado_fake.txt` escrito à mão).
#
# `sv_fork_assimilation_pregerado` é a mesma simulação/erro de
# `sv_fork_assimilation` (mesmo esquema numérico, mesmo formato de retorno) —
# só que compara z(3256,t) e z(751,t) contra os interpoladores pré-gerados em
# vez de `zoutlet`/`zhistomedio` (dados reais de `dado_fork.jl`). Serve para
# testar se a calibração (ex. `ffjm2`) recupera `ng_verdadeiro` quando a
# "verdade" é conhecida de antemão.
# ==============================================================================

using LineSearches
using NLopt
using NOMAD
using Optim

# `sr1_bfgs_backtracking.jl` já inclui `sv_fork.jl` e traz `sv_box_penalty`/
# `sv_objective_from_residual` — o núcleo da penalidade de caixa compartilhado
# por todo `bfgs_*` deste arquivo e de `ffjm2.jl` (ver docstring de
# `sv_objective_from_residual`).
include("sr1_bfgs_backtracking.jl")

# `SidPsm` (busca padrão livre de derivada com direções geradas por
# derivadas de simplex, tradução do SID-PSM original em Matlab) vive em
# `tmp/SidPsm.jl` em vez de `src/` — incluído por caminho relativo e
# referenciado sempre como `SidPsm.<nome>` (nunca `using .SidPsm`) porque o
# módulo exporta nomes genéricos (`Problem`, `Parameters`) que colidiriam
# facilmente com outros arquivos deste projeto, todos compartilhando o
# namespace de `Main` via `include`. Guardado porque `ffjm2.jl` (incluído
# logo abaixo) também o inclui.
if !isdefined(@__MODULE__, :SidPsm)
    include(joinpath(@__DIR__, "..", "tmp", "SidPsm.jl"))
end

# `ffjm2.jl` traz `ffjm2`, `bfgs_puro_penalizado` e as versões reais de
# BOBYQA/MADS/SID-PSM (`bobyqa_puro_penalizado`, `mads_puro_penalizado`,
# `sidpsm_puro_penalizado`) usadas por `comparar_solvers_real` abaixo.
if !isdefined(@__MODULE__, :ffjm2)
    include("ffjm2.jl")
end

function sv_fork_dados_pregerados(
    ng_verdadeiro::AbstractVector{T},
    tbeg,
    tend,
    estado0 = nothing,
) where T<:Real
    # variable
    local alfa = 0.99 # termo de difusão artificial -  alfa = 1 é sem difusao
    local ualfa = 1.0 - alfa
    local xmax = 3256.0 # end point/station
    local xmin = -39.0 # initial point/station
    local dt = 1.0 #second(s)
    local dx = (xmax - xmin) / (nx - 1) #space discretization
    local grav = 9.8 #gravitational constant
    local x = 0.0; zb = zeros(T, nx); z = zeros(T, nx); h = zeros(T, nx); av = zeros(T, nx)
    local ancho = zeros(T, nx); a = zeros(T, nx); v = zeros(T, nx)
    local t = 60.0 * 60.0 * 24.0 * tbeg; ya = 0; imprim = 0.0
    local televa = Float64[]; z_saida = Float64[]; z_interior = Float64[]
    local at, avx; anew = zeros(T, nx); zhatx = zeros(T, nx); av2x = 0.0; peri = 0.0
    local avt = 0.0; eneg = 0.0; avnew = zeros(T, nx); vnew = zeros(T, nx); hnew = zeros(T, nx)
    local tmix_aux = 60.0 * 60.0 * 24.0 * max(3.0, tbeg) # We have data after day 3
    local continuar = true
    local tmax = 60.0 * 60.0 * 24.0 * tend
    local n_man = length(ng_verdadeiro)
    local idx_751 = clamp(round(Int, 1 + (751.0 - xmin) / dx), 1, nx)

    for i = 1:nx
        x = xmin + dx * (i - 1)
        zb[i] = zbfork(x)
        ancho[i] = anchofork(x)
    end

    if tbeg == 0.0 || estado0 === nothing
        for i = 1:nx
            z[i] = zfork(xmin + dx * (i - 1))
            h[i] = z[i] - zb[i]
            a[i] = ancho[i] * h[i]
            av[i] = qinlet(t)
            v[i] = av[i] / a[i]
        end
    else
        for campo in (:z, :a, :h, :v, :av)
            if !hasproperty(estado0, campo)
                error("estado0 deve conter o campo .$campo.")
            end
        end

        if length(estado0.z) != nx ||
           length(estado0.a) != nx ||
           length(estado0.h) != nx ||
           length(estado0.v) != nx ||
           length(estado0.av) != nx
            error("Os vetores de estado0 devem ter tamanho nx = $nx.")
        end

        for i = 1:nx
            z[i] = estado0.z[i]
            a[i] = estado0.a[i]
            h[i] = estado0.h[i]
            v[i] = estado0.v[i]
            av[i] = estado0.av[i]
        end
    end

    while t <= tmax

        #    Smoothing
        for i = 2:nx-1
            h[i] = alfa * h[i] + ualfa * (h[i-1] + h[i+1]) / 2.0
            av[i] = alfa * av[i] + ualfa * (av[i-1] + av[i+1]) / 2.0
        end

        for i = 1:nx
            z[i] = h[i] + zb[i]
            a[i] = ancho[i] * h[i]
            v[i] = av[i] / a[i]
        end
        # End of Smoothing

        #   Writing
        imprim = tmix_aux + ya * timprim
        if (t >= imprim) && (t <= tmax)
            ya = ya + 1
            push!(televa, t)
            push!(z_saida, z[nx])
            push!(z_interior, z[idx_751])
        end
        # end Writing

        t = t + dt

        for i = 1:nx
            # Consider the mass conservation equation
            if i > 1 && i < nx
                avx = (av[i+1] - av[i-1]) / (2.0 * dx)
            end
            if i == 1
                avx = (av[i+1] - av[i]) / dx
            end
            if i == nx
                avx = (av[i] - av[i-1]) / dx
            end

            at = -avx
            anew[i] = a[i] + dt * at

            if anew[i] < 0.0
                continuar = false
                break
            end

            # Consider the momentum conservation equation
            if i > 1 && i < nx
                av2x = (av[i+1] * v[i+1] - av[i-1] * v[i-1]) / (2.0 * dx)
                zhatx[i] = (z[i+1] - z[i-1]) / (2.0 * dx)
            end
            if i == 1
                av2x = (av[i+1] * v[i+1] - av[i] * v[i]) / dx
                zhatx[i] = (z[i+1] - z[i]) / dx
            end
            if i == nx
                av2x = (av[i] * v[i] - av[i-1] * v[i-1]) / dx
                zhatx[i] = (z[i] - z[i-1]) / dx
            end

            zhatx[i] = zhatx[i] / (1.0 + zhatx[i]^2)
            peri = ancho[i] + 2.0 * h[i]

            if n_man == 2
                eneg = (1 - (i - 1) / (nx - 1)) * ng_verdadeiro[1] + (i - 1) / (nx - 1) * ng_verdadeiro[2]
            elseif n_man == 1
                eneg = ng_verdadeiro[1]
            elseif n_man == nx
                eneg = ng_verdadeiro[i]
            elseif 2 < n_man < nx
                pos_ng = 1.0 + (i - 1) * (n_man - 1) / (nx - 1)
                j = floor(Int, pos_ng)
                j = clamp(j, 1, n_man - 1)
                frac = pos_ng - j
                eneg = (1.0 - frac) * ng_verdadeiro[j] + frac * ng_verdadeiro[j+1]
            else
                println("Wrong dimension. Please choose ng with size between 2 and nx")
            end

            rh3 = (a[i] / peri)^(4.0 / 3.0)
            avt = -av2x - grav * a[i] * zhatx[i] - eneg^2 * av[i] * sqrt(av[i]^2 + 1E-3) / (rh3 * a[i])
            avnew[i] = av[i] + dt * avt

            if isnan(avt)
                continuar = false
                break
            end

            if avnew[i] == 0.0
                vnew[i] = 0.0
            else
                vnew[i] = avnew[i] / anew[i]
            end
            hnew[i] = anew[i] / ancho[i]
        end # end for (nx)

        if !continuar
            break # sai do while
        end

        avnew[1] = qinlet(t)
        vnew[1] = avnew[1] / anew[1]

        for i = 1:nx
            a[i] = anew[i]
            v[i] = vnew[i]
            h[i] = hnew[i]
            av[i] = avnew[i]
            z[i] = zb[i] + h[i]
        end
    end # end while (t)

    # Ao contrário de `sv_fork_assimilation` (chamada repetidamente dentro de
    # um laço de calibração, que precisa de um sentinela em vez de exceção),
    # esta função roda uma única vez para gerar a "verdade" — se a simulação
    # divergir aqui, os dados pré-gerados resultantes seriam inválidos, então
    # é melhor falhar alto.
    continuar || error(
        "sv_fork_dados_pregerados: simulação divergiu antes de tend " *
        "(ng_verdadeiro=$ng_verdadeiro, tbeg=$tbeg, tend=$tend).",
    )

    # `continuar == true` só garante que os testes de divergência internos
    # (anew<0, isnan(avt)) não dispararam — não garante que z[nx]/z[idx_751]
    # em si sejam finitos. Checa explicitamente antes de interpolar: um
    # interpolador construído sobre NaN/Inf propagaria isso silenciosamente
    # para todo `sv_fork_assimilation_pregerado` que o usar depois.
    (all(isfinite, z_saida) && all(isfinite, z_interior)) || error(
        "sv_fork_dados_pregerados: dados gerados com ng_verdadeiro=$ng_verdadeiro " *
        "contêm NaN/Inf (tbeg=$tbeg, tend=$tend).",
    )

    z_saida_interp = extrapolate(interpolate((televa,), z_saida, Gridded(Linear())), Line())
    z_interior_interp = extrapolate(interpolate((televa,), z_interior, Gridded(Linear())), Line())

    return (
        t = televa,
        z_saida = z_saida,
        z_interior = z_interior,
        z_saida_interp = z_saida_interp,
        z_interior_interp = z_interior_interp,
    )
end

function sv_fork_assimilation_pregerado(
    ng::AbstractVector{T},
    tbeg,
    tend,
    dados_pregerados,
    estado0 = nothing,
) where T<:Real
    # variable
    local alfa = 0.99 # termo de difusão artificial -  alfa = 1 é sem difusao
    local ualfa = 1.0 - alfa
    local xmax = 3256.0 # end point/station
    local xmin = -39.0 # initial point/station
    local dt = 1.0 #second(s)
    local dx = (xmax - xmin) / (nx - 1) #space discretization
    local grav = 9.8 #gravitational constant
    local x = 0.0; zb = zeros(T, nx); z = zeros(T, nx); h = zeros(T, nx); av = zeros(T, nx)
    local ancho = zeros(T, nx); a = zeros(T, nx); v = zeros(T, nx)
    local t = 60.0 * 60.0 * 24.0 * tbeg; ya = 0; imprim = 0.0; televa = []; zou = []; zinterior = []
    local at, avx; anew = zeros(T, nx); zhatx = zeros(T, nx); av2x = 0.0; peri = 0.0
    local avt = 0.0; eneg = 0.0; avnew = zeros(T, nx); vnew = zeros(T, nx); hnew = zeros(T, nx)
    local tmix_aux = 60.0 * 60.0 * 24.0 * max(3.0, tbeg) # We have data after day 3
    local continuar = true
    local tmax = 60.0 * 60.0 * 24.0 * tend
    local n_man = length(ng)
    local idx_751 = clamp(round(Int, 1 + (751.0 - xmin) / dx), 1, nx)

    for i = 1:nx
        x = xmin + dx * (i - 1)
        zb[i] = zbfork(x)
        ancho[i] = anchofork(x)
    end

    if tbeg == 0.0 || estado0 === nothing
        for i = 1:nx
            z[i] = zfork(xmin + dx * (i - 1))
            h[i] = z[i] - zb[i]
            a[i] = ancho[i] * h[i]
            av[i] = qinlet(t)
            v[i] = av[i] / a[i]
        end
    else
        for campo in (:z, :a, :h, :v, :av)
            if !hasproperty(estado0, campo)
                error("estado0 deve conter o campo .$campo.")
            end
        end

        if length(estado0.z) != nx ||
           length(estado0.a) != nx ||
           length(estado0.h) != nx ||
           length(estado0.v) != nx ||
           length(estado0.av) != nx
            error("Os vetores de estado0 devem ter tamanho nx = $nx.")
        end

        for i = 1:nx
            z[i] = estado0.z[i]
            a[i] = estado0.a[i]
            h[i] = estado0.h[i]
            v[i] = estado0.v[i]
            av[i] = estado0.av[i]
        end
    end

    while t <= tmax

        #    Smoothing
        for i = 2:nx-1
            h[i] = alfa * h[i] + ualfa * (h[i-1] + h[i+1]) / 2.0
            av[i] = alfa * av[i] + ualfa * (av[i-1] + av[i+1]) / 2.0
        end

        for i = 1:nx
            z[i] = h[i] + zb[i]
            a[i] = ancho[i] * h[i]
            v[i] = av[i] / a[i]
        end
        # End of Smoothing

        #   Writing
        imprim = tmix_aux + ya * timprim
        if (t >= imprim) && (t <= tmax)
            ya = ya + 1
            push!(televa, t)
            z_3256 = dados_pregerados.z_saida_interp(t)
            z_751 = dados_pregerados.z_interior_interp(t)
            push!(zou, z[nx] - z_3256)
            push!(zinterior, z[idx_751] - z_751)
        end
        # end Writing

        t = t + dt

        for i = 1:nx
            # Consider the mass conservation equation
            if i > 1 && i < nx
                avx = (av[i+1] - av[i-1]) / (2.0 * dx)
            end
            if i == 1
                avx = (av[i+1] - av[i]) / dx
            end
            if i == nx
                avx = (av[i] - av[i-1]) / dx
            end

            at = -avx
            anew[i] = a[i] + dt * at

            if anew[i] < 0.0
                RMSD = 10e26
                continuar = false
                break
            end

            # Consider the momentum conservation equation
            if i > 1 && i < nx
                av2x = (av[i+1] * v[i+1] - av[i-1] * v[i-1]) / (2.0 * dx)
                zhatx[i] = (z[i+1] - z[i-1]) / (2.0 * dx)
            end
            if i == 1
                av2x = (av[i+1] * v[i+1] - av[i] * v[i]) / dx
                zhatx[i] = (z[i+1] - z[i]) / dx
            end
            if i == nx
                av2x = (av[i] * v[i] - av[i-1] * v[i-1]) / dx
                zhatx[i] = (z[i] - z[i-1]) / dx
            end

            zhatx[i] = zhatx[i] / (1.0 + zhatx[i]^2)
            peri = ancho[i] + 2.0 * h[i]

            if n_man == 2
                eneg = (1 - (i - 1) / (nx - 1)) * ng[1] + (i - 1) / (nx - 1) * ng[2]
            elseif n_man == 1
                eneg = ng[1]
            elseif n_man == nx
                eneg = ng[i]
            elseif 2 < n_man < nx
                pos_ng = 1.0 + (i - 1) * (n_man - 1) / (nx - 1)
                j = floor(Int, pos_ng)
                j = clamp(j, 1, n_man - 1)
                frac = pos_ng - j
                eneg = (1.0 - frac) * ng[j] + frac * ng[j+1]
            else
                println("Wrong dimension. Please choose ng with size between 2 and nx")
            end

            rh3 = (a[i] / peri)^(4.0 / 3.0)
            avt = -av2x - grav * a[i] * zhatx[i] - eneg^2 * av[i] * sqrt(av[i]^2 + 1E-3) / (rh3 * a[i])
            avnew[i] = av[i] + dt * avt

            if isnan(avt)
                RMSD = 10e26
                continuar = false
                break
            end

            if avnew[i] == 0.0
                vnew[i] = 0.0
            else
                vnew[i] = avnew[i] / anew[i]
            end
            hnew[i] = anew[i] / ancho[i]
        end # end for (nx)

        if !continuar
            break # sai do while
        end

        avnew[1] = qinlet(t)
        vnew[1] = avnew[1] / anew[1]

        for i = 1:nx
            a[i] = anew[i]
            v[i] = vnew[i]
            h[i] = hnew[i]
            av[i] = avnew[i]
            z[i] = zb[i] + h[i]
        end
    end # end while (t)

    if !continuar
        vec_aux = fill(10e26, 2 * nt)
        return (
            erro = vec_aux,
            z = copy(z),
            a = copy(a),
            h = copy(h),
            v = copy(v),
            av = copy(av),
            t = t,
            ok = false,
        )
    end

    z_error = [zou; zinterior]

    if isnan(norm(z_error, 1))
        z_error = 10e26 * ones(2 * nt)
    end
    return (
        erro = z_error,
        z = copy(z),
        a = copy(a),
        h = copy(h),
        v = copy(v),
        av = copy(av),
        t = t,
        ok = true,
    )
end

# ==============================================================================
# BFGS puro + penalidade de caixa sobre a soma de quadrados, igual a
# `bfgs_puro_penalizado` em `ffjm2.jl`, só que sobre o resíduo pré-gerado
# (`sv_fork_assimilation_pregerado`) em vez do resíduo real
# (`sv_fork_assimilation`). Serve para o experimento gêmeo: minimizar
# `0.5 * (sum(residual.^2) + sv_box_penalty(...))` a partir de um `x0`
# diferente de `ng_verdadeiro` e checar se o BFGS recupera `ng_verdadeiro`. O
# fator 0.5 iguala esse objetivo ao de `ffjm2` (f(θ) = ½‖F(θ)‖²), que não o
# tem, para que a coluna `f` dos dois fique diretamente comparável.
# ==============================================================================

"""
    bfgs_puro_penalizado_pregerado(x_otimo, x0; tbeg=0.0, tend=31.0, kwargs...)

Roda `Optim.BFGS` sobre `0.5 * (sum(abs2, residual) + sv_box_penalty(x, lower, upper, penalty_weight))`,
onde `residual = sv_fork_assimilation_pregerado(x, tbeg, tend, dados_pregerados, nothing).erro`
e `dados_pregerados = sv_fork_dados_pregerados(x_otimo, tbeg, tend)` — ou seja,
`x_otimo` é o `ng` "verdadeiro" que gera os dados pré-gerados usados como
referência do experimento gêmeo (em vez de já receber `dados_pregerados`
pronto). `sv_fork_dados_pregerados` já verifica que `x_otimo` não gera
NaN/Inf nem faz a simulação divergir antes de `tend`, lançando erro nesses
casos.

Mesma estrutura de `bfgs_puro_penalizado` (`ffjm2.jl`): gradiente por
`ForwardDiff.gradient!` sobre o objetivo penalizado, `Optim.BFGS` com
`LineSearches.BackTracking(order=2)` por padrão, e o mesmo formato de retorno
(`minimizer`, `minimum`, `residual`, `gradient`, `iterations`, `converged`,
`status`, ...), para ficar comparável com `ffjm2`/`bfgs_puro_penalizado`.
"""
function bfgs_puro_penalizado_pregerado(
    x_otimo::AbstractVector,
    x0::AbstractVector;
    tbeg::Real = 0.0,
    tend::Real = 31.0,
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
    dados_pregerados = sv_fork_dados_pregerados(x_otimo, tbeg, tend)
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

    raw_residual(x) = sv_fork_assimilation_pregerado(x, tbeg, tend, dados_pregerados, nothing).erro
    pen_objective(x) = 0.5 * (sum(abs2, raw_residual(x)) + sv_box_penalty(x, lb, ub, penalty_weight))

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
# BOBYQA (NLopt.jl, livre de derivada) sobre a mesma penalidade de caixa do
# experimento gêmeo, igual ao bloco "BOBYQA" de `comparar_bfgs_spg_bobyqa_mads`
# em `ffjm2.jl`, só que sobre o resíduo pré-gerado
# (`sv_fork_assimilation_pregerado`) em vez do resíduo real.
# ==============================================================================

"""
    bobyqa_puro_penalizado_pregerado(x_otimo, x0; tbeg=0.0, tend=31.0, kwargs...)

Roda `NLopt.:LN_BOBYQA` sobre `0.5 * (sum(abs2, residual) + sv_box_penalty(x, lower, upper, penalty_weight))`,
onde `residual = sv_fork_assimilation_pregerado(x, tbeg, tend, dados_pregerados, nothing).erro`
e `dados_pregerados = sv_fork_dados_pregerados(x_otimo, tbeg, tend)` — mesmo
experimento gêmeo de [`bfgs_puro_penalizado_pregerado`](@ref): `x_otimo` é o
`ng` "verdadeiro" usado para gerar os dados de referência.

Mesma configuração do bloco BOBYQA de `comparar_bfgs_spg_bobyqa_mads`
(`ffjm2.jl`): caixa nativa via `lower_bounds`/`upper_bounds`, `initial_step =
rhobeg`, `xtol_abs = rhoend`, `maxeval = f_calls_limit`. BOBYQA não usa
gradiente — `gradient` no retorno é calculado à parte (via
`ForwardDiff.gradient!` sobre o objetivo penalizado) só para permitir
comparação com `bfgs_puro_penalizado_pregerado`, e não conta para
`function_evaluations`/`gradient_evaluations`. O formato de retorno segue o
de `bfgs_puro_penalizado_pregerado` para ficar comparável.
"""
function bobyqa_puro_penalizado_pregerado(
    x_otimo::AbstractVector,
    x0::AbstractVector;
    tbeg::Real = 0.0,
    tend::Real = 31.0,
    f_calls_limit::Integer = 1000,
    rhobeg::Real = 0.005,
    rhoend::Real = 1e-6,
    penalty_weight::Real = 1e6,
    lower::AbstractVector = zeros(length(x0)),
    upper::AbstractVector = fill(0.5, length(x0)),
)
    dados_pregerados = sv_fork_dados_pregerados(x_otimo, tbeg, tend)
    start_time_ns = time_ns()
    x = collect(float.(x0))
    dim = length(x)
    lb = collect(float.(lower))
    ub = collect(float.(upper))
    length(lb) == dim || throw(DimensionMismatch("lower e x0 devem ter o mesmo tamanho"))
    length(ub) == dim || throw(DimensionMismatch("upper e x0 devem ter o mesmo tamanho"))
    all(lb .< ub) || throw(ArgumentError("cada limite inferior deve ser menor que o superior"))
    penalty_weight > 0 || throw(ArgumentError("penalty_weight deve ser positivo"))

    function_evaluations = Ref(0)
    function_evaluation_time_seconds = Ref(0.0)

    raw_residual(x) = sv_fork_assimilation_pregerado(x, tbeg, tend, dados_pregerados, nothing).erro
    pen_objective(x) = 0.5 * (sum(abs2, raw_residual(x)) + sv_box_penalty(x, lb, ub, penalty_weight))

    function objective(x)
        start_ns = time_ns()
        value = pen_objective(x)
        function_evaluations[] += 1
        function_evaluation_time_seconds[] += (time_ns() - start_ns) / 1e9
        return (isfinite(value) && value <= 1000) ? value : oftype(value, Inf)
    end

    optimizer = NLopt.Opt(:LN_BOBYQA, dim)
    optimizer.lower_bounds = lb
    optimizer.upper_bounds = ub
    optimizer.initial_step = fill(Float64(rhobeg), dim)
    optimizer.xtol_abs = fill(Float64(rhoend), dim)
    optimizer.maxeval = Int(f_calls_limit)
    optimizer.min_objective = (x, grad) -> objective(x)

    minimum_value, minimizer, status = NLopt.optimize(optimizer, x)
    minimizer = collect(minimizer)
    execution_time_seconds = (time_ns() - start_time_ns) / 1e9

    final_residual = raw_residual(minimizer)
    config = ForwardDiff.GradientConfig(pen_objective, minimizer, ForwardDiff.Chunk{dim}())
    final_gradient = similar(minimizer, Float64)
    ForwardDiff.gradient!(final_gradient, pen_objective, minimizer, config)

    converged = status in (:SUCCESS, :STOPVAL_REACHED, :FTOL_REACHED, :XTOL_REACHED)
    mean_function_evaluation_time_seconds = function_evaluations[] == 0 ? 0.0 :
        function_evaluation_time_seconds[] / function_evaluations[]

    return (;
        minimizer,
        minimum = minimum_value,
        residual = final_residual,
        gradient = final_gradient,
        hessians = nothing,
        iterations = nothing,
        converged,
        status = Symbol(status),
        execution_time_seconds,
        rejected_directions = 0,
        alphas = nothing,
        solution = (; minimum_value, minimizer, status),
        u = minimizer,
        objective = minimum_value,
        stats = (; minimum_value, minimizer, status),
        retcode = Symbol(status),
        function_evaluations = function_evaluations[],
        gradient_evaluations = 0,
        residual_evaluations = function_evaluations[],
        function_evaluation_time_seconds = mean_function_evaluation_time_seconds,
        gradient_evaluation_time_seconds = 0.0,
        total_function_evaluation_time_seconds = function_evaluation_time_seconds[],
        total_gradient_evaluation_time_seconds = 0.0,
    )
end

# ==============================================================================
# MADS (NOMAD.jl, livre de derivada) sobre a mesma penalidade de caixa do
# experimento gêmeo, igual ao bloco "MADS" de `comparar_bfgs_spg_bobyqa_mads`
# em `ffjm2.jl`, só que sobre o resíduo pré-gerado
# (`sv_fork_assimilation_pregerado`) em vez do resíduo real.
# ==============================================================================

"""
    mads_puro_penalizado_pregerado(x_otimo, x0; tbeg=0.0, tend=31.0, kwargs...)

Roda `NOMAD.solve` (MADS) sobre `sum(abs2, residual) + sv_box_penalty(x, lower, upper, penalty_weight)`,
onde `residual = sv_fork_assimilation_pregerado(x, tbeg, tend, dados_pregerados, nothing).erro`
e `dados_pregerados = sv_fork_dados_pregerados(x_otimo, tbeg, tend)` — mesmo
experimento gêmeo de [`bfgs_puro_penalizado_pregerado`](@ref): `x_otimo` é o
`ng` "verdadeiro" usado para gerar os dados de referência.

Mesma configuração do bloco MADS de `comparar_bfgs_spg_bobyqa_mads`
(`ffjm2.jl`): caixa nativa via `lower_bound`/`upper_bound`, `initial_mesh_size
= rhobeg`, `min_mesh_size = rhoend`, `max_bb_eval = f_calls_limit`. Como
`NOMAD.solve` pode devolver `x_sol === nothing` quando nenhum ponto factível é
aceito, o melhor ponto avaliado (`best_point_mads`) é usado como reserva.
MADS não usa gradiente — `gradient` no retorno é calculado à parte (via
`ForwardDiff.gradient!` sobre o objetivo penalizado) só para permitir
comparação com `bfgs_puro_penalizado_pregerado`, e não conta para
`function_evaluations`/`gradient_evaluations`. O formato de retorno segue o
de `bfgs_puro_penalizado_pregerado` para ficar comparável.
"""
function mads_puro_penalizado_pregerado(
    x_otimo::AbstractVector,
    x0::AbstractVector;
    tbeg::Real = 0.0,
    tend::Real = 31.0,
    f_calls_limit::Integer = 1000,
    rhobeg::Real = 0.005,
    rhoend::Real = 1e-6,
    penalty_weight::Real = 1e6,
    lower::AbstractVector = zeros(length(x0)),
    upper::AbstractVector = fill(0.5, length(x0)),
)
    dados_pregerados = sv_fork_dados_pregerados(x_otimo, tbeg, tend)
    start_time_ns = time_ns()
    x = collect(float.(x0))
    dim = length(x)
    lb = collect(float.(lower))
    ub = collect(float.(upper))
    length(lb) == dim || throw(DimensionMismatch("lower e x0 devem ter o mesmo tamanho"))
    length(ub) == dim || throw(DimensionMismatch("upper e x0 devem ter o mesmo tamanho"))
    all(lb .< ub) || throw(ArgumentError("cada limite inferior deve ser menor que o superior"))
    penalty_weight > 0 || throw(ArgumentError("penalty_weight deve ser positivo"))

    function_evaluations = Ref(0)
    function_evaluation_time_seconds = Ref(0.0)
    best_value_mads = Ref(Inf)
    best_point_mads = copy(x)

    raw_residual(x) = sv_fork_assimilation_pregerado(x, tbeg, tend, dados_pregerados, nothing).erro
    pen_objective(x) = sum(abs2, raw_residual(x)) + sv_box_penalty(x, lb, ub, penalty_weight)

    function objective(x)
        start_ns = time_ns()
        value = pen_objective(x)
        function_evaluations[] += 1
        function_evaluation_time_seconds[] += (time_ns() - start_ns) / 1e9
        return (isfinite(value) && value <= 1000) ? value : oftype(value, Inf)
    end

    objective_mads = function (x)
        value = Float64(objective(x))
        if value < best_value_mads[]
            best_value_mads[] = value
            best_point_mads .= x
        end
        return true, true, [value]
    end

    options_mads = NOMAD.NomadOptions(display_degree = 0, max_bb_eval = Int(f_calls_limit))
    problem_mads = NOMAD.NomadProblem(
        dim, 1, ["OBJ"], objective_mads,
        input_types = fill("R", dim),
        lower_bound = lb, upper_bound = ub,
        min_mesh_size = fill(Float64(rhoend), dim),
        initial_mesh_size = fill(Float64(rhobeg), dim),
        options = options_mads,
    )

    result_mads = NOMAD.solve(problem_mads, x)
    execution_time_seconds = (time_ns() - start_time_ns) / 1e9

    minimizer = result_mads.x_sol === nothing ? copy(best_point_mads) : collect(result_mads.x_sol)
    minimum_value = Float64(objective(minimizer))
    status = result_mads.status
    factivel = result_mads.feasible

    final_residual = raw_residual(minimizer)
    config = ForwardDiff.GradientConfig(pen_objective, minimizer, ForwardDiff.Chunk{dim}())
    final_gradient = similar(minimizer, Float64)
    ForwardDiff.gradient!(final_gradient, pen_objective, minimizer, config)

    mean_function_evaluation_time_seconds = function_evaluations[] == 0 ? 0.0 :
        function_evaluation_time_seconds[] / function_evaluations[]

    return (;
        minimizer,
        minimum = minimum_value,
        residual = final_residual,
        gradient = final_gradient,
        hessians = nothing,
        iterations = nothing,
        converged = factivel,
        status = Symbol(status),
        execution_time_seconds,
        rejected_directions = 0,
        alphas = nothing,
        solution = result_mads,
        u = minimizer,
        objective = minimum_value,
        stats = result_mads,
        retcode = Symbol(status),
        function_evaluations = function_evaluations[],
        gradient_evaluations = 0,
        residual_evaluations = function_evaluations[],
        function_evaluation_time_seconds = mean_function_evaluation_time_seconds,
        gradient_evaluation_time_seconds = 0.0,
        total_function_evaluation_time_seconds = function_evaluation_time_seconds[],
        total_gradient_evaluation_time_seconds = 0.0,
    )
end

# ==============================================================================
# SID-PSM (busca padrão livre de derivada com direções geradas por derivadas
# de simplex, `tmp/SidPsm.jl`) sobre a mesma penalidade de caixa do
# experimento gêmeo, no mesmo espírito do bloco BOBYQA/MADS de
# `comparar_bfgs_spg_bobyqa_mads` em `ffjm2.jl`, só que sobre o resíduo
# pré-gerado (`sv_fork_assimilation_pregerado`) em vez do resíduo real.
# ==============================================================================

"""
    sidpsm_puro_penalizado_pregerado(x_otimo, x0; tbeg=0.0, tend=31.0, kwargs...)

Roda `SidPsm.minimize!` sobre `sum(abs2, residual) + sv_box_penalty(x, lower, upper, penalty_weight)`,
onde `residual = sv_fork_assimilation_pregerado(x, tbeg, tend, dados_pregerados, nothing).erro`
e `dados_pregerados = sv_fork_dados_pregerados(x_otimo, tbeg, tend)` — mesmo
experimento gêmeo de [`bfgs_puro_penalizado_pregerado`](@ref): `x_otimo` é o
`ng` "verdadeiro" usado para gerar os dados de referência.

A caixa (`lower`/`upper`) é passada nativamente via `SidPsm.LinearDomain`
(`SidPsm.Problem(x0, 0, 0, lower, upper; func_f=objective)`, problema sem
restrições não lineares — `m = p = 0`); pontos fora da caixa nem chegam a ser
avaliados. `f_calls_limit` mapeia para `alg.params.fevals_max`
(`alg.params.stop_fevals`). O SID-PSM reescala internamente variáveis com
caixa finita para `[0, 10]` (`alg.scale_x`/`alg.scaling_mask`) — o
minimizador devolvido já é desfeito dessa escala.

SID-PSM não usa gradiente — `gradient` no retorno é calculado à parte (via
`ForwardDiff.gradient!` sobre o objetivo penalizado) só para permitir
comparação com `bfgs_puro_penalizado_pregerado`, e não conta para
`function_evaluations`/`gradient_evaluations`. O formato de retorno segue o
de `bfgs_puro_penalizado_pregerado` para ficar comparável.
"""
function sidpsm_puro_penalizado_pregerado(
    x_otimo::AbstractVector,
    x0::AbstractVector;
    tbeg::Real = 0.0,
    tend::Real = 31.0,
    f_calls_limit::Integer = 1000,
    penalty_weight::Real = 1e6,
    lower::AbstractVector = zeros(length(x0)),
    upper::AbstractVector = fill(0.5, length(x0)),
)
    dados_pregerados = sv_fork_dados_pregerados(x_otimo, tbeg, tend)
    start_time_ns = time_ns()
    x = collect(float.(x0))
    dim = length(x)
    lb = collect(float.(lower))
    ub = collect(float.(upper))
    length(lb) == dim || throw(DimensionMismatch("lower e x0 devem ter o mesmo tamanho"))
    length(ub) == dim || throw(DimensionMismatch("upper e x0 devem ter o mesmo tamanho"))
    all(lb .< ub) || throw(ArgumentError("cada limite inferior deve ser menor que o superior"))
    all(lb .<= x .<= ub) || throw(ArgumentError("x0 deve estar dentro da caixa [lower, upper]"))
    penalty_weight > 0 || throw(ArgumentError("penalty_weight deve ser positivo"))

    function_evaluations = Ref(0)
    function_evaluation_time_seconds = Ref(0.0)

    raw_residual(x) = sv_fork_assimilation_pregerado(x, tbeg, tend, dados_pregerados, nothing).erro
    pen_objective(x) = sum(abs2, raw_residual(x)) + sv_box_penalty(x, lb, ub, penalty_weight)

    function objective(x)
        start_ns = time_ns()
        value = pen_objective(x)
        function_evaluations[] += 1
        function_evaluation_time_seconds[] += (time_ns() - start_ns) / 1e9
        return (isfinite(value) && value <= 1000) ? value : oftype(value, Inf)
    end

    problem = SidPsm.Problem(x, 0, 0, lb, ub; func_f = objective)
    alg = SidPsm.SidPsmAlgorithm(problem)
    alg.params.stop_fevals = true
    alg.params.fevals_max = Int(f_calls_limit)

    SidPsm.minimize!(alg, problem)
    execution_time_seconds = (time_ns() - start_time_ns) / 1e9

    # x0 rejeitado por não-finitude (isfinite(value) ? value : Inf em
    # `objective`, ver `SidPsm.initialization!`) deixa `alg.x_current` vazio
    # — cai de volta em `x` (o único ponto disponível) em vez de indexar um
    # vetor vazio.
    minimizer = if isempty(alg.x_current)
        copy(x)
    else
        m = copy(alg.x_current)
        if alg.scale_x
            mask = alg.scaling_mask
            m[mask] = (m[mask] ./ 10) .* (ub[mask] .- lb[mask]) .+ lb[mask]
        end
        m
    end
    minimum_value = alg.f_obj_current

    final_residual = raw_residual(minimizer)
    config = ForwardDiff.GradientConfig(pen_objective, minimizer, ForwardDiff.Chunk{dim}())
    final_gradient = similar(minimizer, Float64)
    ForwardDiff.gradient!(final_gradient, pen_objective, minimizer, config)

    converged = alg.alfa < alg.params.tol_alfa
    status = converged ? :alfa_tolerance_reached : :fevals_limit_reached
    mean_function_evaluation_time_seconds = function_evaluations[] == 0 ? 0.0 :
        function_evaluation_time_seconds[] / function_evaluations[]

    return (;
        minimizer,
        minimum = minimum_value,
        residual = final_residual,
        gradient = final_gradient,
        hessians = nothing,
        iterations = alg.iter,
        converged,
        status,
        execution_time_seconds,
        rejected_directions = alg.iter_uns,
        alphas = nothing,
        solution = alg,
        u = minimizer,
        objective = minimum_value,
        stats = alg,
        retcode = status,
        function_evaluations = function_evaluations[],
        gradient_evaluations = 0,
        residual_evaluations = function_evaluations[],
        function_evaluation_time_seconds = mean_function_evaluation_time_seconds,
        gradient_evaluation_time_seconds = 0.0,
        total_function_evaluation_time_seconds = function_evaluation_time_seconds[],
        total_gradient_evaluation_time_seconds = 0.0,
    )
end

# ==============================================================================
# Comparação de BFGS, BOBYQA e ffjm2 num único chute inicial —
# experimento gêmeo (`*_pregerado`) ou dados reais (`ffjm2.jl`). As métricas
# (RMSD, gradient_norm) são recalculadas de forma uniforme a partir do
# resíduo bruto (sem penalidade) no minimizador de cada método, para
# comparação direta independente de cada solver usar penalidade de caixa
# internamente ou não — mesma ideia de `record!`/`raw_metrics` em
# `comparar_bfgs_spg_bobyqa_mads`/`comparar_ffjm2_bfgs` (`ffjm2.jl`).
#
# ffjm2 minimiza `raw_residual` diretamente (sem `sv_box_penalty`), igual ao
# uso de `ffjm2` em `comparar_ffjm2_bfgs` — os outros dois solvers usam a
# penalidade de caixa nativamente ou via `sv_box_penalty` (ver docstring de
# cada `*_puro_penalizado*`). Desde que `*_puro_penalizado_pregerado` passou a
# escalar seu objetivo por 0.5 (mesma convenção f(θ) = ½‖F(θ)‖² do ffjm2), a
# coluna `f` também fica diretamente comparável entre os três, não só o RMSD —
# desde que a penalidade de caixa esteja inativa no minimizador encontrado.
# ==============================================================================

function _comparar_solvers(
    raw_residual::Function,
    dim::Integer,
    output,
    solver_runs;
    points_output = nothing,
)
    csv_field(value) = begin
        text = value isa AbstractVector ? repr(collect(value)) : string(value)
        occursin(r"[,\"\n\r]", text) ? "\"$(replace(text, '\"' => "\"\""))\"" : text
    end
    header = (
        "method", "dimension", "RMSD", "gradient_norm", "execution_time_seconds",
        "function_evaluations", "gradient_evaluations", "converged", "status", "f_x", "minimizer",
    )
    points_header = ("method", "point_index", "x")

    function metrics(x)
        residual = collect(raw_residual(x))
        sse = sum(abs2, residual)
        rmsd = sqrt(sse / length(residual))
        raw_objective(z) = sum(abs2, raw_residual(z))
        config = ForwardDiff.GradientConfig(raw_objective, x, ForwardDiff.Chunk{dim}())
        gradient = ForwardDiff.gradient(raw_objective, x, config)
        return (; rmsd, gradient_norm = norm(gradient))
    end

    mkpath(dirname(output))
    points_output === nothing || mkpath(dirname(points_output))
    rows = NamedTuple[]
    open(output, "w") do io
        write(io, join(header, ','), '\n')
        points_io = points_output === nothing ? nothing : open(points_output, "w")
        try
            points_io === nothing || write(points_io, join(points_header, ','), '\n')
            for (method, run) in solver_runs
                println("Executando $method")
                r = run()
                m = metrics(r.minimizer)
                row = (;
                    method, dimension = dim, RMSD = m.rmsd, gradient_norm = m.gradient_norm,
                    execution_time_seconds = r.execution_time_seconds,
                    function_evaluations = r.function_evaluations,
                    gradient_evaluations = r.gradient_evaluations,
                    converged = r.converged, status = string(r.status),
                    f_x = r.minimum, minimizer = copy(r.minimizer),
                )
                push!(rows, row)
                write(io, join(csv_field.(values(row)), ','), '\n')
                flush(io)
                println(
                    "  minimizer=$(row.minimizer) RMSD=$(row.RMSD) fevals=$(row.function_evaluations) ",
                    "tempo=$(round(row.execution_time_seconds, digits=1))s status=$(row.status)",
                )
                if points_io !== nothing
                    accepted_points = get(r, :accepted_points, nothing)
                    if accepted_points !== nothing
                        for (i, xi) in enumerate(accepted_points)
                            write(points_io, join(csv_field.((method, i, collect(xi))), ','), '\n')
                        end
                        flush(points_io)
                    end
                end
            end
        finally
            points_io === nothing || close(points_io)
        end
    end

    println("Comparação salva em: $output")
    points_output === nothing || println("Pontos aceitos salvos em: $points_output")
    return (; rows, output, points_output)
end

"""
    comparar_solvers_pregerado(x_otimo, x0; kwargs...)

Compara BFGS, BOBYQA (`*_puro_penalizado_pregerado`,
`sv_teste_pregenered.jl`) e `ffjm2` no experimento gêmeo definido por
`x_otimo` (mesma convenção de [`bfgs_puro_penalizado_pregerado`](@ref):
`x_otimo` gera os dados de referência via `sv_fork_dados_pregerados`, `x0` é
o chute inicial da otimização). Cada `*_puro_penalizado_pregerado` regenera
os dados de referência (`sv_fork_dados_pregerados`) independentemente — uma
simulação "verdade" extra por solver, negligenciável frente ao custo dos
`f_calls_limit` avaliações de cada otimização.

`f_calls_limit`/`maxiter` controlam o orçamento de avaliações de BFGS e
BOBYQA; `ffjm2_maxiter` controla o número de iterações
externas de `ffjm2` (cada uma custando ao menos 1 avaliação de resíduo, mais
em caso de rejeição por μ). O resultado é salvo em `output` (CSV) e também
devolvido em `rows`.
"""
function comparar_solvers_pregerado(
    x_otimo::AbstractVector,
    x0::AbstractVector;
    tbeg::Real = 0.0,
    tend::Real = 31.0,
    output = normpath(joinpath(
        @__DIR__, "..", "results", "comparacao_solvers_pregerado_dim$(length(x0)).csv",
    )),
    points_output = nothing,
    penalty_weight::Real = 1e6,
    lower::AbstractVector = zeros(length(x0)),
    upper::AbstractVector = fill(0.5, length(x0)),
    maxiter::Integer = 500,
    f_calls_limit::Integer = 1000,
    g_calls_limit::Integer = 500,
    g_tol::Real = 1e-3,
    rhobeg::Real = 0.01,
    rhoend::Real = 1e-6,
    ffjm2_maxiter::Integer = 500,
    ffjm2_options = (;),
    show_trace::Bool = true,
)
    dim = length(x0)
    dados_pregerados = sv_fork_dados_pregerados(x_otimo, tbeg, tend)
    raw_residual(x) = sv_fork_assimilation_pregerado(x, tbeg, tend, dados_pregerados, nothing).erro

    solver_runs = (
        ("BFGS", function ()
            r = bfgs_puro_penalizado_pregerado(
                x_otimo, x0; tbeg, tend, maxiter, f_calls_limit, g_calls_limit, g_tol,
                penalty_weight, lower, upper, show_trace,
            )
            accepted_points = [
                state.metadata["x"]
                for state in Optim.trace(r.solution) if haskey(state.metadata, "x")
            ]
            return (; r.minimizer, r.minimum, r.execution_time_seconds,
                    r.function_evaluations, r.gradient_evaluations, r.converged, r.status,
                    accepted_points)
        end),
         ("BOBYQA", () -> bobyqa_puro_penalizado_pregerado(
             x_otimo, x0; tbeg, tend, f_calls_limit, rhobeg, rhoend, penalty_weight, lower, upper,
         )),
        ("ffjm2", function ()
            external_evaluations = Ref(0)
            counted_residual(x) = (external_evaluations[] += 1; raw_residual(x))
            accepted_points = Vector{Vector{Float64}}()
            track_accepted(state) = (push!(accepted_points, copy(state.x)); false)
            r = ffjm2(
                counted_residual, x0; maxiter = ffjm2_maxiter, g_tol, show_trace,
                callback = track_accepted, ffjm2_options...,
            )
            return (; r.minimizer, r.minimum, r.execution_time_seconds,
                    r.function_evaluations, r.gradient_evaluations, r.converged, r.status,
                    accepted_points)
        end),
    )

    return _comparar_solvers(raw_residual, dim, output, solver_runs; points_output)
end

"""
    comparar_solvers_real(x0; kwargs...)

Compara BFGS (`bfgs_puro_penalizado`), BOBYQA (`bobyqa_puro_penalizado`)
e `ffjm2` sobre dados reais
(`sv_fork_assimilation`, `ffjm2.jl`) — mesma ideia de
[`comparar_solvers_pregerado`](@ref), mas sem `x_otimo` (não é experimento
gêmeo). `bfgs_puro_penalizado` ignora `tbeg`/`tend` (fixos em `0.0`/`31.0`
dentro da própria função); os demais solvers usam os valores passados aqui.

Além do CSV principal (`output`), salva em `points_output` todos os pontos
"aceitos" de cada método, um por linha (`method,point_index,x`): para BFGS,
o `x` de cada iteração do `Optim.trace`; para BOBYQA, os pontos avaliados
que bateram um novo recorde (BOBYQA não expõe aceitação/rejeição interna —
ver docstring de [`bobyqa_puro_penalizado`](@ref)); para `ffjm2`, o `x` no
início de cada iteração externa (via `callback`), que é sempre o último
passo de fato aceito pelo teste de razão ρ.
"""
function comparar_solvers_real(
    x0::AbstractVector;
    tbeg::Real = 0.0,
    tend::Real = 31.0,
    output = normpath(joinpath(
        @__DIR__, "..", "results", "comparacao_solvers_real_dim$(length(x0)).csv",
    )),
    points_output = normpath(joinpath(
        @__DIR__, "..", "results", "comparacao_solvers_real_dim$(length(x0))_pontos.csv",
    )),
    penalty_weight::Real = 1e6,
    lower::AbstractVector = zeros(length(x0)),
    upper::AbstractVector = fill(0.5, length(x0)),
    maxiter::Integer = 500,
    f_calls_limit::Integer = 1000,
    g_calls_limit::Integer = 500,
    g_tol::Real = 1e-3,
    rhobeg::Real = 0.01,
    rhoend::Real = 1e-6,
    ffjm2_maxiter::Integer = 500,
    ffjm2_options = (;),
    show_trace::Bool = true,
)
    dim = length(x0)
    raw_residual(x) = sv_fork_assimilation(x, tbeg, tend, nothing).erro

    solver_runs = (
        ("BFGS", () -> bfgs_puro_penalizado(
            x0; maxiter, f_calls_limit, g_calls_limit, g_tol,
            penalty_weight, lower, upper, show_trace,
        )),
        ("BOBYQA", () -> bobyqa_puro_penalizado(
            x0; tbeg, tend, f_calls_limit, rhobeg, rhoend, penalty_weight, lower, upper,
        )),
        ("ffjm2", function ()
            external_evaluations = Ref(0)
            counted_residual(x) = (external_evaluations[] += 1; raw_residual(x))
            accepted_points = Vector{Vector{Float64}}()
            track_accepted(state) = (push!(accepted_points, copy(state.x)); false)
            r = ffjm2(
                counted_residual, x0; maxiter = ffjm2_maxiter, g_tol, show_trace,
                callback = track_accepted, ffjm2_options...,
            )
            return (; r.minimizer, r.minimum, r.execution_time_seconds,
                    r.function_evaluations, r.gradient_evaluations, r.converged, r.status,
                    accepted_points)
        end),
    )

    return _comparar_solvers(raw_residual, dim, output, solver_runs; points_output)
end

# ==============================================================================
# Heatmap de RMSD "a priori" do experimento gêmeo (dimensão 2, ótimo
# `x_otimo` conhecido) — análogo a `assimilation_rmsd_heatmap`
# (`assimilacao.jl`), que faz a mesma varredura de grade para os dados
# REAIS via `sv_fork_new`. Aqui a varredura usa o resíduo do experimento
# gêmeo (`sv_fork_dados_pregerados`/`sv_fork_assimilation_pregerado`), então
# o RMSD mínimo da grade coincide (a menos da resolução da grade) com
# `x_otimo`, não com um mínimo empírico desconhecido.
#
# Escreve o CSV no MESMO formato lido por `le_assimilation_heatmap_matrix`
# (`assimilacao.jl`, cabeçalho `n2/n1` na célula (1,1), grade de `n1` na
# primeira linha, grade de `n2` na primeira coluna) — ou seja, é "a priori"
# no sentido de que só gera a matriz de RMSD antes de/independente de rodar
# qualquer solver; para plotar, basta apontar `matrix_output` (ou
# `pontos_csv`, se quiser sobrepor caminhos de solvers) para o CSV gerado
# aqui em `plot_assimilacao_heatmap_tend_31_latex`/
# `plot_assimilacao_heatmap_tend_31_latex_com_solvers` (`plots.jl`/
# `scripts/main.jl`) — nenhuma dessas funções de plot precisa mudar.
# ==============================================================================

"""
    assimilation_rmsd_heatmap_pregerado(; x_otimo, tin=0.0, tend=31.0,
                                            grid_points=50, lower=0.05,
                                            upper=0.3, rmsd_max=3.0,
                                            matrix_output=..., show_trace=true)

Varre uma grade `grid_points × grid_points` de `(n1, n2) ∈ [lower,upper]^2`
e calcula, em cada ponto, o RMSD do resíduo do experimento gêmeo contra os
dados sintéticos gerados por `x_otimo` (`sv_fork_dados_pregerados(x_otimo,
tin, tend)`, `x_otimo` deve ter dimensão 2). Valores não finitos ou acima
de `rmsd_max` são saturados em `rmsd_max`, igual a
`assimilation_rmsd_heatmap` (`assimilacao.jl`), cujo formato de CSV este
escreve de volta (compatível com `le_assimilation_heatmap_matrix` e as
funções de plot que dependem dela).

Como os dados são gerados pelo próprio `x_otimo`, o mínimo do RMSD na
grade deve coincidir com `x_otimo` a menos da resolução da grade (`RMSD =
0` exatamente em `x_otimo`, se ele cair sobre um nó) — diferente do
heatmap de dados reais, cujo mínimo empírico é desconhecido a priori.
Devolve `(; n1, n2, RMSD, matrix_output, x_otimo)`.
"""
function assimilation_rmsd_heatmap_pregerado(;
    x_otimo::AbstractVector,
    tin::Real = 0.0,
    tend::Real = 31.0,
    grid_points::Integer = 50,
    lower::Real = 0.05,
    upper::Real = 0.3,
    rmsd_max::Real = 3.0,
    matrix_output = normpath(joinpath(@__DIR__, "..", "results", "assimilacao_heatmap_pregerado_tend_31.csv")),
    show_trace::Bool = true,
)
    length(x_otimo) == 2 ||
        throw(ArgumentError("x_otimo deve ter dimensão 2 (grade 2D de n1 × n2)"))

    dados_pregerados = sv_fork_dados_pregerados(x_otimo, tin, tend)
    raw_residual(x) = sv_fork_assimilation_pregerado(x, tin, tend, dados_pregerados, nothing).erro

    grid = collect(range(lower, upper, length = grid_points))
    Z = Matrix{Float64}(undef, grid_points, grid_points)

    total = grid_points * grid_points
    aval = 0
    for (j, n2) in enumerate(grid)
        for (i, n1) in enumerate(grid)
            aval += 1
            erro = raw_residual([n1, n2])
            rmsd = norm(erro) / sqrt(length(erro))
            if !isfinite(rmsd) || rmsd > rmsd_max
                rmsd = rmsd_max
            end
            Z[j, i] = rmsd

            if show_trace && (aval == 1 || aval % 100 == 0 || aval == total)
                println("Heatmap gêmeo (x_otimo=$x_otimo): avaliação $aval/$total | n1=$n1 | n2=$n2 | RMSD=$rmsd")
            end
        end
    end

    mkpath(dirname(matrix_output))
    open(matrix_output, "w") do io
        write(io, join(vcat("n2/n1", string.(grid)), ','), '\n')
        for j in 1:grid_points
            write(io, join(vcat(string(grid[j]), string.(Z[j, :])), ','), '\n')
        end
    end
    println("Matriz RMSD do experimento gêmeo (x_otimo=$x_otimo) salva em: $matrix_output")

    return (; n1 = grid, n2 = grid, RMSD = Z, matrix_output, x_otimo)
end

# ==============================================================================
# Presets dos 4 cenários de comparação (BFGS/BOBYQA/ffjm2)
# discutidos na sessão: experimento gêmeo em dimensão 2 e 10, e dados reais em
# dimensão 3 e 10. Mesma ideia de `bobyqa_full_dim_problem`
# (`BOBYQA_application.jl`): parâmetros fixos por cima
# de um executor genérico (`comparar_solvers_pregerado`/`comparar_solvers_real`).
#
# Orçamento reduzido (`f_calls_limit=100`, `maxiter=30`, `g_calls_limit=30`,
# `ffjm2_maxiter=30`) escolhido porque uma avaliação de resíduo não-divergente
# a `tend=31` custa ~12s (~2.68M passos de dt=1s) — o orçamento padrão do
# repositório (`f_calls_limit=1000`) levaria horas por solver.
# ==============================================================================

"""
    comparar_solvers_twin_dim2(; kwargs...)

Experimento gêmeo em dimensão 2: `x_otimo = [0.2, 0.15]`, `x0 = fill(0.09, 2)`.
Ver [`comparar_solvers_pregerado`](@ref).
"""
function comparar_solvers_twin_dim2(;
    x_otimo::AbstractVector = [0.2, 0.15],
    x0::AbstractVector = fill(0.25, 2),
    tbeg::Real = 0.0,
    tend::Real = 31.0,
    maxiter::Integer = 500,
    f_calls_limit::Integer = 1000,
    g_calls_limit::Integer = 500,
    ffjm2_maxiter::Integer = 500,
    show_trace::Bool = false,
    output = normpath(joinpath(@__DIR__, "..", "results", "comparacao_solvers_twin_dim2.csv")),
    points_output = normpath(joinpath(@__DIR__, "..", "results", "comparacao_solvers_twin_dim2_pontos.csv")),
    kwargs...,
)
    return comparar_solvers_pregerado(
        x_otimo, x0; tbeg, tend, maxiter, f_calls_limit, g_calls_limit,
        ffjm2_maxiter, show_trace, output, points_output, kwargs...,
    )
end

"""
    comparar_solvers_twin_dim10(; kwargs...)

Experimento gêmeo em dimensão 10: `x_otimo = range(0.12, 0.07, length=10)`
(perfil **decrescente**), `x0 = fill(0.09, 10)`. Perfis crescentes de Manning
ao longo do trecho (rugosidade menor a montante, maior a jusante) divergem
antes de `tend=31` mesmo com variação pequena — testado manualmente com
várias faixas (`0.06`–`0.14`, `0.07`–`0.12`, `0.075`–`0.105`, `0.08`–`0.10`,
todas crescentes, todas divergiram). Perfis decrescentes na mesma ordem de
grandeza são estáveis. Ver [`comparar_solvers_pregerado`](@ref).
"""
function comparar_solvers_twin_dim10(;
    x_otimo::AbstractVector = collect(range(0.12, 0.07, length = 10)),
    x0::AbstractVector = fill(0.25, 10),
    tbeg::Real = 0.0,
    tend::Real = 31.0,
    maxiter::Integer = 500,
    f_calls_limit::Integer = 1000,
    g_calls_limit::Integer = 500,
    ffjm2_maxiter::Integer = 500,
    show_trace::Bool = true,
    output = normpath(joinpath(@__DIR__, "..", "results", "comparacao_solvers_twin_dim10.csv")),
    kwargs...,
)
    return comparar_solvers_pregerado(
        x_otimo, x0; tbeg, tend, maxiter, f_calls_limit, g_calls_limit,
        ffjm2_maxiter, show_trace, output, kwargs...,
    )
end

"""
    comparar_solvers_real_dim2(; kwargs...)

Dados reais em dimensão 2: `x0 = fill(0.09, 2)`. Ver
[`comparar_solvers_real`](@ref) — inclui, além do CSV principal, um segundo
CSV (`points_output`) com todos os pontos aceitos de cada método.
"""
function comparar_solvers_real_dim2(;
    x0::AbstractVector = fill(0.25, 2),
    tbeg::Real = 0.0,
    tend::Real = 31.0,
    maxiter::Integer = 500,
    f_calls_limit::Integer = 1000,
    g_calls_limit::Integer = 500,
    ffjm2_maxiter::Integer = 500,
    show_trace::Bool = true,
    output = normpath(joinpath(@__DIR__, "..", "results", "comparacao_solvers_real_dim2.csv")),
    points_output = normpath(joinpath(@__DIR__, "..", "results", "comparacao_solvers_real_dim2_pontos.csv")),
    kwargs...,
)
    return comparar_solvers_real(
        x0; tbeg, tend, maxiter, f_calls_limit, g_calls_limit,
        ffjm2_maxiter, show_trace, output, points_output, kwargs...,
    )
end

"""
    comparar_solvers_real_dim10(; kwargs...)

Dados reais em dimensão 10: `x0 = fill(0.09, 10)`. Ver [`comparar_solvers_real`](@ref).
"""
function comparar_solvers_real_dim10(;
    x0::AbstractVector = fill(0.25, 10),
    tbeg::Real = 0.0,
    tend::Real = 31.0,
    maxiter::Integer = 500,
    f_calls_limit::Integer = 1000,
    g_calls_limit::Integer = 500,
    ffjm2_maxiter::Integer = 500,
    show_trace::Bool = false,
    output = normpath(joinpath(@__DIR__, "..", "results", "comparacao_solvers_real_dim10.csv")),
    kwargs...,
)
    return comparar_solvers_real(
        x0; tbeg, tend, maxiter, f_calls_limit, g_calls_limit,
        ffjm2_maxiter, show_trace, output, kwargs...,
    )
end

# ==============================================================================
# Acompanhamento do `ffjm2` sozinho (sem BFGS/BOBYQA/MADS) em dados reais,
# iteração a iteração — feito para investigar de perto se a região onde a
# simulação diverge (`P`, na notação do artigo) fica perto do minimizador
# encontrado com dados reais, o que explicaria por que nenhum solver atinge
# `‖∇f‖→0` nesse cenário (ao contrário dos experimentos gêmeos, onde
# `x_otimo` é construído bem dentro da região viável).
# ==============================================================================

"""
    acompanhar_ffjm2_real_dim3(; x0=fill(0.09, 3), tbeg=0.0, tend=31.0,
                                  update=:psb, maxiter=30, show_trace=true,
                                  ffjm2_options=(;))

Roda só o `ffjm2` (nenhum outro solver) no problema de calibração com dados
reais (`sv_fork_assimilation`), dimensão 3. Com `show_trace=true` (padrão),
`ffjm2` imprime, a cada avaliação externa — cada tentativa de `μ`
dentro do laço principal, não só as aceitas —, a iteração `k`, `FO`
(`0.5*||F(x_tentativa)||²` naquele ponto) e `μ` (a regularização usada
naquela tentativa): ver as linhas de `show_trace` em `ffjm2.jl` (dentro do
laço em `μ` de 2.3.2, e na busca linear de `k=0`).
"""
function acompanhar_ffjm2_real_dim3(;
    x0::AbstractVector = fill(0.09, 2),
    tbeg::Real = 0.0,
    tend::Real = 31.0,
    update::Union{Symbol,Tuple{Vararg{Symbol}}} = :psb,
    maxiter::Integer = 100,
    show_trace::Bool = true,
    ffjm2_options = (;),
)
    x = collect(float.(x0))
    raw_residual(z) = sv_fork_assimilation(z, tbeg, tend, nothing).erro
    resultado = ffjm2(raw_residual, x; update, maxiter, show_trace, ffjm2_options...)
    println(
        "\nFinal: minimizer=", resultado.minimizer, " f=", resultado.minimum,
        " ||grad||=", norm(resultado.gradient), " status=", resultado.status,
        " iterations=", resultado.iterations,
    )
    return resultado
end

# ==============================================================================
# Roda só o `ffjm2` (sem BFGS/BOBYQA) nos 4 cenários já usados nas tabelas de
# comparação (twin dim2, twin dim10, real dim3, real dim10) — mesmos
# `x_otimo`/`x0`/orçamento dos presets `comparar_solvers_twin_dim2`/
# `comparar_solvers_twin_dim10`/`comparar_solvers_real_dim10` (o cenário
# "real dim3" não tem mais preset correspondente desde que
# `comparar_solvers_real_dim3` virou `comparar_solvers_real_dim2`; o `x0`
# aqui é reproduzido manualmente), sem recalcular BFGS/BOBYQA (cujos
# resultados já obtidos não são tocados aqui). Feito para re-rodar só a
# linha do `ffjm2` depois de uma correção que só o afeta (ex.: alinhar
# `g_tol` com o do BFGS, ver `comparar_solvers_pregerado`/`comparar_solvers_real`).
# ==============================================================================

"""
    rodar_ffjm2_quatro_cenarios(; tbeg=0.0, tend=31.0, ffjm2_maxiter=500,
                                   g_tol=1e-3, update=:psb, show_trace=true,
                                   ffjm2_options=(;))

Roda só o `ffjm2` nos 4 cenários de parâmetro das tabelas de comparação —
twin dim 2 (`x_otimo=[0.2,0.15]`), twin dim 10 (`x_otimo` decrescente de
`0.12` a `0.07`), dados reais dim 3 e dados reais dim 10 (`x0=fill(0.09,·)`
em todos) — com os mesmos valores usados pelos presets
`comparar_solvers_twin_dim2`/`comparar_solvers_twin_dim10`/
`comparar_solvers_real_dim10` (o cenário "real dim3" não tem mais preset
correspondente, ver comentário acima). Não roda BFGS
nem BOBYQA. Devolve um `Vector` de `NamedTuple`s (`cenario`, `dimension`,
`minimizer`, `rmsd`, `f`, `gradient_norm`, `function_evaluations`,
`gradient_evaluations`, `execution_time_seconds`, `converged`, `status`) e
imprime um resumo por cenário, no mesmo conjunto de métricas das tabelas do
`.tex` (RMSD, `f`, `||grad f||`, Func./Grad. evals, tempo, status).
"""
function rodar_ffjm2_quatro_cenarios(;
    tbeg::Real = 0.0,
    tend::Real = 31.0,
    ffjm2_maxiter::Integer = 500,
    g_tol::Real = 1e-3,
    update::Union{Symbol,Tuple{Vararg{Symbol}}} = :psb,
    show_trace::Bool = true,
    ffjm2_options = (;),
)
    cenarios = (
        (nome = "twin dim2", tipo = :twin, x_otimo = [0.2, 0.15], x0 = fill(0.09, 2)),
        (nome = "twin dim10", tipo = :twin, x_otimo = collect(range(0.12, 0.07, length = 10)), x0 = fill(0.09, 10)),
        (nome = "real dim3", tipo = :real, x_otimo = nothing, x0 = fill(0.09, 3)),
        (nome = "real dim10", tipo = :real, x_otimo = nothing, x0 = fill(0.09, 10)),
    )

    resultados = NamedTuple[]
    for cenario in cenarios
        println("\n=== ffjm2: $(cenario.nome) ===")
        x0 = collect(float.(cenario.x0))

        raw_residual = if cenario.tipo === :real
            (x -> sv_fork_assimilation(x, tbeg, tend, nothing).erro)
        else
            dados_pregerados = sv_fork_dados_pregerados(cenario.x_otimo, tbeg, tend)
            (x -> sv_fork_assimilation_pregerado(x, tbeg, tend, dados_pregerados, nothing).erro)
        end

        r = ffjm2(raw_residual, x0; update, maxiter = ffjm2_maxiter, g_tol, show_trace, ffjm2_options...)

        residual_final = collect(raw_residual(r.minimizer))
        rmsd = norm(residual_final) / sqrt(length(residual_final))
        status = string(r.status)

        resultado = (;
            cenario = cenario.nome,
            dimension = length(x0),
            minimizer = copy(r.minimizer),
            rmsd,
            f = r.minimum,
            gradient_norm = norm(r.gradient),
            function_evaluations = r.function_evaluations,
            gradient_evaluations = r.gradient_evaluations,
            execution_time_seconds = r.execution_time_seconds,
            converged = r.converged,
            status,
        )
        push!(resultados, resultado)

        @printf(
            "ffjm2 (%s): RMSD=%.3e  f=%.4f  ||grad||=%.3e  fevals=%d  gevals=%d  tempo=%.1fs  status=%s\n",
            cenario.nome, rmsd, r.minimum, norm(r.gradient),
            r.function_evaluations, r.gradient_evaluations,
            r.execution_time_seconds, status,
        )
    end

    return resultados
end



function excluir_depois()
    comparar_solvers_real_dim10()
    comparar_solvers_twin_dim10()
    return 10
end
