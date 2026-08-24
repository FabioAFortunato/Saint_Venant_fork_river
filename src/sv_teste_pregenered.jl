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
using Optim

# `sr1_bfgs_backtracking.jl` já inclui `sv_fork.jl` e traz `sv_box_penalty`/
# `sv_objective_from_residual` — o núcleo da penalidade de caixa compartilhado
# por todo `bfgs_*` deste arquivo e de `ffjm2.jl` (ver docstring de
# `sv_objective_from_residual`).
include("sr1_bfgs_backtracking.jl")

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
# `sum(residual.^2) + sv_box_penalty(...)` a partir de um `x0` diferente de
# `ng_verdadeiro` e checar se o BFGS recupera `ng_verdadeiro`.
# ==============================================================================

"""
    bfgs_puro_penalizado_pregerado(x_otimo, x0; tbeg=0.0, tend=31.0, kwargs...)

Roda `Optim.BFGS` sobre `sum(abs2, residual) + sv_box_penalty(x, lower, upper, penalty_weight)`,
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

    dados_pregerados = sv_fork_dados_pregerados(x_otimo, tbeg, tend)

    raw_residual(x) = sv_fork_assimilation_pregerado(x, tbeg, tend, dados_pregerados, nothing).erro
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
