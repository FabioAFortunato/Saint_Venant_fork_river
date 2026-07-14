
using DelimitedFiles
using PGFPlotsX
using Plots

pgfplotsx()

function busca_perturbacao_segura(
    f,
    x_atual;
    taxa_inicial=0.05,
    taxa_min=0.001,
    tentativas_por_taxa=500,
    lb=0.0,
    ub=1.0,
)
    n = length(x_atual)
    taxa = taxa_inicial
    limite_inferior = lb .* ones(n)
    limite_superior = ub .* ones(n)

    while taxa >= taxa_min
        println("Tentando perturbacoes com taxa = ", taxa)

        for tentativa in 1:tentativas_por_taxa
            perturbacao = (2.0 .* rand(n) .- 1.0) .* taxa
            x_temp = clamp.(x_atual .+ perturbacao, limite_inferior, limite_superior)
            f_val = f(x_temp)

            if isfinite(f_val) && !isnan(f_val) && f_val <= 1000.0
                println("Perturbacao segura encontrada na tentativa ", tentativa, " | f = ", f_val)
                return x_temp
            end
        end

        taxa /= 2.0
    end

    println("Aviso: Nenhuma perturbacao segura encontrada. Retornando ponto do BFGS.")
    return x_atual
end




function arquivo_em_results(arquivo)
    mkpath(RESULTS_DIR)
    return joinpath(RESULTS_DIR, basename(String(arquivo)))
end

nome = arquivo_em_results("teste_bfgs_bobyqa_L_sv_5.txt")

function label_tmax(tmax)
    tmax_float = Float64(tmax)
    return isinteger(tmax_float) ? string(Int(round(tmax_float))) : replace(string(tmax_float), "." => "_")
end

function valor_limitado_para_plot(valor, fmax)
    valor_float = try
        Float64(valor)
    catch
        return fmax
    end

    return isfinite(valor_float) ? min(valor_float, fmax) : fmax
end

function arquivo_csv_busca_exaustiva(output)
    output_string = String(output)
    base, ext = splitext(output_string)
    return isempty(ext) ? "$(output_string).csv" : "$(base).csv"
end

function arquivo_rmsd_busca_exaustiva(output)
    output_string = String(output)
    base, ext = splitext(output_string)
    return isempty(ext) ? "$(output_string)_RMSD" : "$(base)_RMSD$(ext)"
end

function direcao_derivada_normalizada(g; sentido = :descida)
    ng = norm(g)
    if ng == 0
        error("Gradiente nulo: nao ha direcao de derivada para plotar.")
    end

    if sentido in (:descida, :negative_gradient)
        return -g ./ ng
    elseif sentido in (:gradiente, :positive_gradient)
        return g ./ ng
    else
        error("sentido desconhecido: $sentido. Use :descida ou :gradiente.")
    end
end

function plot_busca_exaustiva_derivada(
    f,
    g!,
    x0;
    alpha_min = 0.0,
    alpha_max = 1.0,
    n_points = 100,
    fmax = 100.0,
    sentido = :descida,
    output = arquivo_em_results("busca_exaustiva_derivada.png"),
    output_RMSD = output === nothing ? nothing : arquivo_rmsd_busca_exaustiva(output),
    data_output = output === nothing ? nothing : arquivo_csv_busca_exaustiva(output),
    titulo = "Busca exaustiva na direcao da derivada",
)
    x = copy(float.(x0))
    g = similar(x)
    g!(g, x)
    direcao = direcao_derivada_normalizada(g; sentido = sentido)

    alphas = collect(range(Float64(alpha_min), Float64(alpha_max), length = Int(n_points)))
    valores = Vector{Float64}(undef, length(alphas))
    valores_plot = Vector{Float64}(undef, length(alphas))
    pontos = Matrix{Float64}(undef, length(alphas), length(x))

    for (i, alpha) in pairs(alphas)
        x_alpha = x .+ alpha .* direcao
        pontos[i, :] .= x_alpha
        valor = f(x_alpha)
        valores[i] = try
            Float64(valor)
        catch
            NaN
        end
        valores_plot[i] = valor_limitado_para_plot(valor, Float64(fmax))
    end

    valores_plot_RMSD = sqrt.(max.(valores_plot, 0.0) ./ length(alphas))

    y_min = 0.0
    y_max = Float64(fmax)
    y_min_RMSD = 0.0
    y_max_RMSD = sqrt(Float64(fmax) / length(alphas))
    if y_min == y_max
        y_min -= 1.0
    end
    if y_min_RMSD == y_max_RMSD
        y_min_RMSD -= 1.0
    end

    p = Plots.plot(
        alphas,
        valores_plot;
        xlabel = "alpha",
        ylabel = "f(x + alpha d)",
        label = "limited function",
        linewidth = 2,
        legend = :topright,
        title = titulo,
        ylims = (y_min, y_max),
    )

    p2 = Plots.plot(
        alphas,
        valores_plot_RMSD;
        xlabel = "alpha",
        ylabel = "RMSD(x + alpha d)",
        label = "limited RMSD",
        linewidth = 2,
        legend = :topright,
        ylims = (y_min_RMSD, y_max_RMSD),
    )

    Plots.hline!(p, [Float64(fmax)]; label = "teto = $(Float64(fmax))", linestyle = :dash)

    if output !== nothing
        mkpath(dirname(String(output)))
        Plots.savefig(p, String(output))
    end

    if output_RMSD !== nothing
        mkpath(dirname(String(output_RMSD)))
        Plots.savefig(p2, String(output_RMSD))
    end

    if data_output !== nothing
        mkpath(dirname(String(data_output)))
        colunas_x = ["x$i" for i in 1:size(pontos, 2)]
        tabela = Matrix{Any}(undef, length(alphas) + 1, 4 + length(colunas_x))
        tabela[1, :] .= vcat(["alpha", "f", "f_plot", "RMSD_plot"], colunas_x)
        tabela[2:end, 1] .= alphas
        tabela[2:end, 2] .= valores
        tabela[2:end, 3] .= valores_plot
        tabela[2:end, 4] .= valores_plot_RMSD
        tabela[2:end, 5:end] .= pontos
        writedlm(data_output, tabela, ',')
    end

    return (;
        plot = p,
        plot_RMSD = p2,
        output,
        output_RMSD,
        data_output,
        alphas,
        valores,
        valores_plot,
        valores_plot_RMSD,
        pontos,
        direcao,
        gradiente = copy(g),
        gradiente_norma = norm(g),
        fmax = Float64(fmax),
        sentido,
    )
end

function le_busca_exaustiva_derivada(data_output)
    dados = readdlm(data_output, ',', Any)
    if size(dados, 2) < 4
        error("Arquivo invalido: esperado colunas alpha, f, f_plot e pelo menos um x.")
    end

    alphas = Float64.(dados[2:end, 1])
    valores = Float64.(dados[2:end, 2])
    valores_plot = Float64.(dados[2:end, 3])
    tem_rmsd = size(dados, 2) >= 5 && String(dados[1, 4]) == "RMSD_plot"
    valores_plot_RMSD = tem_rmsd ?
        Float64.(dados[2:end, 4]) :
        sqrt.(max.(valores_plot, 0.0) ./ length(alphas))
    primeira_coluna_x = tem_rmsd ? 5 : 4
    pontos = Matrix{Float64}(undef, length(alphas), size(dados, 2) - primeira_coluna_x + 1)

    for j in axes(pontos, 2)
        pontos[:, j] .= Float64.(dados[2:end, j + primeira_coluna_x - 1])
    end

    return (;
        data_output,
        alphas,
        valores,
        valores_plot,
        valores_plot_RMSD,
        pontos,
    )
end


function expande_dimensao(x_antigo::AbstractVector, novo_n_man::Int)
    n_antigo = length(x_antigo)

    if novo_n_man == n_antigo
        return copy(x_antigo)
    end

    if novo_n_man < 1
        error("A nova dimensão deve ser pelo menos 1.")
    end

    if n_antigo == 1
        return fill(x_antigo[1], novo_n_man)
    end

    if novo_n_man == 1
        return [x_antigo[1]]
    end

    x_old_grid = collect(range(0.0, 1.0, length = n_antigo))
    x_new_grid = collect(range(0.0, 1.0, length = novo_n_man))

    x_novo = similar(x_antigo, novo_n_man)

    for (j, xj) in enumerate(x_new_grid)

        if xj <= x_old_grid[1]
            x_novo[j] = x_antigo[1]
        elseif xj >= x_old_grid[end]
            x_novo[j] = x_antigo[end]
        else
            i = searchsortedlast(x_old_grid, xj)

            x_left = x_old_grid[i]
            x_right = x_old_grid[i+1]

            y_left = x_antigo[i]
            y_right = x_antigo[i+1]

            θ = (xj - x_left) / (x_right - x_left)

            x_novo[j] = (1 - θ) * y_left + θ * y_right
        end
    end

    return x_novo
end



function penalidade_caixa_externa(x, lb, ub; rho=1.0e6)
    penalidade = zero(eltype(x))
    for i in eachindex(x)
        abaixo = max(zero(x[i]), lb[i] - x[i])
        acima = max(zero(x[i]), x[i] - ub[i])
        penalidade += abaixo^2 + acima^2
    end
    return rho * penalidade
end

function objetivo_sse_assimilacao(fun, tbeg, tend, estado_prev; fallback = 1.0e26, mostra_rmsd = false)
    return function f_base(x)
        resultado = fun(x, tbeg, tend, estado_prev)
        erro = resultado.erro

        if mostra_rmsd && !(eltype(x) <: ForwardDiff.Dual)
            RMSD_atual = norm(erro / sqrt(size(erro, 1)))
            print("RMSD = $RMSD_atual \n")
        end

        sse = dot(erro, erro)
        return isfinite(sse) ? sse : fallback
    end
end

function objetivo_penalizado_caixa(f_base, lb, ub; rho = 1.0e6)
    return x -> f_base(x) + penalidade_caixa_externa(x, lb, ub; rho = rho)
end

function objetivo_penalizado_caixa_com_gradiente(f_base, x_ref, lb, ub; rho = 1.0e6)
    f_penalizada = objetivo_penalizado_caixa(f_base, lb, ub; rho = rho)
    dim = length(x_ref)
    cfg = ForwardDiff.GradientConfig(f_penalizada, x_ref, ForwardDiff.Chunk{dim}())

    function g!(G, x)
        ForwardDiff.gradient!(G, f_penalizada, x, cfg)
        return G
    end

    return f_penalizada, g!
end


function safe_rand_manning(;n = 101)
    while true
        x = 0.3 .* rand(n)
        val = quad_fun(x)
        if val <=10.0
            return x            
        end
    end
end

struct ArmijoThenWolfe{G}
    g!::G
    c1::Float64
    c2::Float64
    ρ::Float64
    maxiter::Int
end

function ArmijoThenWolfe(g!; c1 = 1e-4, c2 = 0.9, ρ = 0.5, maxiter = 20)
    return ArmijoThenWolfe(g!, c1, c2, ρ, maxiter)
end

function (ls::ArmijoThenWolfe)(d, x, s, α0, x_ls, ϕ0, dϕ0)
    α = α0
    gtrial = similar(x)

    for _ in 1:ls.maxiter
        @. x_ls = x + α * s
        ϕα = Optim.value(d, x_ls)

        if isfinite(ϕα) && ϕα <= ϕ0 + ls.c1 * α * dϕ0
            ls.g!(gtrial, x_ls)
            dϕα = dot(gtrial, s)

            if dϕα >= ls.c2 * dϕ0
                return α, ϕα
            end
        end

        α *= ls.ρ
    end

    x_ls .= x
    return zero(α0), ϕ0
end


struct QuadraticBacktracking
    c1::Float64
    ρ::Float64
    maxiter::Int
    fmax::Float64
end

function QuadraticBacktracking(;
    c1 = 1.0e-4,
    ρ = 0.5,
    maxiter = 20,
    fmax = 10000.0,
)
    return QuadraticBacktracking(
        Float64(c1),
        Float64(ρ),
        Int(maxiter),
        Float64(fmax),
    )
end

function (ls::QuadraticBacktracking)(d, x, s, α0, x_ls, ϕ0, dϕ0)
    α = α0

    for _ in 1:ls.maxiter
        @. x_ls = x + α * s
        ϕα = Optim.value(d, x_ls)

        if isfinite(ϕα) &&
           ϕα < ls.fmax &&
           ϕα <= ϕ0 + ls.c1 * α * dϕ0
            return α, ϕα
        end

        α *= ls.ρ
    end

    x_ls .= x
    return zero(α0), ϕ0
end




struct DefaultLikeLineSearch
    c1::Float64
    c2::Float64
    alpha0::Float64
    alpha_max::Float64
    expansion::Float64
    maxiter::Int
    zoom_maxiter::Int
end

function DefaultLikeLineSearch(;
    c1 = 1.0e-4,
    c2 = 0.9,
    alpha0 = 1.0,
    alpha_max = 10.0,
    expansion = 2.0,
    maxiter = 20,
    zoom_maxiter = 25,
)
    return DefaultLikeLineSearch(
        Float64(c1),
        Float64(c2),
        Float64(alpha0),
        Float64(alpha_max),
        Float64(expansion),
        Int(maxiter),
        Int(zoom_maxiter),
    )
end

function _eval_line_and_grad!(f, g!, x, p, alpha, x_trial, g_trial)
    @. x_trial = x + alpha * p
    phi = Float64(f(x_trial))
    g!(g_trial, x_trial)
    dphi = dot(g_trial, p)
    return phi, dphi
end

function _zoom_default_like!(
    f,
    g!,
    x,
    p,
    phi0,
    dphi0,
    alpha_lo,
    phi_lo,
    dphi_lo,
    alpha_hi,
    alpha_hi_phi,
    alpha_hi_dphi,
    x_trial,
    g_trial,
    ls::DefaultLikeLineSearch,
)
    alpha_left = Float64(alpha_lo)
    phi_left = Float64(phi_lo)
    dphi_left = Float64(dphi_lo)
    alpha_right = Float64(alpha_hi)

    for _ in 1:ls.zoom_maxiter
        alpha = 0.5 * (alpha_left + alpha_right)
        phi, dphi = _eval_line_and_grad!(f, g!, x, p, alpha, x_trial, g_trial)

        if !isfinite(phi) || !isfinite(dphi)
            alpha_right = alpha
            continue
        end

        if (phi > phi0 + ls.c1 * alpha * dphi0) || (phi >= phi_left)
            alpha_right = alpha
            continue
        end

        if abs(dphi) <= -ls.c2 * dphi0
            return (; accepted = true, alpha, phi, dphi, status = "strong_wolfe")
        end

        if dphi * (alpha_right - alpha_left) >= 0
            alpha_right = alpha_left
        end

        alpha_left = alpha
        phi_left = phi
        dphi_left = dphi
    end

    if isfinite(phi_left) && phi_left <= phi0 + ls.c1 * alpha_left * dphi0
        return (;
            accepted = true,
            alpha = alpha_left,
            phi = phi_left,
            dphi = dphi_left,
            status = "armijo_zoom_fallback",
        )
    end

    return (; accepted = false, alpha = 0.0, phi = phi0, dphi = dphi0, status = "zoom_failed")
end

function default_like_linesearch!(
    f,
    g!,
    x,
    p,
    fx,
    gx,
    x_trial,
    g_trial;
    ls = DefaultLikeLineSearch(),
    delta = 0.1,
)
    dphi0 = dot(gx, p)
    if !isfinite(dphi0) || dphi0 >= 0
        return (; accepted = false, alpha = 0.0, phi = fx, dphi = dphi0, status = "not_descent")
    end

    pnorm = norm(p)
    if !isfinite(pnorm) || pnorm <= eps(Float64)
        return (; accepted = false, alpha = 0.0, phi = fx, dphi = dphi0, status = "zero_direction")
    end

    delta_eff = Float64(delta)
    if !(delta_eff > 0)
        delta_eff = Inf
    end

    alpha_prev = 0.0
    phi_prev = fx
    dphi_prev = dphi0
    alpha = min(ls.alpha0, ls.alpha_max)
    trust_region_backtracks = 0

    for iter in 1:ls.maxiter
        if alpha * pnorm > delta_eff
            alpha *= 0.5
            trust_region_backtracks += 1
            if alpha <= eps(Float64)
                break
            end
            continue
        end

        phi, dphi = _eval_line_and_grad!(f, g!, x, p, alpha, x_trial, g_trial)

        if !isfinite(phi) || !isfinite(dphi)
            alpha *= 0.5
            if alpha <= eps(Float64)
                break
            end
            continue
        end

        if (phi > fx + ls.c1 * alpha * dphi0) || (iter > 1 && phi >= phi_prev)
            return _zoom_default_like!(
                f, g!, x, p, fx, dphi0,
                alpha_prev, phi_prev, dphi_prev,
                alpha, phi, dphi,
                x_trial, g_trial, ls,
            )
        end

        if abs(dphi) <= -ls.c2 * dphi0
            return (; accepted = true, alpha, phi, dphi, status = "strong_wolfe")
        end

        if dphi >= 0
            return _zoom_default_like!(
                f, g!, x, p, fx, dphi0,
                alpha, phi, dphi,
                alpha_prev, phi_prev, dphi_prev,
                x_trial, g_trial, ls,
            )
        end

        alpha_prev = alpha
        phi_prev = phi
        dphi_prev = dphi
        alpha = min(ls.expansion * alpha, ls.alpha_max)
    end

    status = trust_region_backtracks > 0 ? "delta_backtracking_failed" : "linesearch_failed"
    return (; accepted = false, alpha = 0.0, phi = fx, dphi = dphi0, status)
end

function _identity_matrix_like(x)
    T = float(eltype(x))
    return Matrix{T}(I, length(x), length(x))
end

function _bfgs_inverse_update!(invH, s, y)
    ys = dot(y, s)
    if !isfinite(ys) || ys <= 0
        return false
    end

    rho = 1 / ys
    invHy = invH * y
    yinvHy = dot(y, invHy)
    coeff = (1 + yinvHy * rho) * rho
    invH .+= coeff .* (s * s') .- rho .* (invHy * s' + s * invHy')
    return true
end

function bfgs_default_like(
    f,
    g!,
    x0;
    maxiter = 100,
    gtol = 1.0e-6,
    x_abstol = 0.0,
    f_abstol = 0.0,
    delta = 0.1,
    linesearch = DefaultLikeLineSearch(),
    initial_invH = nothing,
    reset_if_not_descent = true,
    verbose = false,
)
    x = copy(float.(x0))
    fx = Float64(f(x))
    gx = similar(x)
    g!(gx, x)

    invH = initial_invH === nothing ? _identity_matrix_like(x) : Matrix{Float64}(initial_invH(x))
    x_trial = similar(x)
    g_trial = similar(x)
    s = similar(x)
    y = similar(x)
    history = Vector{NamedTuple}()

    for k in 0:maxiter
        gnorm = norm(gx)
        push!(history, (;
            iter = k,
            f = fx,
            gnorm,
            x = copy(x),
            alpha = k == 0 ? missing : missing,
            accepted = k == 0 ? true : missing,
            linesearch_status = k == 0 ? "initial" : missing,
        ))

        if verbose
            println("bfgs_default_like iter = $k | f = $fx | ||g|| = $gnorm")
        end

        if gnorm <= gtol
            return (; x, f = fx, g = copy(gx), gnorm, invH, iterations = k, status = "gtol", history)
        end

        if k == maxiter
            return (; x, f = fx, g = copy(gx), gnorm, invH, iterations = k, status = "maxiter", history)
        end

        mul!(s, invH, gx)
        rmul!(s, -1.0)
        descent = dot(gx, s)

        if (!isfinite(descent) || descent >= 0) && reset_if_not_descent
            invH .= _identity_matrix_like(x)
            s .= .-gx
        end

        ls_result = default_like_linesearch!(f, g!, x, s, fx, gx, x_trial, g_trial; ls = linesearch, delta = delta)
        accepted = ls_result.accepted && ls_result.alpha > 0

        history[end] = merge(history[end], (;
            alpha = ls_result.alpha,
            accepted,
            linesearch_status = ls_result.status,
            delta,
        ))

        if !accepted
            return (; x, f = fx, g = copy(gx), gnorm, invH, iterations = k, status = ls_result.status, history)
        end

        s .= x_trial .- x
        y .= g_trial .- gx
        stepnorm = norm(s)
        fchange = abs(ls_result.phi - fx)

        x .= x_trial
        gx .= g_trial
        fx = ls_result.phi

        if !_bfgs_inverse_update!(invH, s, y)
            invH .= _identity_matrix_like(x)
        end

        if x_abstol > 0 && stepnorm <= x_abstol
            return (; x, f = fx, g = copy(gx), gnorm = norm(gx), invH, iterations = k + 1, status = "x_abstol", history)
        end

        if f_abstol > 0 && fchange <= f_abstol
            return (; x, f = fx, g = copy(gx), gnorm = norm(gx), invH, iterations = k + 1, status = "f_abstol", history)
        end
    end
end

function passo_cauchy_trust_region(g, B, Δ)
    ng = norm(g)
    if ng == 0 || Δ <= 0
        return zero(g)
    end

    Bg = B * g
    gBg = dot(g, Bg)
    if gBg <= 0
        return -(Δ / ng) .* g
    end

    α = min(dot(g, g) / gBg, Δ / ng)
    return -α .* g
end


function passo_dogleg_bfgs_trust_region(g, B, Δ; regularization = 1.0e-8)
    p_c = passo_cauchy_trust_region(g, B, Δ)
    if norm(p_c) >= Δ
        return p_c
    end

    B_reg = B + regularization * I
    p_b = try
        -(B_reg \ g)
    catch
        p_c
    end

    if norm(p_b) <= Δ
        return p_b
    end

    d = p_b - p_c
    a = dot(d, d)
    b = 2 * dot(p_c, d)
    c = dot(p_c, p_c) - Δ^2
    discriminante = max(0.0, b^2 - 4 * a * c)
    τ = (-b + sqrt(discriminante)) / (2 * a)
    return p_c + τ .* d
end


function trust_region_bfgs(
    f,
    g!,
    x0;
    Δ0 = 1.0,
    Δmax = 1.0,
    η1 = 0.25,
    η2 = 0.75,
    γdec = 0.25,
    γinc = 2.0,
    maxiter = 100,
    gtol = 1.0e-6,
    regularization = 1.0e-8,
    verbose = false,
)
    x = copy(float.(x0))
    n = length(x)
    B = Matrix{Float64}(I, n, n)
    Δ = Float64(Δ0)
    f_x = Float64(f(x))
    g = similar(x)
    g!(g, x)

    history = Vector{NamedTuple}()

    for k in 0:maxiter
        gnorm = norm(g)
        push!(history, (;
            iter = k,
            f = f_x,
            gnorm,
            Δ,
            x = copy(x),
            accepted = k == 0 ? true : missing,
            ρ = k == 0 ? missing : missing,
        ))

        if verbose
            println("trust_region_bfgs iter = $k | f = $f_x | ||g|| = $gnorm | Δ = $Δ")
        end

        if gnorm <= gtol || k == maxiter
            status = gnorm <= gtol ? "gtol" : "maxiter"
            return (;
                x,
                f = f_x,
                g = copy(g),
                gnorm,
                B,
                Δ,
                iterations = k,
                status,
                history,
            )
        end

        p = passo_dogleg_bfgs_trust_region(g, B, Δ; regularization = regularization)
        predicted_reduction = -(dot(g, p) + 0.5 * dot(p, B * p))

        if predicted_reduction <= 0 || !isfinite(predicted_reduction)
            p = passo_cauchy_trust_region(g, B, Δ)
            predicted_reduction = -(dot(g, p) + 0.5 * dot(p, B * p))
        end

        x_trial = x + p
        f_trial = Float64(f(x_trial))
        actual_reduction = f_x - f_trial
        ρ = predicted_reduction > 0 ? actual_reduction / predicted_reduction : -Inf
        accepted = isfinite(f_trial) && ρ > η1

        if ρ < η1
            Δ = max(γdec * Δ, eps(Float64))
        elseif ρ > η2 && norm(p) >= 0.8 * Δ
            Δ = min(γinc * Δ, Δmax)
        end

        if accepted
            g_trial = similar(g)
            g!(g_trial, x_trial)

            s = x_trial - x
            y = g_trial - g
            ys = dot(y, s)

            if ys > 1.0e-12 * norm(y) * norm(s)
                Bs = B * s
                sBs = dot(s, Bs)
                if sBs > 0
                    B = B - (Bs * Bs') / sBs + (y * y') / ys
                else
                    B = Matrix{Float64}(I, n, n)
                end
            end

            x = x_trial
            f_x = f_trial
            g .= g_trial
        end

        history[end] = merge(history[end], (; accepted, ρ))
    end
end

function trust_region_bfgs(
    f,
    g!,
    x0;
    delta0 = 0.25,
    delta_max = 1.0,
    eta1 = 0.10,
    eta2 = 0.90,
    gamma_dec = 0.5,
    gamma_inc = 2.0,
    Δ0 = nothing,
    Δmax = nothing,
    η1 = nothing,
    η2 = nothing,
    γdec = nothing,
    γinc = nothing,
    maxiter = 100,
    gtol = 1.0e-6,
    regularization = 1.0e-8,
    lower = nothing,
    upper = nothing,
    invalid_f_threshold = 1000.0,
    scale_initial_hessian = true,
    verbose = false,
)
    x = copy(float.(x0))
    n = length(x)
    delta0 = Δ0 === nothing ? delta0 : Float64(Δ0)
    delta_max = Δmax === nothing ? delta_max : Float64(Δmax)
    eta1 = η1 === nothing ? eta1 : Float64(η1)
    eta2 = η2 === nothing ? eta2 : Float64(η2)
    gamma_dec = γdec === nothing ? gamma_dec : Float64(γdec)
    gamma_inc = γinc === nothing ? gamma_inc : Float64(γinc)

    if lower === nothing && upper === nothing
        lower_vec = zeros(n)
        upper_vec = fill(0.5, n)
    else
        lower_vec = lower === nothing ? fill(-Inf, n) : Float64.(collect(lower))
        upper_vec = upper === nothing ? fill(Inf, n) : Float64.(collect(upper))
    end

    if length(lower_vec) != n || length(upper_vec) != n
        error("lower e upper devem ter o mesmo tamanho de x0.")
    end

    x .= clamp.(x, lower_vec, upper_vec)

    B = Matrix{Float64}(I, n, n)
    delta = min(Float64(delta0), Float64(delta_max))
    f_x = Float64(f(x))
    g = similar(x)
    g!(g, x)

    if scale_initial_hessian
        scale = max(norm(g), 1.0)
        if isfinite(scale) && scale > 0
            B .*= scale
        end
    end

    history = Vector{NamedTuple}()

    for k in 0:maxiter
        gnorm = norm(g)
        push!(history, (;
            iter = k,
            f = f_x,
            gnorm,
            delta,
            x = copy(x),
            accepted = k == 0 ? true : missing,
            rho = k == 0 ? missing : missing,
        ))

        if verbose
            println("trust_region_bfgs iter = $k | f = $f_x | ||g|| = $gnorm | delta = $delta")
        end

        if gnorm <= gtol || k == maxiter
            status = gnorm <= gtol ? "gtol" : "maxiter"
            return (;
                x,
                f = f_x,
                g = copy(g),
                gnorm,
                B,
                delta,
                Δ = delta,
                iterations = k,
                status,
                history,
            )
        end

        p = passo_dogleg_bfgs_trust_region(g, B, delta; regularization = regularization)
        x_trial = clamp.(x .+ p, lower_vec, upper_vec)
        p = x_trial - x
        pnorm = norm(p)

        if pnorm <= eps(Float64) * max(1.0, norm(x))
            delta = max(gamma_dec * delta, eps(Float64))
            history[end] = merge(history[end], (; accepted = false, rho = -Inf, pnorm, invalid_trial = false))
            continue
        end

        predicted_reduction = -(dot(g, p) + 0.5 * dot(p, B * p))

        if predicted_reduction <= 0 || !isfinite(predicted_reduction)
            p = passo_cauchy_trust_region(g, B, delta)
            x_trial = clamp.(x .+ p, lower_vec, upper_vec)
            p = x_trial - x
            pnorm = norm(p)

            if pnorm <= eps(Float64) * max(1.0, norm(x))
                delta = max(gamma_dec * delta, eps(Float64))
                history[end] = merge(history[end], (; accepted = false, rho = -Inf, pnorm, invalid_trial = false))
                continue
            end

            predicted_reduction = -(dot(g, p) + 0.5 * dot(p, B * p))
        end

        f_trial = Float64(f(x_trial))
        invalid_trial = !isfinite(f_trial) || isnan(f_trial) || f_trial > invalid_f_threshold

        if invalid_trial
            delta = max(gamma_dec * pnorm, eps(Float64))
            history[end] = merge(history[end], (; accepted = false, rho = -Inf, pnorm, invalid_trial, f_trial))
            continue
        end

        actual_reduction = f_x - f_trial
        rho = predicted_reduction > 0 ? actual_reduction / predicted_reduction : -Inf
        accepted = isfinite(f_trial) && rho > eta1

        if rho < eta1
            delta = max(gamma_dec * pnorm, eps(Float64))
        elseif rho > eta2 && pnorm >= 0.8 * delta
            delta = min(gamma_inc * delta, delta_max)
        end

        if accepted
            g_trial = similar(g)
            g!(g_trial, x_trial)

            s = x_trial - x
            y = g_trial - g
            ys = dot(y, s)

            if ys > 1.0e-12 * norm(y) * norm(s)
                Bs = B * s
                sBs = dot(s, Bs)
                if sBs > 0
                    B = B - (Bs * Bs') / sBs + (y * y') / ys
                else
                    B = Matrix{Float64}(I, n, n)
                end
            end

            x = x_trial
            f_x = f_trial
            g .= g_trial
        end

        history[end] = merge(history[end], (; accepted, rho, pnorm, invalid_trial = false, f_trial))
    end
end


using Optim
using LineSearches

struct SafeThenBackTracking{BT}
    fmax::Float64
    ρ::Float64
    maxiter::Int
    bt::BT
end

SafeThenBackTracking(; fmax = 10000.0, ρ = 0.5, maxiter = 20, hagerzhang_maxiter = 50) =
    SafeThenBackTracking(
        fmax,
        ρ,
        maxiter,
        LineSearches.HagerZhang(linesearchmax = Int(hagerzhang_maxiter)),
    )

    function (ls::SafeThenBackTracking)(d, x, s, α0, x_ls, ϕ0, dϕ0)
        α = α0 > 0 ? α0 : one(α0)
    
        for _ in 1:ls.maxiter
            @. x_ls = x + α * s
            ϕα = Optim.value(d, x_ls)
    
            if isfinite(ϕα) && ϕα < ls.fmax && ϕα > 0.6 * ϕ0
    
                return ls.bt(d, x, s, α, x_ls, ϕ0, dϕ0)
    
            elseif isfinite(ϕα) && ϕα < ls.fmax && ϕα <= 0.6 * ϕ0
    
                αq, ϕq, accepted = quadratic_minimum_trial!(
                    d, x, s, α, x_ls, ϕ0, dϕ0, ϕα;
                    c1 = 1.0e-4,
                    fmax = ls.fmax,
                )
    
                if accepted
                    return αq, ϕq
                end
    
                # Se o mínimo quadrático não foi aceito,
                # mas o passo atual já satisfaz Armijo, aceita α.
                if ϕα <= ϕ0 + 1.0e-4 * α * dϕ0
                    return α, ϕα
                end
            end
    
            α *= ls.ρ
        end
    
        x_ls .= x
        return zero(α0), ϕ0
    end




function quadratic_minimum_trial!(
    d, x, s, α, x_ls, ϕ0, dϕ0, ϕα;
    c1 = 1.0e-4,
    fmax = 10000.0,
    max_expand = 5.0,
    )

    # Precisamos de uma direção de descida e de um passo positivo
    if !(α > 0) || !(dϕ0 < 0)
        return α, ϕα, false
    end

    # q(t) = ϕ0 + dϕ0*t + a*t^2
    a = (ϕα - ϕ0 - dϕ0 * α) / α^2

    # Se a <= 0, o polinômio não tem mínimo finito
    if !(isfinite(a)) || a <= 0
        return α, ϕα, false
    end

    # Mínimo do polinômio quadrático
    αq = -dϕ0 / (2a)

    if !(isfinite(αq)) || αq <= 0
        return α, ϕα, false
    end

    # Salvaguarda para não deixar o passo explodir
    αq = min(αq, max_expand * α)

    # Avalia a função real no mínimo previsto pelo polinômio
    @. x_ls = x + αq * s
    ϕq = Optim.value(d, x_ls)

    # Aceita apenas se a função real for válida e satisfizer Armijo
    if isfinite(ϕq) &&
       ϕq < fmax &&
       ϕq <= ϕ0 + c1 * αq * dϕ0
        return αq, ϕq, true
    end

    # Se não aceitou αq, restaura x_ls para o passo original α
    @. x_ls = x + α * s

    return α, ϕα, false
end


# ng = fill(0.08, n_man + 1)
# run_L_sv_bfgs_bobyqa_tmax = true
# result = run_L_sv_bfgs_bobyqa_tmax ? testar_L_sv_bfgs_bobyqa_tmax(ng) : ng
