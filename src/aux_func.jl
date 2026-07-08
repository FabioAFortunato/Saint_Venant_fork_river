
using DelimitedFiles

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
    n_points = 200,
    fmax = 100.0,
    sentido = :descida,
    output = arquivo_em_results("busca_exaustiva_derivada.png"),
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

    y_min = minimum(valores_plot)
    y_max = Float64(fmax)
    if y_min == y_max
        y_min -= 1.0
    end

    p = Plots.plot(
        alphas,
        valores_plot;
        xlabel = "alpha",
        ylabel = "f(x + alpha d)",
        label = "funcao limitada",
        linewidth = 2,
        legend = :topright,
        title = titulo,
        ylims = (y_min, y_max),
    )

    Plots.hline!(p, [Float64(fmax)]; label = "teto = $(Float64(fmax))", linestyle = :dash)

    if output !== nothing
        mkpath(dirname(String(output)))
        Plots.savefig(p, String(output))
    end

    if data_output !== nothing
        mkpath(dirname(String(data_output)))
        colunas_x = ["x$i" for i in 1:size(pontos, 2)]
        tabela = Matrix{Any}(undef, length(alphas) + 1, 3 + length(colunas_x))
        tabela[1, :] .= vcat(["alpha", "f", "f_plot"], colunas_x)
        tabela[2:end, 1] .= alphas
        tabela[2:end, 2] .= valores
        tabela[2:end, 3] .= valores_plot
        tabela[2:end, 4:end] .= pontos
        writedlm(data_output, tabela, ',')
    end

    return (;
        plot = p,
        output,
        data_output,
        alphas,
        valores,
        valores_plot,
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
    pontos = Matrix{Float64}(undef, length(alphas), size(dados, 2) - 3)

    for j in axes(pontos, 2)
        pontos[:, j] .= Float64.(dados[2:end, j + 3])
    end

    return (;
        data_output,
        alphas,
        valores,
        valores_plot,
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
    min_factor::Float64
    max_factor::Float64
    flat_tol::Float64
end

function QuadraticBacktracking(;
    c1 = 1.0e-4,
    ρ = 0.5,
    maxiter = 20,
    fmax = 10000.0,
    min_factor = 0.1,
    max_factor = 0.5,
    flat_tol = 1.0e-8,
)
    return QuadraticBacktracking(
        Float64(c1),
        Float64(ρ),
        Int(maxiter),
        Float64(fmax),
        Float64(min_factor),
        Float64(max_factor),
        Float64(flat_tol),
    )
end

function valores_quase_planos(valores, flat_tol)
    escala = max(1.0, maximum(abs.(valores)))
    return maximum(valores) - minimum(valores) <= flat_tol * escala
end

function (ls::QuadraticBacktracking)(d, x, s, α0, x_ls, ϕ0, dϕ0)
    α = α0
    bons_α = Float64[]
    bons_ϕ = Float64[]
    melhor_α = zero(α0)
    melhor_ϕ = ϕ0

    for _ in 1:ls.maxiter
        @. x_ls = x + α * s
        ϕα = Optim.value(d, x_ls)

        if isfinite(ϕα) && ϕα < ls.fmax
            push!(bons_α, Float64(α))
            push!(bons_ϕ, Float64(ϕα))

            if ϕα <= ϕ0 + ls.c1 * α * dϕ0
                melhor_α = α
                melhor_ϕ = ϕα
            end

            if length(bons_α) >= 2
                α1, α2 = bons_α[end-1], bons_α[end]
                ϕ1, ϕ2 = bons_ϕ[end-1], bons_ϕ[end]
                d1 = ϕ1 - ϕ0
                d2 = ϕ2 - ϕ0

                if valores_quase_planos((ϕ0, ϕ1, ϕ2), ls.flat_tol)
                    if melhor_α > 0
                        @. x_ls = x + melhor_α * s
                        return melhor_α, melhor_ϕ
                    end

                    x_ls .= x
                    return zero(α0), ϕ0
                end

                c = (d2 / α2 - d1 / α1) / (α2 - α1)
                b = d1 / α1 - c * α1
                α_quad = c > 0 ? -b / (2 * c) : NaN

                if isfinite(α_quad) && α_quad > 0
                    limite_min = ls.min_factor * min(α1, α2)
                    limite_max = max(α1, α2)
                    α_quad = clamp(α_quad, limite_min, limite_max)

                    @. x_ls = x + α_quad * s
                    ϕ_quad = Optim.value(d, x_ls)

                    if valores_quase_planos((ϕ0, ϕ1, ϕ2, ϕ_quad), ls.flat_tol)
                        if melhor_α > 0
                            @. x_ls = x + melhor_α * s
                            return melhor_α, melhor_ϕ
                        end

                        x_ls .= x
                        return zero(α0), ϕ0
                    end

                    if ϕ_quad <= ϕ0 + ls.c1 * α_quad * dϕ0
                        return α_quad, ϕ_quad
                    end
                end

                if melhor_α > 0
                    @. x_ls = x + melhor_α * s
                    return melhor_α, melhor_ϕ
                end

                α *= ls.ρ
            else
                α *= ls.ρ
            end
        else
            α *= ls.ρ
        end
    end

    if melhor_α > 0
        @. x_ls = x + melhor_α * s
        return melhor_α, melhor_ϕ
    end

    x_ls .= x
    return zero(α0), ϕ0
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
    Δ0 = 0.1,
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




# ng = fill(0.08, n_man + 1)
# run_L_sv_bfgs_bobyqa_tmax = true
# result = run_L_sv_bfgs_bobyqa_tmax ? testar_L_sv_bfgs_bobyqa_tmax(ng) : ng
