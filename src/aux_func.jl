
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

    melhor_α = α
    melhor_ϕ = ϕ0
    achou_armijo = false

    gtrial = similar(x)

    for k in 1:ls.maxiter

        # x_ls = x + α s
        @. x_ls = x + α * s

        # Avalia somente a função objetivo por meio do objeto interno do Optim
        ϕα = Optim.value(d, x_ls)

        # Armijo
        if isfinite(ϕα) && ϕα <= ϕ0 + ls.c1 * α * dϕ0

            achou_armijo = true
            melhor_α = α
            melhor_ϕ = ϕα

            # Agora sim calcula o gradiente usando a SUA função
            ls.g!(gtrial, x_ls)

            dϕα = dot(gtrial, s)

            # Wolfe fraco
            if dϕα >= ls.c2 * dϕ0
                return α, ϕα
            end
        end

        α *= ls.ρ
    end

    if achou_armijo
        return melhor_α, melhor_ϕ
    else
        x_ls .= x
        return zero(α), ϕ0
    end
end





# ng = fill(0.08, n_man + 1)
# run_L_sv_bfgs_bobyqa_tmax = true
# result = run_L_sv_bfgs_bobyqa_tmax ? testar_L_sv_bfgs_bobyqa_tmax(ng) : ng
