function derivativefree_full_dim_problem(;
    x0 = collect(range(0.1, 0.2, length = 101)),
    lb = 0.0,
    ub = 0.5,
    iterations = 10,
    penalty_weight = 1.0e6,
    f_calls_limit = 50,
    g_calls_limit = 20,
    arquivo_aceitos = "results/full_dim_problem_pontos_aceitos.txt",
    arquivo_avaliacoes = "results/full_dim_problem_avaliacoes.txt",
    )
    n = length(x0)
    X = copy(float.(x0))
    limites_inferiores = fill(Float64(lb), n)
    limites_superiores = fill(Float64(ub), n)
    arquivo_pontos = arquivo_em_results(arquivo_aceitos)

    global n_calfun = 0
    global inicio_s = time()
    global nome = arquivo_em_results(arquivo_avaliacoes)
    colunas_x = join(("x$i" for i in 1:n), "\t")

    open(nome, "w") do file
        write(file, "Avaliacoes normais de quad_fun no full_dim_problem\n")
        write(file, "tempo_s\tn_calfun\tRMSD\terro_quadratico_total\n")
    end

    open(arquivo_pontos, "w") do file
        write(file, "Pontos aceitos no full_dim_problem\n")
        write(file, "Metodo = BFGS() com penalidade externa de caixa\n")
        write(file, "x0 = $(collect(X))\n")
        write(file, "lb = $lb | ub = $ub\n")
        write(file, "iterations = $iterations | penalty_weight = $penalty_weight\n")
        write(file, "\niter\tf\tgnorm\t$colunas_x\n")
    end

    x_avaliados = Vector{Vector{Float64}}()
    f_avaliados = Float64[]
    x_gradientes = Vector{Vector{Float64}}()
    gnorm_gradientes = Float64[]

    function f_obj(x_vars)
        valor = quad_fun(x_vars)
        if !(eltype(x_vars) <: ForwardDiff.Dual)
            push!(x_avaliados, copy(float.(x_vars)))
            push!(f_avaliados, Float64(valor))
        end
        return valor
    end
    f_penalizada(x_vars) = f_obj(x_vars) +
                           penalidade_caixa_externa(
                               x_vars,
                               limites_inferiores,
                               limites_superiores;
                               rho=penalty_weight,
                           )

    cfg_bfgs = ForwardDiff.GradientConfig(f_penalizada, X, ForwardDiff.Chunk{n}())
    function g_obj!(G, x_k)
        ForwardDiff.gradient!(G, f_penalizada, x_k, cfg_bfgs)
        push!(x_gradientes, copy(float.(x_k)))
        push!(gnorm_gradientes, norm(G))
        return G
    end

    pontos_aceitos = Vector{Vector{Float64}}()
    valores_aceitos = Float64[]
    valores_penalizados_aceitos = Float64[]
    gnorms_aceitos = Float64[]

    function valor_mais_recente(x_atual, pontos, valores, fallback)
        for i in length(pontos):-1:1
            if norm(x_atual - pontos[i]) <= 1.0e-10
                return valores[i]
            end
        end
        return fallback
    end

    function salva_ponto_aceito!(estado)
        x_atual = copy(estado.x)

        f_atual = valor_mais_recente(x_atual, x_avaliados, f_avaliados, estado.f_x)
        gnorm_atual = valor_mais_recente(x_atual, x_gradientes, gnorm_gradientes, norm(estado.g_x))

        push!(pontos_aceitos, x_atual)
        push!(valores_aceitos, f_atual)
        push!(valores_penalizados_aceitos, Float64(estado.f_x))
        push!(gnorms_aceitos, gnorm_atual)

        open(arquivo_pontos, "a") do file
            iter = length(pontos_aceitos) - 1
            write(file, "$iter\t$f_atual\t$gnorm_atual\t$(join(x_atual, "\t"))\n")
        end
        return false
    end

    resultado_bfgs = Optim.optimize(
        f_penalizada,
        g_obj!,
        X,
        BFGS(),
        Optim.Options(
            iterations = iterations,
            f_calls_limit = f_calls_limit,
            g_calls_limit = g_calls_limit,
            callback = salva_ponto_aceito!,
        ),
    )

    if isempty(pontos_aceitos)
        x_final = Optim.minimizer(resultado_bfgs)
        f_final_penalizado = Optim.minimum(resultado_bfgs)
        f_final = valor_mais_recente(x_final, x_avaliados, f_avaliados, f_final_penalizado)
    else
        melhor_indice = argmin(valores_penalizados_aceitos)
        x_final = copy(pontos_aceitos[melhor_indice])
        f_final = valores_aceitos[melhor_indice]
        f_final_penalizado = valores_penalizados_aceitos[melhor_indice]
    end

    open(arquivo_pontos, "a") do file
        write(file, "\nResumo final\n")
        write(file, "convergiu = $(Optim.converged(resultado_bfgs))\n")
        write(file, "iteracoes = $(Optim.iterations(resultado_bfgs))\n")
        write(file, "f_final = $f_final\n")
        write(file, "f_final_penalizado = $f_final_penalizado\n")
        write(file, "x_final = $(collect(x_final))\n")
        write(file, "n_pontos_aceitos = $(length(pontos_aceitos))\n")
    end

    return (;
        resultado = resultado_bfgs,
        x_final,
        f_final,
        pontos_aceitos,
        valores_aceitos,
        gnorms_aceitos,
        arquivo_pontos,
        arquivo_avaliacoes = nome,
    )
end

function bobyqa_two_dim_problem(;
    x0 = [0.1; 0.20],
    lb = 0.0,
    ub = 0.5,
    rhobeg=0.05,
    rhoend=0.001,
    maxeval=500,
    arquivo_aceitos = "results/derivate_free/bobyqa_two_dim_points.txt",
    arquivo_avaliacoes = "results/derivate_free/bobyqa_two_dim_problem_avaliacoes.txt",
    )
    n = length(x0)
    npt = n+1
    X = copy(float.(x0))
    limites_inferiores = fill(Float64(lb), n)
    limites_superiores = fill(Float64(ub), n)
    arquivo_pontos = arquivo_em_results(arquivo_aceitos)

    global n_calfun = 0
    global inicio_s = time()
    global nome = arquivo_em_results(arquivo_avaliacoes)

    open(nome, "w") do file
        write(file, "Avaliacoes normais de quad_fun no two_dim_problem\n")
        write(file, "tempo_s\tn_calfun\tRMSD\terro_quadratico_total\n")
    end

    open(arquivo_pontos, "w") do file
        write(file, "Pontos aceitos no two_dim_problem\n")
        write(file, "x0 = $(collect(X))\n")
        write(file, "lb = $lb | ub = $ub\n")
        write(file, "max_evaluation = $maxeval\n")
    end

    x_avaliados = Vector{Vector{Float64}}()
    f_avaliados = Float64[]
    x_gradientes = Vector{Vector{Float64}}()
    gnorm_gradientes = Float64[]

    function f_obj(x_vars)
        valor = quad_fun(x_vars)
        if !(eltype(x_vars) <: ForwardDiff.Dual)
            push!(x_avaliados, copy(float.(x_vars)))
            push!(f_avaliados, Float64(valor))
        end
        return valor
    end

    cfg_bfgs = ForwardDiff.GradientConfig(f_obj, X, ForwardDiff.Chunk{n}())
    function g_obj!(G, x_k)
        ForwardDiff.gradient!(G, f_obj, x_k, cfg_bfgs)
        push!(x_gradientes, copy(float.(x_k)))
        push!(gnorm_gradientes, norm(G))
        return G
    end

    pontos_aceitos = Vector{Vector{Float64}}()
    valores_aceitos = Float64[]

    function valor_mais_recente(x_atual, pontos, valores, fallback)
        for i in length(pontos):-1:1
            if norm(x_atual - pontos[i]) <= 1.0e-10
                return valores[i]
            end
        end
        return fallback
    end

    function salva_ponto_aceito!(estado)
        x_atual = copy(estado.x)

        f_atual = valor_mais_recente(x_atual, x_avaliados, f_avaliados, estado.f_x)

        push!(pontos_aceitos, x_atual)
        push!(valores_aceitos, f_atual)

        open(arquivo_pontos, "a") do file
            iter = length(pontos_aceitos) - 1
            write(file, "$iter\t$f_atual\t$(x_atual[1])\t$(x_atual[2])\n")
        end
        return false
    end


    opt = Opt(:LN_BOBYQA, n)
    opt.lower_bounds = limites_inferiores
    opt.upper_bounds = limites_superiores
    opt.initial_step = fill(Float64(rhobeg), n)
    opt.xtol_abs = fill(Float64(rhoend), n)
    opt.maxeval = maxeval
    opt.population = npt
    opt.min_objective = f_obj

    f_final, x_final, status_bobyqa = NLopt.optimize(opt, x_inicial_bobyqa)

    open(arquivo_pontos, "a") do file
        write(file, "\nResumo final\n")
        write(file, "convergiu = $(Optim.converged(resultado_bfgs))\n")
        write(file, "iteracoes = $(Optim.iterations(resultado_bfgs))\n")
        write(file, "f_final = $f_final\n")
        write(file, "x_final = $(collect(x_final))\n")
        write(file, "n_pontos_aceitos = $(length(pontos_aceitos))\n")
    end

    return (;
        resultado = resultado_bfgs,
        x_final,
        f_final,
        pontos_aceitos,
        valores_aceitos,
        gnorms_aceitos,
        arquivo_pontos,
        arquivo_avaliacoes = nome,
    )
end
