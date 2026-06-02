using Optim
using NLopt
using ForwardDiff
using LinearAlgebra

include("obj_func.jl")
include("aux_func.jl")

const FIRST_ORDER_RESULTS_DIR = normpath(joinpath(@__DIR__, "..", "results", "first_order"))

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

function full_dim_problem(;
    x0 = collect(range(0.1, 0.2, length = 101)),
    lb = 0.0,
    ub = 0.5,
    iterations = 10,
    penalty_weight = 1.0e6,
    f_calls_limit = 50,
    g_calls_limit = 20,
    arquivo_aceitos = joinpath(FIRST_ORDER_RESULTS_DIR, "full_dim_problem_pontos_aceitos.txt"),
    arquivo_avaliacoes = joinpath(FIRST_ORDER_RESULTS_DIR, "full_dim_problem_avaliacoes.txt"),
)
    n = length(x0)
    X = copy(float.(x0))
    limites_inferiores = fill(Float64(lb), n)
    limites_superiores = fill(Float64(ub), n)
    arquivo_pontos = normpath(arquivo_aceitos)
    mkpath(dirname(arquivo_pontos))

    global n_calfun = 0
    global inicio_s = time()
    global nome = normpath(arquivo_avaliacoes)
    mkpath(dirname(nome))
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

function two_dim_problem(;
    x0 = [0.1; 0.20],
    lb = 0.0,
    ub = 0.5,
    iterations = 10,
    penalty_weight = 1.0e6,
    f_calls_limit = 50,
    g_calls_limit = 20,
    arquivo_aceitos = joinpath(FIRST_ORDER_RESULTS_DIR, "BFGS_two_dim_problem_pontos_aceitos.txt"),
    arquivo_avaliacoes = joinpath(FIRST_ORDER_RESULTS_DIR, "BFGS_two_dim_problem_avaliacoes.txt"),
    )
    n = length(x0)
    X = copy(float.(x0))
    limites_inferiores = fill(Float64(lb), n)
    limites_superiores = fill(Float64(ub), n)
    arquivo_pontos = normpath(arquivo_aceitos)
    mkpath(dirname(arquivo_pontos))

    global n_calfun = 0
    global inicio_s = time()
    global nome = normpath(arquivo_avaliacoes)
    mkpath(dirname(nome))

    open(nome, "w") do file
        write(file, "Avaliacoes normais de quad_fun no two_dim_problem\n")
        write(file, "tempo_s\tn_calfun\tRMSD\terro_quadratico_total\n")
    end

    open(arquivo_pontos, "w") do file
        write(file, "Pontos aceitos no two_dim_problem\n")
        write(file, "Metodo = BFGS() com penalidade externa de caixa\n")
        write(file, "x0 = $(collect(X))\n")
        write(file, "lb = $lb | ub = $ub\n")
        write(file, "iterations = $iterations | penalty_weight = $penalty_weight\n")
        write(file, "\niter\tf\tgnorm\tx1\tx2\n")
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
            write(file, "$iter\t$f_atual\t$gnorm_atual\t$(x_atual[1])\t$(x_atual[2])\n")
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






function minimizar_L_sv_bfgs(
    x::AbstractVector{T};
    tmax=5.0,
    arquivo_saida=nothing,
    rotulo_execucao="BFGS",
    n_restart=1,
    ) where T<:Real
    n = length(x)
    lb = fill(0.0, n)
    ub = fill(0.5, n)

    global n_calfun = 0
    global inicio_s = time()
    global nome = arquivo_em_results(
        arquivo_saida === nothing ? "results/teste_bfgs_$(label_tmax(tmax)).txt" : arquivo_saida,
    )

    rho = 10.0
    lambda = zeros(2 * nt)
    bfgs_iterations = 10
    fminbox_outer_iterations = 10
    f_calls_limit_bfgs = 50
    g_calls_limit_bfgs = 20
    taxa_inicial = 0.05
    taxa_min = taxa_inicial
    tentativas_por_taxa = 10
    rmsd_tol_restart = 0.06
    X = copy(float.(x))
    tempo_total_inicio = time()

    open(nome, "w") do file
        write(file, "Metodo: BFGS em L_sv\n")
        write(file, "lambda fixo = 0.0\n")
        write(file, "rho fixo = $rho\n")
        write(file, "tmax = $tmax\n")
        write(file, "lb = 0.0 | ub = 0.5\n")
        write(file, "bfgs_iterations = $bfgs_iterations | fminbox_outer_iterations = $fminbox_outer_iterations | f_calls_limit = $f_calls_limit_bfgs | g_calls_limit = $g_calls_limit_bfgs\n")
        write(file, "restarts = $n_restart\n")
        write(file, "tempo_s\tn_calfun\tRMSD\tsum_x_xref2\n")
    end

    for i in 1:n_restart
        println(n_restart == 1 ? "\n--- $rotulo_execucao ---" : "\n--- RESTART $rotulo_execucao = $i ---")
        n_calfun_inicio = n_calfun

        f_obj(x_vars) = L_sv(x_vars, lambda, rho, tmax)
        cfg_bfgs = ForwardDiff.GradientConfig(f_obj, X, ForwardDiff.Chunk{n}())
        function g_obj!(G, x_k)
            ForwardDiff.gradient!(G, f_obj, x_k, cfg_bfgs)
            return G
        end

        resultado_bfgs = Optim.optimize(
            f_obj,
            g_obj!,
            lb,
            ub,
            X,
            Fminbox(BFGS()),
            Optim.Options(
                iterations=bfgs_iterations,
                outer_iterations=fminbox_outer_iterations,
                f_calls_limit=f_calls_limit_bfgs,
                g_calls_limit=g_calls_limit_bfgs,
            ),
        )

        f_bfgs = Optim.minimum(resultado_bfgs)

        z_erro = sv_fork(X[1:end-1], tmax)
        rmsd = norm(z_erro) / sqrt(length(z_erro))
        f_obj_base = soma_desvio_quadratico(X)
        avaliacoes_restart = n_calfun - n_calfun_inicio

        println("BFGS finalizado | f = $f_bfgs | soma = $f_obj_base | RMSD = $rmsd | avaliacoes = $avaliacoes_restart")

        open(nome, "a") do file
            write(file, "\nBFGS etapa $i\n")
            write(file, "f_bfgs = $f_bfgs\n")
            write(file, "convergiu_bfgs = $(Optim.converged(resultado_bfgs))\n")
            write(file, "iter_bfgs = $(Optim.iterations(resultado_bfgs))\n")
            write(file, "RMSD_bfgs = $rmsd\n")
            write(file, "sum_x_xref2_bfgs = $f_obj_base\n")
            write(file, "avaliacoes_bfgs = $avaliacoes_restart\n")
            write(file, "x_bfgs = $(collect(X))\n")
        end

        if rmsd < rmsd_tol_restart
            println("Parada antecipada no BFGS: RMSD = $rmsd < $rmsd_tol_restart")
            break
        end

        if i < n_restart
            X = busca_perturbacao_segura(
                f_obj,
                X;
                taxa_inicial=taxa_inicial,
                taxa_min=taxa_min,
                tentativas_por_taxa=tentativas_por_taxa,
                lb=lb,
                ub=ub,
            )
        end
    end

    tempo_total = time() - tempo_total_inicio
    open(nome, "a") do file
        write(file, "\ntempo_bfgs_s = $(round(tempo_total, digits=3))\n")
    end

    return X
end


function minimizar_L_sv_bfgs_bobyqa(;
    x0::AbstractVector{T}=fill(0.08, n_man + 1),
    tmax=5.0,
    rhobeg=0.05,
    rhoend=0.001,
    maxeval=500,
    lb=0.0,
    ub=0.5,
    arquivo_saida=nothing,
    ) where T<:Real
    n = length(x0)
    npt = 2 * n + 1
    lambda = zeros(2 * nt)
    rho = 10.0
    limites_inferiores = fill(Float64(lb), n)
    limites_superiores = fill(Float64(ub), n)
    arquivo = arquivo_em_results(
        arquivo_saida === nothing ? "results/bfgs_bobyqa_L_sv_$(label_tmax(tmax)).txt" : arquivo_saida,
    )

    x_bfgs = minimizar_L_sv_bfgs(
        copy(x0);
        tmax=tmax,
        arquivo_saida=arquivo,
        rotulo_execucao="BFGS do teste BFGS + BOBYQA",
        n_restart=1,
    )

    open(arquivo, "a") do file
        write(file, "\n\nEtapa BOBYQA sobre saida do BFGS\n")
        write(file, "Metodo: BFGS seguido de BOBYQA minimizando L_sv\n")
        write(file, "lambda fixo = 0.0\n")
        write(file, "rho fixo = $rho\n")
        write(file, "tmax = $tmax\n")
        write(file, "lb = $lb | ub = $ub\n")
        write(file, "rhobeg = $rhobeg | rhoend = $rhoend | npt = $npt | maxeval = $maxeval\n")
        write(file, "x_inicial_bobyqa = $(collect(x_inicial_bobyqa))\n")
    end

    objetivo_bobyqa(x, grad) = L_sv(x, lambda, rho, tmax)

    opt = Opt(:LN_BOBYQA, n)
    opt.lower_bounds = limites_inferiores
    opt.upper_bounds = limites_superiores
    opt.initial_step = fill(Float64(rhobeg), n)
    opt.xtol_abs = fill(Float64(rhoend), n)
    opt.maxeval = maxeval
    opt.population = npt
    opt.min_objective = objetivo_bobyqa

    println("Iniciando BOBYQA em L_sv apos BFGS | tmax = $tmax | n = $n | npt = $npt | maxeval = $maxeval")
    n_calfun_inicio_bobyqa = n_calfun
    tempo_inicio_bobyqa = time()
    f_bobyqa, x_bobyqa, status_bobyqa = NLopt.optimize(opt, x_inicial_bobyqa)
    tempo_bobyqa = time() - tempo_inicio_bobyqa

    z_erro = sv_fork(x_final[1:end-1], tmax)
    rmsd_final = any(isnan, z_erro) ? Inf : norm(z_erro) / sqrt(length(z_erro))
    f_obj_base = soma_desvio_quadratico(x_final)
    avaliacoes_bobyqa = n_calfun - n_calfun_inicio_bobyqa
    avaliacoes_nlopt = opt.numevals

    println(
        "BOBYQA em L_sv tmax = $tmax finalizado | f = $f_bobyqa | RMSD = $rmsd_final | ",
        "avaliacoes = $avaliacoes_bobyqa | tempo = $(round(tempo_bobyqa, digits=3)) s | status = $status_bobyqa",
    )

    open(arquivo, "a") do file
        write(file, "\nResumo BOBYQA\n")
        write(file, "status_bobyqa = $status_bobyqa\n")
        write(file, "f_bobyqa = $f_bobyqa\n")
        write(file, "RMSD_final_bobyqa = $rmsd_final\n")
        write(file, "sum_x_xref2_final_bobyqa = $f_obj_base\n")
        write(file, "avaliacoes_bobyqa = $avaliacoes_bobyqa\n")
        write(file, "avaliacoes_nlopt_bobyqa = $avaliacoes_nlopt\n")
        write(file, "tempo_bobyqa_s = $(round(tempo_bobyqa, digits=3))\n")
        write(file, "x_final_bobyqa = $(collect(x_final))\n")
    end

    return (
        x_bfgs=x_bfgs,
        x=x_final,
        f=f_bobyqa,
        RMSD=rmsd_final,
        sum_x_xref2=f_obj_base,
        avaliacoes_bobyqa=avaliacoes_bobyqa,
        avaliacoes_nlopt=avaliacoes_nlopt,
        tempo_bobyqa_s=tempo_bobyqa,
        status=status_bobyqa,
        arquivo=arquivo,
    )
end

function BFGS_BOBYQA_test(;
    x0::AbstractVector{T}=fill(0.08, n_man + 1),
    tmax_values=(5.0, 15.0),
    arquivo_prefixo="results/bfgs_bobyqa_L_sv",
    ) where T<:Real
    resultados = Dict{Float64,NamedTuple}()

    for tmax in Float64.(collect(tmax_values))
        arquivo_saida = arquivo_em_results("$(arquivo_prefixo)_$(label_tmax(tmax)).txt")
        println("\n=== TESTE BFGS + BOBYQA EM L_sv | tmax = $tmax ===")
        tempo_inicio = time()
        resultado = minimizar_L_sv_bfgs_bobyqa(
            copy(x0);
            tmax=tmax,
            rhobeg=0.05,
            rhoend=0.001,
            maxeval=500,
            arquivo_saida=arquivo_saida,
        )
        tempo_total = time() - tempo_inicio

        open(arquivo_saida, "a") do file
            write(file, "\nResumo teste BFGS + BOBYQA\n")
            write(file, "tmax_teste = $tmax\n")
            write(file, "tempo_teste_s = $(round(tempo_total, digits=3))\n")
        end

        resultados[tmax] = merge(resultado, (tempo_s=tempo_total,))
    end

    return resultados
end
