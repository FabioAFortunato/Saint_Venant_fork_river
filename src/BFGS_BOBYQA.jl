using Optim
using NLopt
using ForwardDiff
using LinearAlgebra
using LineSearches
using NOMAD

include("obj_func.jl")
include("aux_func.jl")

const FIRST_ORDER_RESULTS_DIR = normpath(joinpath(@__DIR__, "..", "results", "first_order"))
const HYBRID_RESULTS_DIR = normpath(joinpath(@__DIR__, "..", "results", "hybrid"))

function first_order_results_dir_for_tmax(tmax)
    tmax_float = Float64(tmax)
    pasta = isinteger(tmax_float) ? "$(Int(round(tmax_float)))_days" : "$(label_tmax(tmax_float))_days"
    return joinpath(FIRST_ORDER_RESULTS_DIR, pasta)
end

function hybrid_results_dir_for_tmax(tmax)
    tmax_float = Float64(tmax)
    pasta = isinteger(tmax_float) ? "$(Int(round(tmax_float)))_days" : "$(label_tmax(tmax_float))_days"
    return joinpath(HYBRID_RESULTS_DIR, pasta)
end


function run_problems(; 
    X0 = fill(0.09, 2),
    tins = 5.0,
    λ = 1.0,
    penalty_weight = 1.0e6,
    f_calls_limit = 50,
    g_calls_limit = 20,
    iterations = 10,
    rhobeg = 0.05,
    rhoend = 0.001,
    rmsd_tol = 0.09,
    fun::Function = sv_fork_assimilation,
    method = "all"
    )

    X_prev = copy(X0)
    estado_prev = nothing
    tbeg = 0.0
    tend = tins isa Number ? Float64(tins) : Float64(last(tins))
    resultados = Dict{Symbol, Any}()

    dim = length(X_prev)

    println("\n==============================")
    println("Assimilação de tbeg = ", tbeg, " até tend = ", tend)
    println("==============================")

    resultado_inicial = fun(X_prev, tbeg, tend, estado_prev)
    RMSD = norm(resultado_inicial.erro / sqrt(size(resultado_inicial.erro, 1)))

    println("RMSD inicial = ", RMSD)

    if RMSD > 100.0
        println("Chute inicial ficou 0.08")
        return resultados
    end

    lower_bfgs = zeros(dim)
    upper_bfgs = fill(0.5, dim)

    X_ref = copy(X_prev)

    function f_bfgs(ng)
        resultado = fun(ng, tbeg, tend, estado_prev)
        erro = resultado.erro

        if !(eltype(ng) <: ForwardDiff.Dual)
            RMSD_atual = norm(erro / sqrt(size(erro, 1)))
            print("RMSD = $RMSD_atual \n")
        end

        return dot(erro, erro)
    end

    f_penalizada_bfgs(x_vars) = f_bfgs(x_vars) + penalidade_caixa_externa(
            x_vars,
            lower_bfgs,
            upper_bfgs;
            rho = penalty_weight,
        )

    metodos = lowercase(String(method)) == "all" ?
        [:bfgs, :spg, :bobyqa, :mads] :
        [Symbol(lowercase(String(method)))]

    cfg_bfgs = ForwardDiff.GradientConfig(
            f_penalizada_bfgs,
            X_ref,
            ForwardDiff.Chunk{dim}()
        )

    function g_obj!(G, x_k)
        ForwardDiff.gradient!(G, f_penalizada_bfgs, x_k, cfg_bfgs)
        return G
    end

    function arquivo_metodo(metodo)
        return arquivo_em_results("$(uppercase(String(metodo)))_$(label_tmax(tend))_$(dim).txt")
    end

    function inicializa_arquivo(metodo)
        arquivo = arquivo_metodo(metodo)
        colunas_x = join(("x$i" for i in 1:dim), "\t")
        open(arquivo, "w") do file
            write(file, "Metodo = $(uppercase(String(metodo)))\n")
            write(file, "tbeg = $tbeg | tend = $tend | dim = $dim\n")
            write(file, "x0 = $(collect(X_ref))\n")
            write(file, "lower = $(collect(lower_bfgs))\n")
            write(file, "upper = $(collect(upper_bfgs))\n")
            write(file, "rmsd_tol = $rmsd_tol\n")
            write(file, "\niter\tf\t$colunas_x\n")
        end
        return arquivo
    end

    function registra_avaliacao!(arquivo, iter, f, x)
        open(arquivo, "a") do file
            write(file, "$iter\t$f\t$(join(float.(x), "\t"))\n")
        end
    end

    function salva_resumo!(arquivo; f_final, RMSD, x_final, status, avaliacoes, tempo_s, extra = "")
        open(arquivo, "a") do file
            write(file, "\nResumo final\n")
            write(file, "status = $status\n")
            write(file, "f_final = $f_final\n")
            write(file, "RMSD = $RMSD\n")
            write(file, "avaliacoes = $avaliacoes\n")
            write(file, "tempo_s = $(round(tempo_s, digits=6))\n")
            write(file, "x_final = $(collect(x_final))\n")
            if !isempty(extra)
                write(file, extra)
            end
        end
    end

    function monta_resultado(metodo, resultado, x_final, f_final, arquivo; status = "", avaliacoes = missing, tempo_s = missing)
        estado_final = fun(x_final, tbeg, tend, estado_prev)
        RMSD_final = norm(estado_final.erro / sqrt(size(estado_final.erro, 1)))
        println("fval $(uppercase(String(metodo))) = ", f_final)
        println("RMSD após $(uppercase(String(metodo))) = ", RMSD_final)
        println("Avaliações $(uppercase(String(metodo))) = ", avaliacoes)
        println("Tempo $(uppercase(String(metodo))) = ", round(tempo_s, digits=6), " s")
        return (;
            metodo,
            tbeg,
            tend,
            dim,
            fval = f_final,
            RMSD = RMSD_final,
            X = copy(x_final),
            estado = estado_final,
            result = resultado,
            status,
            avaliacoes,
            tempo_s,
            arquivo,
        )
    end

    for metodo in metodos
        arquivo = inicializa_arquivo(metodo)

        if metodo == :bfgs
            println("Rodando BFGS com dimensão = ", dim)
            iter_bfgs = Ref(0)

            function callback_bfgs(estado)
                registra_avaliacao!(arquivo, iter_bfgs[], estado.f_x, estado.x)
                iter_bfgs[] += 1
                return false
            end

            inicio_metodo = time()
            resultado_bfgs = Optim.optimize(
                    f_penalizada_bfgs,
                    g_obj!,
                    X_ref,
                    BFGS(),
                    Optim.Options(
                        iterations = iterations,
                        f_calls_limit = f_calls_limit,
                        g_calls_limit = g_calls_limit,
                        callback = callback_bfgs,
                    ),
                )
            tempo_bfgs = time() - inicio_metodo

            X_bfgs = clamp.(Optim.minimizer(resultado_bfgs), lower_bfgs, upper_bfgs)
            fval_bfgs = Optim.minimum(resultado_bfgs)
            avaliacoes_bfgs = Optim.f_calls(resultado_bfgs)
            resultado = monta_resultado(:bfgs, resultado_bfgs, X_bfgs, fval_bfgs, arquivo; status = Optim.converged(resultado_bfgs), avaliacoes = avaliacoes_bfgs, tempo_s = tempo_bfgs)
            salva_resumo!(arquivo; f_final = fval_bfgs, RMSD = resultado.RMSD, x_final = X_bfgs, status = Optim.converged(resultado_bfgs), avaliacoes = avaliacoes_bfgs, tempo_s = tempo_bfgs, extra = "iteracoes = $(Optim.iterations(resultado_bfgs))\ngradientes = $(Optim.g_calls(resultado_bfgs))\n")
            resultados[:bfgs] = resultado

        elseif metodo == :spg
            println("Rodando SPGBox com dimensão = ", dim)
            spgbox = getproperty(Base.require(Base.PkgId(Base.UUID("bf97046b-3e66-4aa0-9aed-26efb7fac769"), "SPGBox")), :spgbox)
            iter_spg = Ref(0)

            function callback_spg(resultado_spg)
                registra_avaliacao!(arquivo, iter_spg[], resultado_spg.f, resultado_spg.x)
                iter_spg[] += 1
                return false
            end

            inicio_metodo = time()
            resultado_spg = Base.invokelatest(
                spgbox,
                f_penalizada_bfgs,
                g_obj!,
                X_ref;
                lower = lower_bfgs,
                upper = upper_bfgs,
                nitmax = iterations,
                nfevalmax = f_calls_limit,
                callback = callback_spg,
            )
            tempo_spg = time() - inicio_metodo

            X_spg = copy(resultado_spg.x)
            fval_spg = resultado_spg.f
            avaliacoes_spg = resultado_spg.nfeval
            resultado = monta_resultado(:spg, resultado_spg, X_spg, fval_spg, arquivo; status = resultado_spg.ierr, avaliacoes = avaliacoes_spg, tempo_s = tempo_spg)
            salva_resumo!(arquivo; f_final = fval_spg, RMSD = resultado.RMSD, x_final = X_spg, status = resultado_spg.ierr, avaliacoes = avaliacoes_spg, tempo_s = tempo_spg, extra = "iteracoes = $(resultado_spg.nit)\ngnorm = $(resultado_spg.gnorm)\n")
            resultados[:spg] = resultado

        elseif metodo == :bobyqa
            println("Rodando BOBYQA com dimensão = ", dim)
            iter_bobyqa = Ref(0)

            function objetivo_bobyqa(x, grad)
                valor = Float64(f_penalizada_bfgs(x))
                registra_avaliacao!(arquivo, iter_bobyqa[], valor, x)
                iter_bobyqa[] += 1
                return valor
            end

            opt = Opt(:LN_BOBYQA, dim)
            lower_bounds!(opt, lower_bfgs)
            upper_bounds!(opt, upper_bfgs)
            maxeval!(opt, f_calls_limit)
            min_objective!(opt, objetivo_bobyqa)

            inicio_metodo = time()
            fval_bobyqa, X_bobyqa, status_bobyqa = NLopt.optimize(opt, X_ref)
            tempo_bobyqa = time() - inicio_metodo
            avaliacoes_bobyqa = iter_bobyqa[]
            resultado = monta_resultado(:bobyqa, status_bobyqa, X_bobyqa, fval_bobyqa, arquivo; status = status_bobyqa, avaliacoes = avaliacoes_bobyqa, tempo_s = tempo_bobyqa)
            salva_resumo!(arquivo; f_final = fval_bobyqa, RMSD = resultado.RMSD, x_final = X_bobyqa, status = status_bobyqa, avaliacoes = avaliacoes_bobyqa, tempo_s = tempo_bobyqa)
            resultados[:bobyqa] = resultado

        elseif metodo == :mads
            println("Rodando MADS/NOMAD com dimensão = ", dim)
            pontos_mads = Vector{Vector{Float64}}()
            valores_mads = Float64[]

            function objetivo_mads(x)
                valor = Float64(f_penalizada_bfgs(x))
                x_atual = copy(float.(x))
                push!(pontos_mads, x_atual)
                push!(valores_mads, valor)
                registra_avaliacao!(arquivo, length(pontos_mads) - 1, valor, x_atual)
                return (true, true, [valor])
            end

            opcoes = NOMAD.NomadOptions(
                display_degree = 0,
                max_bb_eval = f_calls_limit,
            )

            problema = NOMAD.NomadProblem(
                dim,
                1,
                ["OBJ"],
                objetivo_mads,
                input_types = fill("R", dim),
                lower_bound = lower_bfgs,
                upper_bound = upper_bfgs,
                min_mesh_size = fill(Float64(rhoend), dim),
                initial_mesh_size = fill(Float64(rhobeg), dim),
                options = opcoes,
            )

            inicio_metodo = time()
            resultado_mads = NOMAD.solve(problema, X_ref)
            tempo_mads = time() - inicio_metodo
            melhor_indice = argmin(valores_mads)
            X_mads = get(resultado_mads, :x_sol, pontos_mads[melhor_indice])
            fval_mads = Float64(f_penalizada_bfgs(X_mads))
            avaliacoes_mads = length(pontos_mads)
            resultado = monta_resultado(:mads, resultado_mads, X_mads, fval_mads, arquivo; status = resultado_mads.status, avaliacoes = avaliacoes_mads, tempo_s = tempo_mads)
            salva_resumo!(arquivo; f_final = fval_mads, RMSD = resultado.RMSD, x_final = X_mads, status = resultado_mads.status, avaliacoes = avaliacoes_mads, tempo_s = tempo_mads, extra = "factivel = $(resultado_mads.feasible)\n")
            resultados[:mads] = resultado

        else
            error("Metodo desconhecido: $metodo. Use bfgs, spg, bobyqa, mads ou all.")
        end
    end

    return resultados

end


function full_dim_problem(;
    x0 = collect(range(0.08, 0.08, length = 2)),
    tmax = 15.0,
    lb = 0.0,
    ub = 0.5,
    iterations = 10,
    penalty_weight = 1.0e6,
    f_calls_limit = 50,
    g_calls_limit = 20,
    arquivo_aceitos = joinpath(first_order_results_dir_for_tmax(tmax), "full_dim_problem_pontos_aceitos.txt"),
    arquivo_avaliacoes = joinpath(first_order_results_dir_for_tmax(tmax), "full_dim_problem_avaliacoes.txt"),
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
        write(file, "Metodo = Fminbox(BFGS()) com caixa\n")
        write(file, "tmax = $tmax\n")
        write(file, "x0 = $(collect(X))\n")
        write(file, "lb = $lb | ub = $ub\n")
        write(file, "iterations = $iterations\n")
        write(file, "\niter\tf\tgnorm\t$colunas_x\n")
    end

    x_avaliados = Vector{Vector{Float64}}()
    f_avaliados = Float64[]
    x_gradientes = Vector{Vector{Float64}}()
    gnorm_gradientes = Float64[]

    function f_obj(x_vars)
        valor = quad_fun(x_vars; tmax=tmax)
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

    metodo_bfgs = BFGS()
    
    resultado_bfgs = Optim.optimize(
        f_penalizada,
        g_obj!,
        X,
        metodo_bfgs,
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
    x0 = [0.2; 0.06],
    tmax = 5.0,
    lb = 0.0,
    ub = 0.5,
    iterations = 10,
    penalty_weight = 1.0e6,
    f_calls_limit = 50,
    g_calls_limit = 20,
    arquivo_aceitos = joinpath(first_order_results_dir_for_tmax(tmax), "BFGS_two_dim_problem_pontos_aceitos.txt"),
    arquivo_avaliacoes = joinpath(first_order_results_dir_for_tmax(tmax), "BFGS_two_dim_problem_avaliacoes.txt"),
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
        write(file, "tmax = $tmax\n")
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
        valor = quad_fun(x_vars; tmax=tmax)
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



function BFGS_BOBYQA_full_dim_problem(;
    x0 = collect(range(0.2, 0.06, length = 10)),
    tmax = 5.0,
    lb = 0.0,
    ub = 0.5,
    iterations = 10,
    penalty_weight = 1.0e6,
    f_calls_limit = 50,
    g_calls_limit = 20,
    rhobeg = 0.05,
    rhoend = 0.001,
    maxeval = 500,
    arquivo_aceitos = joinpath(hybrid_results_dir_for_tmax(tmax), "BFGS_BOBYQA_full_dim_problem_pontos_aceitos.txt"),
    arquivo_avaliacoes = joinpath(hybrid_results_dir_for_tmax(tmax), "BFGS_BOBYQA_full_dim_problem_avaliacoes.txt"),
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
    npt = 2 * n + 1

    open(nome, "w") do file
        write(file, "Avaliacoes normais de quad_fun no BFGS_BOBYQA_full_dim_problem\n")
        write(file, "tempo_s\tn_calfun\tRMSD\terro_quadratico_total\n")
    end

    open(arquivo_pontos, "w") do file
        write(file, "Pontos aceitos no BFGS_BOBYQA_full_dim_problem\n")
        write(file, "Metodo = BFGS seguido de BOBYQA com caixa\n")
        write(file, "tmax = $tmax\n")
        write(file, "x0 = $(collect(X))\n")
        write(file, "lb = $lb | ub = $ub\n")
        write(file, "iterations = $iterations | penalty_weight = $penalty_weight | rhobeg = $rhobeg | rhoend = $rhoend | maxeval = $maxeval\n")
        write(file, "\niter\tf\tgnorm\t$colunas_x\n")
    end

    x_avaliados = Vector{Vector{Float64}}()
    f_avaliados = Float64[]
    x_gradientes = Vector{Vector{Float64}}()
    gnorm_gradientes = Float64[]

    function f_obj(x_vars)
        valor = quad_fun(x_vars; tmax=tmax)
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
        write(file, "\nResumo BFGS\n")
        write(file, "convergiu_bfgs = $(Optim.converged(resultado_bfgs))\n")
        write(file, "iteracoes_bfgs = $(Optim.iterations(resultado_bfgs))\n")
        write(file, "f_final_bfgs = $f_final\n")
        write(file, "f_final_penalizado_bfgs = $f_final_penalizado\n")
        write(file, "x_final_bfgs = $(collect(x_final))\n")
        write(file, "n_pontos_aceitos_bfgs = $(length(pontos_aceitos))\n")
        write(file, "\nEtapa BOBYQA\n")
        write(file, "x_inicial_bobyqa = $(collect(x_final))\n")
        write(file, "\niter_bobyqa\tf\t$colunas_x\n")
    end

    open(nome, "a") do file
        write(file, "\n# Etapa BOBYQA\n")
    end

    pontos_bobyqa = Vector{Vector{Float64}}()
    valores_bobyqa = Float64[]
    objetivo_bobyqa(x, grad) = begin
        valor = quad_fun(x; tmax=tmax)
        x_atual = copy(float.(x))
        push!(pontos_bobyqa, x_atual)
        push!(valores_bobyqa, Float64(valor))
        open(arquivo_pontos, "a") do file
            iter = length(pontos_bobyqa) - 1
            write(file, "$iter\t$valor\t$(join(x_atual, "\t"))\n")
        end
        return valor
    end

    opt = Opt(:LN_BOBYQA, n)
    opt.lower_bounds = limites_inferiores
    opt.upper_bounds = limites_superiores
    opt.initial_step = fill(Float64(rhobeg), n)
    opt.xtol_abs = fill(Float64(rhoend), n)
    opt.maxeval = maxeval
    opt.population = npt
    opt.min_objective = objetivo_bobyqa

    n_calfun_inicio_bobyqa = n_calfun
    tempo_inicio_bobyqa = time()
    f_bobyqa, x_bobyqa, status_bobyqa = NLopt.optimize(opt, x_final)
    tempo_bobyqa = time() - tempo_inicio_bobyqa
    avaliacoes_bobyqa = n_calfun - n_calfun_inicio_bobyqa
    avaliacoes_nlopt = opt.numevals

    open(arquivo_pontos, "a") do file
        write(file, "\nResumo BOBYQA\n")
        write(file, "status_bobyqa = $status_bobyqa\n")
        write(file, "f_final_bobyqa = $f_bobyqa\n")
        write(file, "x_final_bobyqa = $(collect(x_bobyqa))\n")
        write(file, "avaliacoes_bobyqa = $avaliacoes_bobyqa\n")
        write(file, "avaliacoes_nlopt_bobyqa = $avaliacoes_nlopt\n")
        write(file, "tempo_bobyqa_s = $(round(tempo_bobyqa, digits=3))\n")
    end

    return (;
        resultado_bfgs = resultado_bfgs,
        x_bfgs = x_final,
        f_bfgs = f_final,
        x_final = x_bobyqa,
        f_final = f_bobyqa,
        pontos_aceitos_bfgs = pontos_aceitos,
        pontos_aceitos_bobyqa = pontos_bobyqa,
        valores_aceitos_bfgs = valores_aceitos,
        valores_aceitos_bobyqa = valores_bobyqa,
        gnorms_aceitos_bfgs = gnorms_aceitos,
        arquivo_pontos,
        arquivo_avaliacoes = nome,
        status_bobyqa = status_bobyqa,
        avaliacoes_bobyqa = avaliacoes_bobyqa,
        avaliacoes_nlopt_bobyqa = avaliacoes_nlopt,
        tempo_bobyqa_s = tempo_bobyqa,
    )
end

function BFGS_BOBYQA_two_dim_problem(;
    x0 = fill(0.08, 10),
    tmax = 5.0,
    bobyqa_dim = 10,
    lb = 0.0,
    ub = 0.5,
    iterations = 10,
    penalty_weight = 1.0e6,
    f_calls_limit = 50,
    g_calls_limit = 20,
    rhobeg = 0.05,
    rhoend = 0.001,
    maxeval = 500,
    arquivo_aceitos = joinpath(hybrid_results_dir_for_tmax(tmax), "BFGS_BOBYQA_two_dim_problem_pontos_aceitos.txt"),
    arquivo_avaliacoes = joinpath(hybrid_results_dir_for_tmax(tmax), "BFGS_BOBYQA_two_dim_problem_avaliacoes.txt"),
    )
    n_bfgs = length(x0)
    X = copy(float.(x0))
    limites_inferiores_bfgs = fill(Float64(lb), n_bfgs)
    limites_superiores_bfgs = fill(Float64(ub), n_bfgs)
    arquivo_pontos = normpath(arquivo_aceitos)
    mkpath(dirname(arquivo_pontos))

    global n_calfun = 0
    global inicio_s = time()
    global nome = normpath(arquivo_avaliacoes)
    mkpath(dirname(nome))
    colunas_x_bfgs = join(("x$i" for i in 1:n_bfgs), "\t")
    colunas_x_bobyqa = join(("x$i" for i in 1:bobyqa_dim), "\t")
    npt_bobyqa = 2 * bobyqa_dim + 1

    open(nome, "w") do file
        write(file, "Avaliacoes normais de quad_fun no BFGS_BOBYQA_two_dim_problem\n")
        write(file, "tempo_s\tn_calfun\tRMSD\terro_quadratico_total\n")
    end

    open(arquivo_pontos, "w") do file
        write(file, "Pontos aceitos no BFGS_BOBYQA_two_dim_problem\n")
        write(file, "Metodo = BFGS seguido de BOBYQA com caixa\n")
        write(file, "tmax = $tmax\n")
        write(file, "x0 = $(collect(X))\n")
        write(file, "dim_bfgs = $n_bfgs | dim_bobyqa = $bobyqa_dim\n")
        write(file, "lb = $lb | ub = $ub\n")
        write(file, "iterations = $iterations | penalty_weight = $penalty_weight | rhobeg = $rhobeg | rhoend = $rhoend | maxeval = $maxeval\n")
        write(file, "\niter\tf\tgnorm\t$colunas_x_bfgs\n")
    end

    x_avaliados = Vector{Vector{Float64}}()
    f_avaliados = Float64[]
    x_gradientes = Vector{Vector{Float64}}()
    gnorm_gradientes = Float64[]

    function f_obj(x_vars)
        valor = quad_fun(x_vars; tmax=tmax)
        if !(eltype(x_vars) <: ForwardDiff.Dual)
            push!(x_avaliados, copy(float.(x_vars)))
            push!(f_avaliados, Float64(valor))
        end
        return valor
    end
    f_penalizada(x_vars) = f_obj(x_vars) +
                           penalidade_caixa_externa(
                               x_vars,
                               limites_inferiores_bfgs,
                               limites_superiores_bfgs;
                               rho=penalty_weight,
                           )

    cfg_bfgs = ForwardDiff.GradientConfig(f_penalizada, X, ForwardDiff.Chunk{n_bfgs}())
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

    x_inicial_bobyqa = expande_dimensao(x_final, bobyqa_dim)
    limites_inferiores_bobyqa = fill(Float64(lb), bobyqa_dim)
    limites_superiores_bobyqa = fill(Float64(ub), bobyqa_dim)

    open(arquivo_pontos, "a") do file
        write(file, "\nResumo BFGS\n")
        write(file, "convergiu_bfgs = $(Optim.converged(resultado_bfgs))\n")
        write(file, "iteracoes_bfgs = $(Optim.iterations(resultado_bfgs))\n")
        write(file, "f_final_bfgs = $f_final\n")
        write(file, "f_final_penalizado_bfgs = $f_final_penalizado\n")
        write(file, "x_final_bfgs = $(collect(x_final))\n")
        write(file, "n_pontos_aceitos_bfgs = $(length(pontos_aceitos))\n")
        write(file, "\nEtapa BOBYQA\n")
        write(file, "x_inicial_bobyqa = $(collect(x_inicial_bobyqa))\n")
        write(file, "\niter_bobyqa\tf\t$colunas_x_bobyqa\n")
    end

    open(nome, "a") do file
        write(file, "\n# Etapa BOBYQA\n")
    end

    pontos_bobyqa = Vector{Vector{Float64}}()
    valores_bobyqa = Float64[]
    objetivo_bobyqa(x, grad) = begin
        valor = quad_fun(x; tmax=tmax)
        x_atual = copy(float.(x))
        push!(pontos_bobyqa, x_atual)
        push!(valores_bobyqa, Float64(valor))
        open(arquivo_pontos, "a") do file
            iter = length(pontos_bobyqa) - 1
            write(file, "$iter\t$valor\t$(join(x_atual, "\t"))\n")
        end
        return valor
    end

    opt = Opt(:LN_BOBYQA, bobyqa_dim)
    opt.lower_bounds = limites_inferiores_bobyqa
    opt.upper_bounds = limites_superiores_bobyqa
    opt.initial_step = fill(Float64(rhobeg), bobyqa_dim)
    opt.xtol_abs = fill(Float64(rhoend), bobyqa_dim)
    opt.maxeval = maxeval
    opt.population = npt_bobyqa
    opt.min_objective = objetivo_bobyqa

    n_calfun_inicio_bobyqa = n_calfun
    tempo_inicio_bobyqa = time()
    f_bobyqa, x_bobyqa, status_bobyqa = NLopt.optimize(opt, x_inicial_bobyqa)
    tempo_bobyqa = time() - tempo_inicio_bobyqa
    avaliacoes_bobyqa = n_calfun - n_calfun_inicio_bobyqa
    avaliacoes_nlopt = opt.numevals

    open(arquivo_pontos, "a") do file
        write(file, "\nResumo BOBYQA\n")
        write(file, "status_bobyqa = $status_bobyqa\n")
        write(file, "f_final_bobyqa = $f_bobyqa\n")
        write(file, "x_final_bobyqa = $(collect(x_bobyqa))\n")
        write(file, "avaliacoes_bobyqa = $avaliacoes_bobyqa\n")
        write(file, "avaliacoes_nlopt_bobyqa = $avaliacoes_nlopt\n")
        write(file, "tempo_bobyqa_s = $(round(tempo_bobyqa, digits=3))\n")
    end

    return (;
        resultado_bfgs = resultado_bfgs,
        x_bfgs = x_final,
        f_bfgs = f_final,
        x_inicial_bobyqa = x_inicial_bobyqa,
        x_final = x_bobyqa,
        f_final = f_bobyqa,
        pontos_aceitos_bfgs = pontos_aceitos,
        pontos_aceitos_bobyqa = pontos_bobyqa,
        valores_aceitos_bfgs = valores_aceitos,
        valores_aceitos_bobyqa = valores_bobyqa,
        gnorms_aceitos_bfgs = gnorms_aceitos,
        arquivo_pontos,
        arquivo_avaliacoes = nome,
        status_bobyqa = status_bobyqa,
        avaliacoes_bobyqa = avaliacoes_bobyqa,
        avaliacoes_nlopt_bobyqa = avaliacoes_nlopt,
        tempo_bobyqa_s = tempo_bobyqa,
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
