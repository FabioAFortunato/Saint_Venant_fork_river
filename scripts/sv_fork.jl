#git init # inicia o repositório local
#git add <arquivos> <ou * para tudo.. use com responsabilidade>
#git commit -m "comentario"
#git pull # puchar alterações do repositório remoto
#git push # enviar alterações para o repositório remoto
#git remote add <link> # adiciona e configura um repositório remoto

using DelimitedFiles
using ForwardDiff
using Interpolations
using LinearAlgebra
using NLopt
using Optim
using Random
using PGFPlotsX

include("../data/processed/dado_fork.jl")
include("../src/BFGS.jl")

const fake = 0 # if fake = 1 then we use pregenerated data
const nx = 101 # grid number
const tmin = 0.0 # seconds
const tmax1 = 0.0 + 60.0 * 0.0 + 60.0 * 60.0 * 0.0 + 60.0 * 60.0 * 24.0 * 31.0
const timprim = 0.0 + 60.0 * 0.0 + 60.0 * 60.0 * 4.0 + 60.0 * 60.0 * 24.0 * 0.0
const nt = round(Int32, 1 + (tmax1 - tmin - 60.0 * 60.0 * 24.0 * 3.0) / timprim)
const n_man = 101
const RESULTS_DIR = normpath(joinpath(@__DIR__, "..", "results"))

if fake == 1
    dados_fake = readdlm("dado_fake.txt", ' ')
    tfake = dados_fake[:, 1] * (24 * 60 * 60)
    z1fake = dados_fake[:, 2]
    z2fake = dados_fake[:, 3]

    zfinal_fake = interpolate((tfake,), z1fake, Gridded(Linear()))
    zfinal_fake = extrapolate(zfinal_fake, Line())

    zmeio_fake = interpolate((tfake,), z2fake, Gridded(Linear()))
    zmeio_fake = extrapolate(zmeio_fake, Line())
end

n_calfun = 0
inicio_s = time()


function arquivo_em_results(arquivo)
    mkpath(RESULTS_DIR)
    return joinpath(RESULTS_DIR, basename(String(arquivo)))
end

nome = arquivo_em_results("teste_bfgs_bobyqa_L_sv_5.txt")

function label_tmax(tmax)
    tmax_float = Float64(tmax)
    return isinteger(tmax_float) ? string(Int(round(tmax_float))) : replace(string(tmax_float), "." => "_")
end

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
        arquivo_saida === nothing ? "teste_bfgs_$(label_tmax(tmax)).txt" : arquivo_saida,
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
    X = clamp.(copy(float.(x)), lb, ub)
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

        X = clamp.(Optim.minimizer(resultado_bfgs), lb, ub)
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

const LA_method2 = minimizar_L_sv_bfgs

function minimizar_L_sv_bfgs_bobyqa(
    x0::AbstractVector{T}=fill(0.08, n_man + 1);
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
        arquivo_saida === nothing ? "teste_bfgs_bobyqa_L_sv_$(label_tmax(tmax)).txt" : arquivo_saida,
    )

    x_bfgs = minimizar_L_sv_bfgs(
        copy(x0);
        tmax=tmax,
        arquivo_saida=arquivo,
        rotulo_execucao="BFGS do teste BFGS + BOBYQA",
        n_restart=1,
    )
    x_inicial_bobyqa = clamp.(Float64.(x_bfgs), lb, ub)

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

    x_final = clamp.(x_bobyqa, lb, ub)
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

function testar_L_sv_bfgs_bobyqa_tmax(
    x0::AbstractVector{T}=fill(0.08, n_man + 1);
    tmax_values=(5.0, 15.0),
    arquivo_prefixo="teste_bfgs_bobyqa_L_sv",
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

# ng = fill(0.08, n_man + 1)
# run_L_sv_bfgs_bobyqa_tmax = true
# result = run_L_sv_bfgs_bobyqa_tmax ? testar_L_sv_bfgs_bobyqa_tmax(ng) : ng
