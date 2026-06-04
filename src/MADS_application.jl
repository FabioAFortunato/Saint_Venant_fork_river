using NOMAD

const DERIVATIVE_FREE_RESULTS_DIR = normpath(joinpath(@__DIR__, "..", "results", "derivative_free"))

function derivative_free_results_dir_for_tmax(tmax)
    tmax_float = Float64(tmax)
    pasta = isinteger(tmax_float) ? "$(Int(round(tmax_float)))_days" : "$(label_tmax(tmax_float))_days"
    return joinpath(DERIVATIVE_FREE_RESULTS_DIR, pasta)
end

function executar_MADS_problem(;
    x0,
    tmax,
    lb,
    ub,
    rhobeg,
    rhoend,
    maxeval,
    arquivo_aceitos,
    arquivo_avaliacoes,
    problem_label,
)
    n = length(x0)
    X = copy(float.(x0))
    limites_inferiores = fill(Float64(lb), n)
    limites_superiores = fill(Float64(ub), n)
    arquivo_pontos = normpath(arquivo_aceitos)
    nome_avaliacoes = normpath(arquivo_avaliacoes)
    mkpath(dirname(arquivo_pontos))
    mkpath(dirname(nome_avaliacoes))
    colunas_x = join(("x$i" for i in 1:n), "\t")

    global n_calfun = 0
    global inicio_s = time()
    global nome = nome_avaliacoes

    open(nome, "w") do file
        write(file, "Avaliacoes normais de quad_fun no $problem_label\n")
        write(file, "tempo_s\tn_calfun\tRMSD\terro_quadratico_total\n")
    end

    open(arquivo_pontos, "w") do file
        write(file, "Pontos avaliados pelo MADS no $problem_label\n")
        write(file, "Metodo = NOMAD MADS\n")
        write(file, "tmax = $tmax\n")
        write(file, "x0 = $(collect(X))\n")
        write(file, "lb = $lb | ub = $ub\n")
        write(file, "rhobeg = $rhobeg | rhoend = $rhoend | maxeval = $maxeval\n")
        write(file, "\niter\tf\t$colunas_x\n")
    end

    pontos_avaliados = Vector{Vector{Float64}}()
    valores_avaliados = Float64[]

    function objetivo_mads(x)
        valor = quad_fun(x; tmax=tmax)

        x_atual = copy(float.(x))
        push!(pontos_avaliados, x_atual)
        push!(valores_avaliados, Float64(valor))

        open(arquivo_pontos, "a") do file
            iter = length(pontos_avaliados) - 1
            write(file, "$iter\t$valor\t$(join(x_atual, "\t"))\n")
        end

        return (true, true, [Float64(valor)])
    end

    opcoes = NOMAD.NomadOptions(
        display_degree = 0,
        max_bb_eval = maxeval,
    )

    problema = NOMAD.NomadProblem(
        n,
        1,
        ["OBJ"],
        objetivo_mads,
        input_types = fill("R", n),
        lower_bound = limites_inferiores,
        upper_bound = limites_superiores,
        min_mesh_size = fill(Float64(rhoend), n),
        initial_mesh_size = fill(Float64(rhobeg), n),
        options = opcoes,
    )

    resultado = NOMAD.solve(problema, X)
    x_final = resultado.x_sol
    f_final = resultado.bbo_sol[1]
    status_mads = resultado.status
    factivel = resultado.feasible

    open(arquivo_pontos, "a") do file
        write(file, "\nResumo final\n")
        write(file, "status_mads = $status_mads\n")
        write(file, "solucao_factivel = $factivel\n")
        write(file, "f_final = $f_final\n")
        write(file, "x_final = $(collect(x_final))\n")
        write(file, "avaliacoes_mads = $(length(pontos_avaliados))\n")
    end

    return (;
        x_final,
        f_final,
        pontos_aceitos = pontos_avaliados,
        valores_aceitos = valores_avaliados,
        arquivo_pontos,
        arquivo_avaliacoes = nome,
        status = status_mads,
        factivel = factivel,
        avaliacoes_mads = length(pontos_avaliados),
    )
end

function mads_full_dim_problem(;
    x0 = collect(range(0.1, 0.2, length = 101)),
    tmax = 5.0,
    lb = 0.0,
    ub = 0.5,
    rhobeg = 0.05,
    rhoend = 0.001,
    maxeval = 500,
    arquivo_aceitos = joinpath(derivative_free_results_dir_for_tmax(tmax), "MADS_full_dim_problem_pontos_aceitos.txt"),
    arquivo_avaliacoes = joinpath(derivative_free_results_dir_for_tmax(tmax), "MADS_full_dim_problem_avaliacoes.txt"),
)
    return executar_MADS_problem(
        x0 = x0,
        tmax = tmax,
        lb = lb,
        ub = ub,
        rhobeg = rhobeg,
        rhoend = rhoend,
        maxeval = maxeval,
        arquivo_aceitos = arquivo_aceitos,
        arquivo_avaliacoes = arquivo_avaliacoes,
        problem_label = "full_dim_problem",
    )
end

function mads_two_dim_problem(;
    x0 = [0.1; 0.20],
    tmax = 5.0,
    lb = 0.0,
    ub = 0.5,
    rhobeg = 0.05,
    rhoend = 0.001,
    maxeval = 500,
    arquivo_aceitos = joinpath(derivative_free_results_dir_for_tmax(tmax), "MADS_two_dim_problem_pontos_aceitos.txt"),
    arquivo_avaliacoes = joinpath(derivative_free_results_dir_for_tmax(tmax), "MADS_two_dim_problem_avaliacoes.txt"),
)
    return executar_MADS_problem(
        x0 = x0,
        tmax = tmax,
        lb = lb,
        ub = ub,
        rhobeg = rhobeg,
        rhoend = rhoend,
        maxeval = maxeval,
        arquivo_aceitos = arquivo_aceitos,
        arquivo_avaliacoes = arquivo_avaliacoes,
        problem_label = "two_dim_problem",
    )
end
