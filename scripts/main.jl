using DataFrames
using Printf
using GLM
using LaTeXStrings
n_man = 101

include("../src/obj_func.jl")
include("../src/BFGS_BOBYQA.jl")
include("../src/BOBYQA_application.jl")
include("../src/MADS_application.jl")
include("../src/assimilacao.jl")
include("../src/solve_sv_beta.jl")

struct struct_Result
    method::String
    dim::Int
    tmax::Float64
    RMSD::Float64
    gnorm::Float64
    evaluation::Int
    time::Float64
    x_final::Vector
end

# X_ref = fill(0.09, 101)

# tempo = time()
# jac = sv_fork(X_ref, 31.0)
# tempo = time() - tempo
# print("$tempo \n")

# aux_derivada(x) = sv_fork(x, 31.0)


# n_aux = length(X_ref)

# cfg_bfgs = ForwardDiff.JacobianConfig(
#     aux_derivada,
#     X_ref,
#     ForwardDiff.Chunk{n_aux}()
# )

# tempo = time()

# J = ForwardDiff.jacobian(aux_derivada, X_ref, cfg_bfgs)

# tempo = time() - tempo

# jac = jac - J 
# norma_J = norm(jac)

# println("$tempo \n")
# println("norma da Jacobiana = $norma_J")





function teste_derivada_tempo()

    df = DataFrame(
        dim = Int[],
        tempo_derivada = Float64[],
        tempo_fun = Float64[],
        razao_derivada_fun = Float64[]
    )

    for i in 1:101
        X0, t_d, t_f = run_assimilation(
            X0 = fill(0.09, i),
            tins = 31.0
        )

        raz = t_d / t_f

        push!(df, (
            dim = i,
            tempo_derivada = t_d,
            tempo_fun = t_f,
            razao_derivada_fun = raz
        ))
    end

    modelo = lm(@formula(tempo_derivada ~ dim), df)
    b, a = coef(modelo)

    df_plot = sort(df, :dim)

    x_plot = df_plot.dim
    y_plot = df_plot.tempo_derivada

    y_line = predict(modelo, DataFrame(dim = x_plot))

    p = plot(
        x_plot,
        y_line,
        label = "tempo_derivada ≈ $a + $b * dim",
        xlabel = "Dimension",
        ylabel = "Time (s)"
    )

    scatter!(
        p,
        x_plot,
        y_plot,
        label = "Observed times"
    )
    savefig("results/derivada.png")

    return df, p
end

function teste_todos_novo(;
    tins = [5.0, 15.0, 31.0],
    dim_init = 1,
    dims = 3,
    filename = "results_new_SV.tex",
    method = "all",
    )
    todos_resultados = struct_Result[]
    for t in tins
        for i = dim_init:dims
            res_todos = run_problems(
                X0 = fill(0.09, i),
                tins = t,
                method = method,
            )
            for res in res_todos
                push!(todos_resultados, 
                        struct_Result(
                            String(res[2].metodo), 
                            res[2].dim, 
                            res[2].tend,
                            res[2].RMSD,
                            res[2].gnorm,
                            res[2].avaliacoes,
                            res[2].tempo_s,
                            res[2].X))
            end
        end
    end

    df = DataFrame(
        Method = [r.method for r in todos_resultados],
        tmax = [r.tmax for r in todos_resultados],
        n_grid = [r.dim for r in todos_resultados],
        RMSD = [r.RMSD for r in todos_resultados],
        GradNorm = [r.gnorm for r in todos_resultados],
        Time = [r.time for r in todos_resultados],
        Evaluations = [r.evaluation for r in todos_resultados],
    )


    open(filename, "w") do io

        println(io, raw"\begin{table}[htbp]")
        println(io, raw"\centering")
        println(io, raw"\caption{Results obtained by the solvers for the smoothed Saint-Venant model.}")
        println(io, raw"\label{tab:methods_new_SV}")
        println(io, raw"\begin{tabular}{c|c|ccccc}")
        println(io, raw"\hline")
        println(io, raw"Method & $t_{\max}$ & $n_{grid}$ & RMSD & $\|\nabla f\|$ & Time (s) & Evaluations \\ \hline")


        methods = unique(df.Method)

        for method in methods
            df_method = df[df.Method .== method, :]
            tmax_values = unique(df_method.tmax)

            first_method_row = true

            for (k, tmax) in enumerate(tmax_values)
                df_t = df_method[df_method.tmax .== tmax, :]
                sort!(df_t, :n_grid)
                n_linhas_tmax = nrow(df_t)

                for (j, row) in enumerate(eachrow(df_t))
                    method_entry = first_method_row && j == 1 ? method : ""
                    tmax_entry = j == 1 ? "\\multirow{$n_linhas_tmax}{*}{$(Int(tmax))}" : ""

                    rmsd_str = @sprintf("%.4f", row.RMSD)
                    gnorm_str = @sprintf("%.4e", row.GradNorm)
                    time_str = @sprintf("%.2f", row.Time)

                    println(io,
                        "$(method_entry) & $(tmax_entry) & $(row.n_grid) & " *
                        "$(rmsd_str) & $(gnorm_str) & $(time_str) & $(row.Evaluations) \\\\"
                    )
                end

                first_method_row = false

                if k < length(tmax_values)
                    println(io, raw"\cline{2-7}")
                end
            end

            println(io, raw"\hline")
        end

        println(io, raw"\end{tabular}")
        println(io, raw"\end{table}")
    end


end

function teste_todos(; t = 5.0, x = [0.1; 0.06])

    BFGS_BOBYQA_full_dim_problem(tmax = t, x0 = collect(range(x[1], x[2], length = 101)))
    BFGS_BOBYQA_two_dim_problem(tmax = t, x0 = x)
    full_dim_problem(tmax = t, x0 = collect(range(x[1], x[2], length = 101)))
    two_dim_problem(tmax = t, x0 = x)
    bobyqa_full_dim_problem(tmax = t, x0 = collect(range(x[1], x[2], length = 101)))
    bobyqa_two_dim_problem(tmax = t, x0 = x)
    mads_full_dim_problem(tmax = t, x0 = collect(range(x[1], x[2], length = 101)))
    mads_two_dim_problem(tmax = t, x0 = x)
end

function plot_busca_exaustiva_derivada_31_3_latex(;
    data_output = "results/busca_exaustiva_derivada_31_3.csv",
    output = "results/busca_exaustiva_derivada_31_3_latex.pdf",
    output_RMSD = "results/busca_exaustiva_derivada_31_3_RMSD_latex.pdf",
    )
    dados = le_busca_exaustiva_derivada(data_output)
    fmax = maximum(dados.valores_plot)
    rmsd_max = maximum(dados.valores_plot_RMSD)

    p = Plots.plot(
        dados.alphas,
        dados.valores_plot;
        xlabel = L"\alpha",
        ylabel = L"f(x + \alpha d)",
        label = "limited function",
        linewidth = 2,
        legend = :topleft,
        ylims = (0, fmax),
    )
    Plots.hline!(p, [fmax]; label = "max. value = $fmax", linestyle = :dash)

    p_RMSD = Plots.plot(
        dados.alphas,
        dados.valores_plot_RMSD;
        xlabel = L"\alpha",
        ylabel = L"\mathrm{RMSD}(x + \alpha d)",
        label = "limited RMSD",
        linewidth = 2,
        legend = :topleft,
        ylims = (0, rmsd_max),
    )
    Plots.hline!(p_RMSD, [rmsd_max]; label = "max. value = $(trunc(rmsd_max, digits=1))", linestyle = :dash)


    mkpath(dirname(output))
    Plots.savefig(p, output)

    mkpath(dirname(output_RMSD))
    Plots.savefig(p_RMSD, output_RMSD)

    return (; plot = p, plot_RMSD = p_RMSD, output, output_RMSD, data = dados)
end

function plot_assimilacao_heatmap_tend_31_latex(;
    matrix_output = "results/assimilacao_heatmap_tend_31.csv",
    output = "results/assimilacao_heatmap_tend_31_latex.pdf",
    rmsd_max = 2.0,
    tamanho = (700, 600),
)
    Plots.gr()
    heat = le_assimilation_heatmap_matrix(matrix_output)
    min_idx = argmin(heat.RMSD)
    n1_min = heat.n1[min_idx[2]]
    n2_min = heat.n2[min_idx[1]]
    rmsd_min = heat.RMSD[min_idx]
    Z_plot = map(v -> isfinite(v) && v <= rmsd_max ? v : NaN, heat.RMSD)

    p = Plots.heatmap(
        heat.n1,
        heat.n2,
        Z_plot;
        xlabel = "Manning coefficient x1",
        ylabel = "Manning coefficient x2",
        colorbar_title = L"\mathrm{RMSD}",
        clim = (0.0, rmsd_max),
        aspect_ratio = :equal,
        xlims = (minimum(heat.n1), maximum(heat.n1)),
        ylims = (minimum(heat.n2), maximum(heat.n2)),
        color = :viridis,
        background_color = :white,
        background_color_inside = :gold,
        size = tamanho,
        legend = :topleft,
    )

    Plots.scatter!(
        p,
        [NaN],
        [NaN];
        markershape = :rect,
        markersize = 14,
        markercolor = :gold,
        markerstrokecolor = :black,
        label = "Yellow region = NaN",
    )

    Plots.contour!(
        p,
        heat.n1,
        heat.n2,
        heat.RMSD;
        linewidth = 1.2,
        color = :black,
        labels = false,
        levels = 8,
    )

    Plots.scatter!(
        p,
        [n1_min],
        [n2_min];
        marker = :star5,
        markersize = 10,
        markercolor = :white,
        markerstrokecolor = :black,
        markerstrokewidth = 1.5,
        label = "($(round(n1_min, digits=4)), $(round(n2_min, digits=4))), RMSD = $(round(rmsd_min, digits=4))",
    )

    mkpath(dirname(output))
    Plots.savefig(p, output)

    return (; plot = p, output, data = heat, minimo = (n1 = n1_min, n2 = n2_min, RMSD = rmsd_min))
end

function le_pontos_aceitos_por_aumento_proximo_bfgs(txt_output = "results/BFGS_31_2.txt")
    linhas = readlines(txt_output)
    inicio = findfirst(linha -> startswith(linha, "iter"), linhas)
    if inicio === nothing
        error("Arquivo invalido: nao encontrei o cabecalho com iter/f/x.")
    end

    iteracoes = Int[]
    valores = Float64[]
    pontos = Vector{Vector{Float64}}()

    for linha in linhas[(inicio + 1):end]
        isempty(strip(linha)) && break
        startswith(linha, "Resumo") && break

        campos = split(strip(linha))
        length(campos) < 4 && continue

        push!(iteracoes, parse(Int, campos[1]))
        push!(valores, parse(Float64, campos[2]))
        push!(pontos, parse.(Float64, campos[3:end]))
    end

    indices_aceitos = [i for i in 1:(length(valores) - 1) if valores[i + 1] > valores[i]]
    pontos_aceitos = pontos[indices_aceitos]
    valores_aceitos = valores[indices_aceitos]
    iteracoes_aceitas = iteracoes[indices_aceitos]
    rmsd_resumo = missing

    for linha in linhas
        if startswith(strip(linha), "RMSD =")
            rmsd_resumo = parse(Float64, strip(split(linha, "=")[2]))
        end
    end

    return (;
        txt_output,
        iteracoes,
        valores,
        pontos,
        indices_aceitos,
        iteracoes_aceitas,
        valores_aceitos,
        pontos_aceitos,
        rmsd_resumo,
    )
end

function plot_assimilacao_heatmap_tend_31_latex_com_bfgs_aceitos(;
    bfgs_output = "results/BFGS_31_2.txt",
    output = "results/assimilacao_heatmap_tend_31_latex_bfgs_aceitos.pdf",
    kwargs...,
)
    base = plot_assimilacao_heatmap_tend_31_latex(; output, kwargs...)
    aceitos = le_pontos_aceitos_por_aumento_proximo_bfgs(bfgs_output)

    Plots.contour!(
        base.plot,
        base.data.n1,
        base.data.n2,
        base.data.RMSD;
        linewidth = 1.2,
        color = :black,
        labels = false,
        levels = 8,
    )

    if isempty(aceitos.pontos_aceitos)
        @warn "Nenhum ponto aceito inferido por f[i+1] > f[i]."
        return merge(base, (; bfgs = aceitos))
    end

    xs = [p[1] for p in aceitos.pontos_aceitos]
    ys = [p[2] for p in aceitos.pontos_aceitos]
    rmsd_bfgs = aceitos.rmsd_resumo

    if ismissing(rmsd_bfgs)
        resultado_bfgs = sv_fork_assimilation([xs[end], ys[end]], 0.0, 31.0, nothing)
        erro_bfgs = resultado_bfgs.erro
        rmsd_bfgs = norm(erro_bfgs / sqrt(length(erro_bfgs)))
    end

    Plots.plot!(
        base.plot,
        xs,
        ys;
        color = :red,
        linewidth = 2,
        marker = :circle,
        markersize = 4,
        markercolor = :red,
        markerstrokecolor = :white,
        label = "BFGS accepted points",
    )

    Plots.scatter!(
        base.plot,
        [xs[1]],
        [ys[1]];
        marker = :diamond,
        markersize = 7,
        markercolor = :white,
        markerstrokecolor = :red,
        markerstrokewidth = 1.5,
        label = "BFGS start",
    )

    Plots.scatter!(
        base.plot,
        [xs[end]],
        [ys[end]];
        marker = :star5,
        markersize = 9,
        markercolor = :red,
        markerstrokecolor = :white,
        markerstrokewidth = 1.5,
        label = "BFGS RMSD = $(round(rmsd_bfgs, digits=4))",
    )

    mkpath(dirname(output))
    Plots.savefig(base.plot, output)

    return merge(base, (; output, bfgs = aceitos, pontos_bfgs = (x = xs, y = ys), rmsd_bfgs))
end

function plot_assimilacao_heatmap_tend_31_latex_com_todos_bfgs_aceitos(;
    indices = 0:6,
    bfgs_outputs = ["results/$(i)_BFGS_31_2.txt" for i in indices],
    output = "results/assimilacao_heatmap_tend_31_latex_todos_bfgs_aceitos.pdf",
    kwargs...,
)
    base = plot_assimilacao_heatmap_tend_31_latex(; output, kwargs...)

    cores = [:red, :blue, :orange, :green, :purple, :brown]
    marcadores = [:circle, :diamond, :utriangle, :rect, :hexagon, :star5]
    rotulos_indices = collect(indices)
    testes = []

    for (j, bfgs_output) in enumerate(bfgs_outputs)
        if !isfile(bfgs_output)
            @warn "Arquivo de BFGS nao encontrado; ignorando." bfgs_output
            continue
        end

        aceitos = le_pontos_aceitos_por_aumento_proximo_bfgs(bfgs_output)
        if isempty(aceitos.pontos_aceitos)
            @warn "Nenhum ponto aceito inferido por f[i+1] > f[i]; ignorando." bfgs_output
            continue
        end

        xs = [p[1] for p in aceitos.pontos_aceitos]
        ys = [p[2] for p in aceitos.pontos_aceitos]
        rmsd_bfgs = aceitos.rmsd_resumo

        if ismissing(rmsd_bfgs)
            resultado_bfgs = sv_fork_assimilation([xs[end], ys[end]], 0.0, 31.0, nothing)
            erro_bfgs = resultado_bfgs.erro
            rmsd_bfgs = norm(erro_bfgs / sqrt(length(erro_bfgs)))
        end

        cor = cores[mod1(j, length(cores))]
        marcador = marcadores[mod1(j, length(marcadores))]
        indice_teste = j <= length(rotulos_indices) ? rotulos_indices[j] : j
        rotulo = "BFGS $(indice_teste) | RMSD = $(round(rmsd_bfgs, digits=4))"

        Plots.plot!(
            base.plot,
            xs,
            ys;
            color = cor,
            linewidth = 2,
            marker = marcador,
            markersize = 4,
            markercolor = cor,
            markerstrokecolor = :white,
            label = rotulo,
        )

        Plots.scatter!(
            base.plot,
            [xs[1]],
            [ys[1]];
            marker = :diamond,
            markersize = 6,
            markercolor = :white,
            markerstrokecolor = cor,
            markerstrokewidth = 1.5,
            label = false,
        )

        Plots.scatter!(
            base.plot,
            [xs[end]],
            [ys[end]];
            marker = :star5,
            markersize = 8,
            markercolor = cor,
            markerstrokecolor = :white,
            markerstrokewidth = 1.5,
            label = false,
        )

        push!(
            testes,
            (;
                indice = indice_teste,
                txt_output = bfgs_output,
                bfgs = aceitos,
                pontos_bfgs = (x = xs, y = ys),
                rmsd_bfgs,
            ),
        )
    end

    if isempty(testes)
        @warn "Nenhum teste de BFGS foi plotado."
    end

    mkpath(dirname(output))
    Plots.savefig(base.plot, output)

    return merge(base, (; output, testes_bfgs = testes))
end


function excluir()
    plot_busca_exaustiva_derivada_run_problems(
    X0 = [0.09, 0.09],
    tins = 31.0,
    n_points = 200,
    fmax = 50.0,
    )
    plot_heatmap_bfgs_default_assimilacao(grid_points = 50)
end
