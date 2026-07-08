using DataFrames
using Printf
using GLM
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


function excluir()
    plot_busca_exaustiva_derivada_run_problems(
    X0 = [0.09, 0.09],
    tins = 31.0,
    n_points = 200,
    )
    plot_heatmap_bfgs_default_assimilacao(grid_points = 50)
end
