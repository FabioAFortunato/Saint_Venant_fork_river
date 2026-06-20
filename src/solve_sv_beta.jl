using LinearAlgebra
using Optim
using ForwardDiff

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Plots

include("obj_func.jl")
include("sv_fork.jl")
include("aux_func.jl")


function beta_obj(ng_beta::AbstractVector{T},
                          tend::Real) where {T<:Real}
    n = size(ng_beta, 1)
    resultado = sv_fork_beta(ng_beta[1:1], beta = ng_beta[n], tend  = tend)
    erro = resultado.erro

    if !(eltype(ng_beta) <: ForwardDiff.Dual)
        RMSD = norm(erro/sqrt(size(erro, 1)))
        print("RMSD = $RMSD \n")
    end

    return dot(erro, erro)
end


function grid_beta()

    df = DataFrame(ng = Float64[], beta = Float64[], RMSD = Float64[])

    for i in 0.05:0.005:0.2
        for j in 1.0:0.05:2.0
            error = beta_obj([i, j], 31.0)
            RMSD = norm(error)/sqrt(size(error, 1)) 
            if !isfinite(RMSD) || RMSD > 2.0
                RMSD = 2.0
            end
            push!(df, (i,j, RMSD))
        end
    end
    
    xs = sort(unique(df.ng))
    ys = sort(unique(df.beta))

    Z = Matrix{Float64}(undef, length(ys), length(xs))

    for i in eachindex(xs)
        for j in eachindex(ys)
            linha = df[(df.ng .== xs[i]) .& (df.beta .== ys[j]), :]

            if nrow(linha) == 0
                Z[j, i] = NaN
            else
                Z[j, i] = linha.RMSD[1]
            end
        end
    end

    p = heatmap(
        xs, ys, Z,
        xlabel = "Manning coefficient",
        ylabel = "\\beta",
        title = "RMSD as a function of Manning and beta",
        colorbar_title = "RMSD"
    )

    contour!(
        p,
        xs, ys, Z,
        linewidth = 1.5
    )

    savefig("results/beta_plot.png")
    
    
    return p
    
end


function manning_reta_na_malha(x::AbstractVector; n = 101)
    return collect(range(x[1], x[end], length = n))
end

function run_beta_prob(; 
    X0 = fill(0.08, 2),
    beta = 1.0,
    tend = 5.0,
    penalty_weight = 1.0e6,
    bfgs_f_calls_limit = 50,
    bfgs_g_calls_limit = 20,
    bfgs_iterations = 10,
    rmsd_tol = 0.09,
    )

    X_prev = copy(X0)
    append!(X_prev, beta)
    history = []

    dim = length(X_prev)

    println("\n==============================")
    println("Solução até tend = ", tend)
    println("==============================")

    resultado_inicial = sv_fork_beta(X_prev[1:dim-1], beta = X_prev[dim], tend = tend)
    RMSD = norm(resultado_inicial.erro / sqrt(size(resultado_inicial.erro, 1)))

    println("RMSD inicial = ", RMSD)

    if RMSD > 100.0
        println("Chute inicial ficou 0.08")
        plot_assimilation_profiles(history)
        return history
    end

    lower_bfgs = zeros(dim)
    upper_bfgs = fill(0.5, dim)
    lower_bfgs[dim] = 1.0
    upper_bfgs[dim] = Inf
    X_ref = copy(X_prev)

    f_bfgs(ng) = beta_obj(ng, tend)

    f_penalizada_bfgs(x_vars) = f_bfgs(x_vars) + penalidade_caixa_externa(
            x_vars,
            lower_bfgs,
            upper_bfgs;
            rho = penalty_weight,
        )

    cfg_bfgs = ForwardDiff.GradientConfig(
            f_penalizada_bfgs,
            X_ref,
            ForwardDiff.Chunk{dim}()
        )

    function g_obj!(G, x_k)
        ForwardDiff.gradient!(G, f_penalizada_bfgs, x_k, cfg_bfgs)
        return G
    end

    println("Rodando BFGS com dimensão = ", dim)
    
    resultado_bfgs = Optim.optimize(
            f_penalizada_bfgs,
            g_obj!,
            X_ref,
            BFGS(linesearch = ArmijoThenWolfe(g_obj!)),
            Optim.Options(
                iterations = bfgs_iterations,
                f_calls_limit = bfgs_f_calls_limit,
                g_calls_limit = bfgs_g_calls_limit,
            ),
        )

    X_bfgs = Optim.minimizer(resultado_bfgs)
    X_bfgs = clamp.(X_bfgs, lower_bfgs, upper_bfgs)
    fval_bfgs = Optim.minimum(resultado_bfgs)

    estado_bfgs = sv_fork_beta(X_bfgs[1:dim-1], beta = X_bfgs[dim], tend = tend)
    RMSD_bfgs = norm(estado_bfgs.erro / sqrt(size(estado_bfgs.erro, 1)))

    println("fval BFGS = ", fval_bfgs)
    println("RMSD após BFGS = ", RMSD_bfgs)

    if RMSD_bfgs < 10.0

        push!(history, (
                tend = tend,
                stage = :BFGS,
                dim = dim,
                fval = fval_bfgs,
                RMSD = RMSD_bfgs,
                X = copy(X_bfgs),
                estado = estado_bfgs,
                result = resultado_bfgs,
            ))

        X_prev = copy(X_bfgs)
        estado_prev = estado_bfgs
        println("RMSD aceitável com BFGS para o intervalo")
        
    end

    push!(history, (
            tend = tend,
            stage = :BFGS,
            dim = dim,
            fval = fval_bfgs,
            RMSD = RMSD_bfgs,
            X = copy(X_bfgs),
            estado = estado_bfgs,
            result = resultado_bfgs,
        ))

    println("O BFGS não encontrou um ponto adequado para o intervalo")
    return history
end

