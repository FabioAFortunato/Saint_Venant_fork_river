#!/usr/bin/env julia

# Translation of tmp/geodis.for.
# The values that the Fortran program requested interactively are fixed here.

const NNPUN = 10000
const NNDIM = 3

Base.@kwdef struct Config
    npun::Int = 100
    ndim::Int = 2
    perc::Float64 = 0.0
    petol::Float64 = 0.0
    konmax::Int = 5000
    continue_coordinate_search::Bool = true
    step::Float64 = 0.5
    delta::Float64 = 0.001
    overwrite_distances_with_random_exact::Bool = true
end

struct GeodisProblem
    dislow::Matrix{Float64}
    disup::Matrix{Float64}
    ndim::Int
    npun::Int
end

mutable struct SchrageRNG
    seed::Float64
end

function rando!(rng::SchrageRNG)
    a = 16807.0
    b15 = 32768.0
    b16 = 65536.0
    p = 2147483647.0

    xhi = floor(rng.seed / b16)
    xalo = (rng.seed - xhi * b16) * a
    leftlo = floor(xalo / b16)
    fhi = xhi * a + leftlo
    k = floor(fhi / b15)
    rng.seed = (((xalo - leftlo * b16) - p) + (fhi - k * b15) * b16) + k
    if rng.seed < 0.0
        rng.seed += p
    end
    return rng.seed * 4.656612875e-10
end

coord_index(point::Int, coord::Int, ndim::Int) = (point - 1) * ndim + coord

function fun(problem::GeodisProblem, x::AbstractVector{Float64})
    f = 0.0
    for j in 1:problem.npun
        for i in 1:j-1
            z = 0.0
            for k in 1:problem.ndim
                diff = x[coord_index(j, k, problem.ndim)] -
                       x[coord_index(i, k, problem.ndim)]
                z += diff^2
            end

            if z > problem.disup[i, j]
                f += (z - problem.disup[i, j])^2
            end
            if z < problem.dislow[i, j]
                f += (z - problem.dislow[i, j])^2
            end
        end
    end
    n = size(x, 1)
    for i = 1:2:round(Int32, n)
        if x[i] > 0.1*x[i+1]
            return 10E+26
        end
    end
    return f
end

function grad!(g::Vector{Float64}, problem::GeodisProblem, x::Vector{Float64})
    fill!(g, 0.0)

    for j in 1:problem.npun
        for i in 1:j-1
            z = 0.0
            for k in 1:problem.ndim
                diff = x[coord_index(j, k, problem.ndim)] -
                       x[coord_index(i, k, problem.ndim)]
                z += diff^2
            end

            dz = 0.0
            if z > problem.disup[i, j]
                dz = 2.0 * (z - problem.disup[i, j])
            elseif z < problem.dislow[i, j]
                dz = 2.0 * (z - problem.dislow[i, j])
            end

            if dz != 0.0
                for k in 1:problem.ndim
                    idx_j = coord_index(j, k, problem.ndim)
                    idx_i = coord_index(i, k, problem.ndim)
                    diff = x[idx_j] - x[idx_i]
                    g[idx_j] += 2.0 * dz * diff
                    g[idx_i] -= 2.0 * dz * diff
                end
            end
        end
    end

    return g
end

function grad2(problem::GeodisProblem, x::Vector{Float64}, f::Float64)
    h = 1.0e-6
    g = zeros(length(x))
    aux = zeros(length(x), 3)

    for k in eachindex(x)
        save = x[k]
        x[k] = save + h
        fmas = fun(problem, x)
        aux[k, 3] = fmas

        x[k] = save - h
        fmen = fun(problem, x)
        aux[k, 2] = fmen

        g[k] = (fmas - fmen) / (2.0 * h)
        x[k] = save
        aux[k, 1] = (fmas < f || fmen < f) ? 1.0 : 0.0
    end

    return g, aux
end

function generate_problem(config::Config, rng::SchrageRNG)
    if config.ndim > NNDIM
        error("ndim=$(config.ndim) exceeds translated Fortran limit NNDIM=$NNDIM")
    end
    if config.npun > NNPUN
        error("npun=$(config.npun) exceeds translated Fortran limit NNPUN=$NNPUN")
    end

    n = config.ndim * config.npun
    x = zeros(n)
    distrue = zeros(config.npun, config.npun)
    disobs = zeros(config.npun, config.npun)
    dislow = zeros(config.npun, config.npun)
    disup = zeros(config.npun, config.npun)

    for j in 1:config.npun
        for k in 1:config.ndim
            x[coord_index(j, k, config.ndim)] = 100.0 * rando!(rng)
        end
    end
    

    for j in 1:config.npun
        for i in 1:j-1
            z = 0.0
            for k in 1:config.ndim
                diff = x[coord_index(j, k, config.ndim)] -
                       x[coord_index(i, k, config.ndim)]
                z += diff^2
            end
            distrue[i, j] = z
            distrue[j, i] = z
        end
    end

    for j in 1:config.npun
        for i in 1:j-1
            ran = (rando!(rng) - 0.5) * 2.0
            disobs[i, j] = distrue[i, j] + distrue[i, j] * ran * config.perc / 100.0
            disobs[j, i] = disobs[i, j]
        end
    end

    for j in 1:config.npun
        for i in 1:j-1
            dislow[i, j] = disobs[i, j] - disobs[i, j] * config.petol / 100.0
            disup[i, j] = disobs[i, j] + disobs[i, j] * config.petol / 100.0
            dislow[j, i] = dislow[i, j]
            disup[j, i] = disup[i, j]
        end
    end

    # This block preserves the current Fortran behavior at lines 84-94.
    # It replaces the generated observed distances by exact random distances.
    if config.overwrite_distances_with_random_exact
        for j in 1:config.npun
            for i in 1:j-1
                disobs[i, j] = rando!(rng)
                disobs[j, i] = disobs[i, j]
                dislow[i, j] = disobs[i, j]
                disup[i, j] = disobs[i, j]
                dislow[j, i] = dislow[i, j]
                disup[j, i] = disup[i, j]
            end
        end
    end

    for j in 1:config.npun
        for k in 1:config.ndim
            x[coord_index(j, k, config.ndim)] = 100.0 * rando!(rng)
        end
    end

    for i = 1:2:round(Int32, n)
        if x[i] > 0.1*x[i+1]
            x[i] = 0.1*x[i+1]
        end
    end


    return GeodisProblem(dislow, disup, config.ndim, config.npun), x
end

function spgbox!(
    problem::GeodisProblem,
    x::Vector{Float64},
    lower::Vector{Float64},
    upper::Vector{Float64};
    m::Int = 5,
    konmax::Int = 1000,
    nafmax::Int = 1_000_000,
    eps::Float64 = 1.0e-4,
    epslog::Float64 = 1.0e-15,
    ftarget::Float64 = 1.0e-16,
    tmax::Float64 = 1.0e4,
)
    n = length(x)
    g = zeros(n)
    xn = similar(x)
    gn = similar(x)
    fant = zeros(100)

    kon = 0
    f = fun(problem, x)
    fmin = f
    naf = 1
    grad!(g, problem, x)
    gnoran = 0.0
    tspg = tmax

    while true
        println()
        println("Iteracao ", kon)
        println("X = ", x[1], " ... ", x[end])
        println("f(X) = ", f)
        println("fmin = ", fmin)

        ninf = count(i -> x[i] <= lower[i], eachindex(x))
        nsup = count(i -> x[i] >= upper[i], eachindex(x))
        println("Variables in lower bound: ", ninf,
                " Variables in upper bound: ", nsup)

        gnor = 0.0
        for i in 1:n
            z = max(lower[i], min(upper[i], x[i] - g[i])) - x[i]
            gnor = max(gnor, abs(z))
        end
        println("Norma sup do gradiente projetado = ", gnor)
        println("Avaliacoes de f(x) = ", naf)

        if kon != 0
            coc = gnor / gnoran
            println("Cociente entre gradientes projetados consecutivos: ", coc)
        end
        gnoran = gnor

        if gnor <= eps
            return (; ier = 0, kon, naf, f, fmin, gnor, g)
        end
        if gnor / (1.0 + f) <= epslog
            println("Return because gradient of logarithm < ", epslog)
            return (; ier = 0, kon, naf, f, fmin, gnor, g)
        end
        if f <= ftarget
            return (; ier = 0, kon, naf, f, fmin, gnor, g)
        end
        if kon >= konmax
            println("Maximo de iteracoes atingido")
            return (; ier = 1, kon, naf, f, fmin, gnor, g)
        end
        if naf >= nafmax
            println("Maximo de avaliacoes atingido")
            return (; ier = 2, kon, naf, f, fmin, gnor, g)
        end

        if kon == 0
            for i in 1:n
                xn[i] = x[i] - 0.01 * g[i] / gnor
            end
            grad!(gn, problem, xn)
            num = 0.0
            den = 0.0
            for i in 1:n
                s = xn[i] - x[i]
                num += s^2
                den += s * (gn[i] - g[i])
            end
            tspg = den <= 0.0 ? tmax : min(num / den, tmax)
            println("Initial step when koko = 0 : ", tspg)
            fill!(view(fant, 1:m), f)
        end

        t = tspg
        fref = maximum(view(fant, 1:m))

        while true
            for i in 1:n
                xn[i] = max(lower[i], min(x[i] - t * g[i], upper[i]))
            end

            fn = fun(problem, xn)
            fmin = min(fn, fmin)
            naf += 1

            if naf > nafmax
                println("Maximo de avaliacoes atingido")
                return (; ier = 2, kon, naf, f, fmin, gnor, g)
            end

            if fn <= fref
                grad!(gn, problem, xn)
                num = 0.0
                den = 0.0
                for i in 1:n
                    s = xn[i] - x[i]
                    num += s^2
                    den += s * (gn[i] - g[i])
                end
                tspg = den <= 0.0 ? tmax : min(num / den, tmax)

                f = fn
                copyto!(x, xn)
                copyto!(g, gn)
                for i in 1:m-1
                    fant[i] = fant[i + 1]
                end
                fant[m] = f
                kon += 1
                break
            end

            t /= 2.0
        end
    end
end

function buscoor!(problem::GeodisProblem, x::Vector{Float64}, konmax::Int, h::Float64, rng::SchrageRNG)
    f = fun(problem, x)
    println("f initial: ", f)
    kon = 1

    while true
        if kon >= konmax
            println("Evaluations exhausted ", kon)
            println(f)
            return (; f, kon)
        end

        ico = Int(floor(length(x) * rando!(rng) + 1.0))
        save = x[ico]
        x[ico] = save + h
        fn = fun(problem, x)
        if fn < f
            f = fn
            println("Success, kon = ", kon, " f= ", f)
        end

        if fn <= f
            f = fn
            continue
        end

        x[ico] = save - h
        fn = fun(problem, x)
        kon += 1
        if fn < f
            f = fn
            println("Success, kon = ", kon, " f= ", f)
        end

        if fn <= f
            f = fn
            continue
        end

        x[ico] = save
    end
end

function minud!(problem::GeodisProblem, amin::Float64, amax::Float64,
                x::Vector{Float64}, d::Vector{Float64}, xtmp::Vector{Float64};
                tol::Float64 = 1.0e-4, maxit::Int = 20)
    c_gold = 0.3819660112501051
    a = amin
    b = amax
    x_min = a + c_gold * (b - a)
    w = x_min
    v = w
    e = 0.0
    d_step = 0.0

    @. xtmp = x + x_min * d
    fx = fun(problem, xtmp)
    println("f inicial en minud: ", fx)
    fw = fx
    fv = fx
    it = 0

    for iter in 1:maxit
        it = iter
        xm = 0.5 * (a + b)
        tol1 = tol * abs(x_min) + tol / 10.0
        tol2 = 2.0 * tol1
        if abs(x_min - xm) <= (tol2 - 0.5 * (b - a))
            break
        end

        if abs(e) > tol1
            r = (x_min - w) * (fx - fv)
            q = (x_min - v) * (fx - fw)
            p = (x_min - v) * q - (x_min - w) * r
            q = 2.0 * (q - r)
            if q > 0.0
                p = -p
            end
            q = abs(q)
            r = e
            e = d_step

            if abs(p) >= abs(0.5 * q * r) || p <= q * (a - x_min) || p >= q * (b - x_min)
                e = x_min >= xm ? a - x_min : b - x_min
                d_step = c_gold * e
            else
                d_step = p / q
                u = x_min + d_step
                if u - a < tol2 || b - u < tol2
                    d_step = copysign(tol1, xm - x_min)
                end
            end
        else
            e = x_min >= xm ? a - x_min : b - x_min
            d_step = c_gold * e
        end

        u = x_min + d_step
        @. xtmp = x + u * d
        fu = fun(problem, xtmp)

        if fu <= fx
            if u >= x_min
                a = x_min
            else
                b = x_min
            end
            v = w
            fv = fw
            w = x_min
            fw = fx
            x_min = u
            fx = fu
        else
            if u < x_min
                a = u
            else
                b = u
            end
            if fu <= fw || w == x_min
                v = w
                fv = fw
                w = u
                fw = fu
            elseif fu <= fv || v == x_min || v == w
                v = u
                fv = fu
            end
        end
    end

    alpha = x_min
    @. x = x + alpha * d
    f = fun(problem, x)
    println("Iterations of Minud: ", it)
    println("f(x) final en Minud = ", f)
    return alpha, f
end

function buscar!(problem::GeodisProblem, x::Vector{Float64}, konmax::Int,
                 h::Float64, rng::SchrageRNG)
    f = fun(problem, x)
    println("f initial en buscar: ", f)
    if f <= 1.0e-6
        return (; f, kon = 0)
    end

    n = length(x)
    d = zeros(n)
    aux = zeros(n)
    kon = 1

    while true
        if kon >= konmax
            println("Evaluations exhausted ", kon)
            println(f)
            return (; f, kon)
        end

        ico = Int(floor(n * rando!(rng) + 1.0))
        fill!(d, 0.0)
        d[ico] = 1.0
        amax = max(1.0, abs(x[ico]))
        amin = -amax
        println("ico = ", ico)

        if rando!(rng) >= 0.5
            amax = 0.0
        else
            amin = 0.0
        end

        alpha, f = minud!(problem, amin, amax, x, d, aux)
        x[ico] += alpha
        f = fun(problem, x)
    end
end

function coort2!(problem::GeodisProblem, x::Vector{Float64}, step::Float64)
    f = fun(problem, x)
    fini = f
    n = length(x)

    while true
        fail = 0
        for j in 1:n
            save = x[j]

            x[j] = save + step
            xmas = x[j]
            fmas = fun(problem, x)

            x[j] = save - step
            xmen = x[j]
            fmen = fun(problem, x)

            x[j] = save
            fnew = min(fmas, fmen)
            if fnew >= f
                fail += 1
                continue
            end

            if fmas == fnew
                h = step
                x[j] = xmas
            else
                h = -step
                x[j] = xmen
            end
            f = fnew

            while true
                save = x[j]
                x[j] = save + h
                fnew = fun(problem, x)
                if fnew < f
                    f = fnew
                else
                    x[j] = save
                    break
                end
            end
        end

        if fail == n
            break
        end
    end

    f = fun(problem, x)
    if fini < f
        error("Warning, fini = $fini f = $f")
    end
    return f
end

function coortos!(problem::GeodisProblem, x::Vector{Float64}, step::Float64, delta::Float64)
    f = fun(problem, x)
    while true
        f = coort2!(problem, x, step)
        if step <= delta
            return f
        end
        step /= 2.0
    end
end



using NOMAD
using LinearAlgebra
using Plots 
using PGFPlotsX

function main(config::Config = Config())
    rng = SchrageRNG(17171717172223.0)
    problem, x = generate_problem(config, rng)
    n = config.ndim * config.npun
    lower = fill(-1.0e3, n)
    upper = fill(1.0e3, n)
    nome = "tmp/geodis_NaN.txt"

    println("Number of points: ", config.npun)
    println("Dimension of the space: ", config.ndim)
    println("Percentage of generated error at observed distance: ", config.perc)
    println("Percent uncertainty for each measured square distance: ", config.petol)
    println("Maximo de iteracoes em spgbox: ", config.konmax)
    println("Calling spgbox:")

    start_time = time()
    result = spgbox!(
        problem,
        x,
        lower,
        upper;
        m = 5,
        konmax = config.konmax,
        nafmax = 1_000_000,
        eps = 1.0e-4,
        epslog = 1.0e-15,
        ftarget = 1.0e-16,
        tmax = 1.0e4,
    )

    # Ploting the results
    x_plot = x[1:2:n]
    y_plot = x[2:2:n]
    f_SPG = fun(problem, x)
    g_SPG = zeros(n)
    grad!(g_SPG, problem, x)
    gnorm_SPG = norm(g_SPG)

    open(nome, "w") do file
        println(file, "Resultado obtido via SPG com funcao penalizada por 10E+26 quando x > 0.1y")
        println(file, "Resultado apenas utilizando SPG")
        println(file, "Resumo SPG")
        println(file, "npun = $(config.npun)")
        println(file, "ndim = $(config.ndim)")
        println(file, "f_SPG = $f_SPG")
        println(file, "||g_SPG|| = $gnorm_SPG")
        println(file, "x_min = $(minimum(x_plot)) | x_max = $(maximum(x_plot))")
        println(file, "y_min = $(minimum(y_plot)) | y_max = $(maximum(y_plot))")
        println(file, "")
        println(file, "metodo\tponto\tx\ty")
        for i in eachindex(x_plot)
            println(file, "SPG\t$i\t$(x_plot[i])\t$(y_plot[i])")
        end
        println(file, "---------------------------------------------------")
        println(file, "---------------------------------------------------")
        println(file, "---------------------------------------------------")
    end

    x_min, x_max = extrema(x_plot)
    y_min, y_max = extrema(y_plot)
    pl_SPG = scatter(
        x_plot,
        y_plot,
        label = "",
        xlims = (x_min, x_max),
        ylims = (y_min, y_max),
    )

    println("Solucao obtida por spgbox:")
    for j in 1:n
        println(j, " ", x[j])
    end

    println("cpu-time: ", time() - start_time, " seconds")

    if config.continue_coordinate_search
        function eval_fct(x_vec)
            f = fun(problem, x_vec)
            return (true, true, [f])
        end
        pb = NomadProblem(n, # number of inputs of the blackbox
                  1, # number of outputs of the blackbox
                  ["OBJ"], # type of outputs of the blackbox
                  eval_fct;
                  initial_mesh_size = 0.1*x,
                  min_mesh_size = 1E-8*x,
                  lower_bound=lower,
                  upper_bound=upper)
        pb.options.max_bb_eval = 5000
        ϵ = 10^-1
        result = solve(pb, x)
        x_new = result.x_sol
        x_plot = x_new[1:2:n]
        y_plot = x_new[2:2:n]
        f_MADS = fun(problem, x_new)
        g_MADS = zeros(n)
        grad!(g_MADS, problem, x_new)
        gnorm_MADS = norm(g_MADS)

        open(nome, "a") do file
            println(file, "Resultado obtido via MADS após aplicar SPG.")
            println(file, "f_SPG = $f_SPG")
            println(file, "||g_SPG|| = $gnorm_SPG")
            println(file, "Resumo MADS")
            println(file, "f_MADS = $f_MADS")
            println(file, "||g_MADS|| = $gnorm_MADS")
            println(file, "x_min = $(minimum(x_plot)) | x_max = $(maximum(x_plot))")
            println(file, "y_min = $(minimum(y_plot)) | y_max = $(maximum(y_plot))")
            println(file, "")
            println(file, "metodo\tponto\tx\ty")
            for i in eachindex(x_plot)
                println(file, "MADS\t$i\t$(x_plot[i])\t$(y_plot[i])")
            end
        end

        x_min, x_max = extrema(x_plot)
        y_min, y_max = extrema(y_plot)

        pl_MADS = scatter(
            x_plot,
            y_plot,
            label = "",
            xlims = (x_min, x_max),
            ylims = (y_min, y_max),
        )
        
        xgrid = range(x_min-ϵ, x_max+ϵ, length = 300)
        yline = 10 .* xgrid
        
        plot!(
            pl_MADS,
            xgrid,
            yline,
            fillrange = y_max+ϵ,
            fillalpha = 0.25,
            label = "y ≥ 10x"
        )
        
        plot!(
            pl_MADS,
            xgrid,
            yline,
            linewidth = 2,
            label = "x = 0.1y"
        )
        
        plot!(pl_MADS, xlims = (x_min-ϵ, x_max+ϵ), ylims = (y_min-ϵ, y_max+ϵ))
        
        # gn, aux = grad2(problem, x_new, f)

        # println("Gradiente analitico e discreto, f, fmen, fmas:")
        # for j in 1:n
        #     println(j, " ", norm(g[j]), " ", " ", aux[j, 2],
        #             " ", f, " ", aux[j, 3], " ", aux[j, 1])
        # end
    end

    png(pl_MADS, "tmp/plot_MADS.png")
    png(pl_SPG, "tmp/plot_SPG.png")

    return config.continue_coordinate_search ? (; problem, x, result, pl_SPG, pl_MADS) : (; problem, x, result, pl_SPG)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
