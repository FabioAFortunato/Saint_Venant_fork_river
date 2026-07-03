using LinearAlgebra
using Optim
using ForwardDiff

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Plots

include("obj_func.jl")
include("aux_func.jl")


function assimilation_obj(f::Function,
                          ng::AbstractVector{T},
                          n_prev::AbstractVector{S},
                          tbeg::Real,
                          tend::Real,
                          estado0 = nothing;
                          λ::Real = 1.0) where {T<:Real,S<:Real}

    resultado = f(ng, tbeg, tend, estado0)
    erro = resultado.erro

    if !(eltype(ng) <: ForwardDiff.Dual)
        RMSD = norm(erro/sqrt(size(erro, 1)))
        print("RMSD = $RMSD \n")
    end

    return dot(erro, erro)
end


function manning_reta_na_malha(x::AbstractVector; n = 101)
    return collect(range(x[1], x[end], length = n))
end

function run_assimilation(; 
    X0 = fill(0.09, 2),
    tins = 31.0:1.0:31.0,
    λ = 1.0,
    penalty_weight = 1.0e6,
    bfgs_f_calls_limit = 50,
    bfgs_g_calls_limit = 20,
    bfgs_iterations = 10,
    rmsd_tol = 0.09,
    fun::Function = sv_fork_assimilation,
    )

    X_prev = copy(X0)
    estado_prev = nothing
    tbeg = 0.0

    history = []

    for tend in tins 

        dim = length(X_prev)

        println("\n==============================")
        println("Assimilação de tbeg = ", tbeg, " até tend = ", tend)
        println("==============================")

        tempo = time()
        resultado_inicial = fun(X_prev, tbeg, tend, estado_prev)
        tempo_final_fun = time() - tempo
        RMSD = norm(resultado_inicial.erro / sqrt(size(resultado_inicial.erro, 1)))

        println("RMSD inicial = ", RMSD)

        if RMSD > 100.0
            println("Chute inicial ficou 0.08")
            plot_assimilation_profiles(history)
            return history
        end

        lower_bfgs = zeros(dim)
        upper_bfgs = fill(0.5, dim)

        X_ref = copy(X_prev)

        f_bfgs(ng) = assimilation_obj(fun, ng, X_ref, tbeg, tend, estado_prev; λ = λ)

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

        # tempo = time()
        result = zeros(2)
        # g_obj!(result, X0)

        tempo_final = 0.0
        return result, tempo_final_fun, tempo_final

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

        estado_bfgs = fun(X_bfgs, tbeg, tend, estado_prev)
        RMSD_bfgs = norm(estado_bfgs.erro / sqrt(size(estado_bfgs.erro, 1)))

        println("fval BFGS = ", fval_bfgs)
        println("RMSD após BFGS = ", RMSD_bfgs)

        if RMSD_bfgs < 10.0
            tbeg_aceito = tbeg

            push!(history, (
                    tbeg = tbeg,
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
            tbeg = tend
            println("RMSD aceitável com BFGS para o intervalo [", tbeg_aceito, ", ", tend, "]")
            continue
        end

        push!(history, (
                tbeg = tbeg,
                tend = tend,
                stage = :BFGS,
                dim = dim,
                fval = fval_bfgs,
                RMSD = RMSD_bfgs,
                X = copy(X_bfgs),
                estado = estado_bfgs,
                result = resultado_bfgs,
            ))

        println("O BFGS não encontrou um ponto adequado para o intervalo [", tbeg, ", ", tend, "]")
        plot_assimilation_profiles(history)
        return history
    end

    plot_assimilation_profiles(history)
    return history
end


function sv_fork_new(
    ng::AbstractVector{T},
    tbeg,
    tend,
    estado0 = nothing;
    plot_bool = false,
    ) where T<:Real

    # ------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------
    local alfa = T(0.99)          # artificial diffusion; alfa = 1 means no smoothing
    local ualfa = one(T) - alfa
    local xmax = 3256.0
    local xmin = -39.0
    local dt = 1.0
    local grav = 9.8
    local dx = (xmax - xmin)/(nx - 1)

    local penalty = T(1e27)
    local amin_safe = T(1e-10)
    local hmin_safe = T(1e-8)

    local n_man = length(ng)

    if n_man < 1 || n_man > nx
        error("Wrong dimension. Please choose ng with size between 2 and nx.")
    end

    # ------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------
    local zb = zeros(T, nx)
    local z = zeros(T, nx)
    local h = zeros(T, nx)
    local av = zeros(T, nx)

    local ancho = zeros(T, nx)
    local a = zeros(T, nx)
    local v = zeros(T, nx)

    local anew = zeros(T, nx)
    local avnew = zeros(T, nx)
    local vnew = zeros(T, nx)
    local hnew = zeros(T, nx)
    local zhatx = zeros(T, nx)

    local televa = Float64[]
    local zou = T[]
    local zinterior = T[]
    local z3256_estimado = T[]
    local z751_estimado = T[]
    local z3256 = T[]
    local z751 = T[]

    local t = 60.0*60.0*24.0*tbeg
    local tmax = 60.0*60.0*24.0*tend
    local tmix_aux = 60.0*60.0*24.0*max(3.0, tbeg)
    local nt_local = floor(Int, (tmax - tmix_aux)/timprim + 1e-10) + 1


    local ya = 0
    local continuar = true

    # Diagnostics
    local h_min_global = T(Inf)
    local v_max_global = T(0.0)
    local cfl_max_global = T(0.0)
    local fr_max_global = T(0.0)

    # Correct index for x = 751
    local idx_751 = clamp(round(Int, 1 + (751.0 - xmin)/dx), 1, nx)

    # ------------------------------------------------------------
    # Auxiliary function: penalty return
    # ------------------------------------------------------------
    function failed_return(t_current)
        vec_aux = fill(penalty, 2*nt_local)
        return (
            erro = vec_aux,
            z = copy(z),
            a = copy(a),
            h = copy(h),
            v = copy(v),
            av = copy(av),
            t = t_current,
            ok = false,
            h_min = h_min_global,
            v_max = v_max_global,
            cfl_max = cfl_max_global,
            fr_max = fr_max_global,
        )
    end

    # ------------------------------------------------------------
    # Auxiliary function: Manning interpolation
    # ------------------------------------------------------------
    function manning_at(i::Int)
        if n_man == 2
            theta = T((i - 1)/(nx - 1))
            return (one(T) - theta)*ng[1] + theta*ng[2]
        elseif n_man == 1
            return ng[1]
        elseif n_man == nx
            return ng[i]
        else
            pos_ng = 1.0 + (i - 1)*(n_man - 1)/(nx - 1)
            j = floor(Int, pos_ng)
            j = clamp(j, 1, n_man - 1)
            frac = T(pos_ng - j)

            return (one(T) - frac)*ng[j] + frac*ng[j+1]
        end
    end

    # ------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------
    for i = 1:nx
        x = xmin + dx*(i - 1)
        zb[i] = zbfork(x)
        ancho[i] = anchofork(x)

        if ancho[i] <= 0 || !isfinite(ancho[i])
            return failed_return(t)
        end
    end

    # ------------------------------------------------------------
    # Initial condition
    # ------------------------------------------------------------
    if tbeg == 0.0 || estado0 === nothing
        for i = 1:nx
            x = xmin + dx*(i - 1)

            z[i] = zfork(x)
            h[i] = z[i] - zb[i]
            a[i] = ancho[i]*h[i]
            av[i] = qinlet(t)

            if a[i] <= amin_safe || h[i] <= hmin_safe || !isfinite(a[i]) || !isfinite(h[i])
                return failed_return(t)
            end

            v[i] = av[i]/a[i]

            if !isfinite(v[i])
                return failed_return(t)
            end
        end
    else
        for campo in (:z, :a, :h, :v, :av)
            if !hasproperty(estado0, campo)
                error("estado0 deve conter o campo .$campo.")
            end
        end

        if length(estado0.z) != nx ||
           length(estado0.a) != nx ||
           length(estado0.h) != nx ||
           length(estado0.v) != nx ||
           length(estado0.av) != nx
            error("Os vetores de estado0 devem ter tamanho nx = $nx.")
        end

        for i = 1:nx
            z[i] = estado0.z[i]
            a[i] = estado0.a[i]
            h[i] = estado0.h[i]
            v[i] = estado0.v[i]
            av[i] = estado0.av[i]

            if a[i] <= amin_safe || h[i] <= hmin_safe ||
               !isfinite(a[i]) || !isfinite(h[i]) ||
               !isfinite(v[i]) || !isfinite(av[i])
                return failed_return(t)
            end
        end
    end

    # ------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------
    while t <= tmax


        av_old = copy(av)
        h_old = copy(h)

        for i = 2:nx-1
            av[i] = alfa*av_old[i] + ualfa*(av_old[i-1] + av_old[i+1])/2
            h[i]  = alfa*h_old[i]  + ualfa*(h_old[i-1]  + h_old[i+1])/2
        end

        for i = 1:nx
            z[i] = h[i] + zb[i]
            a[i] = ancho[i]*h[i]
            v[i] = av[i]/a[i]
        end

        # --------------------------------------------------------
        # Writing / observations
        # --------------------------------------------------------
        imprim = tmix_aux + ya*timprim

        if (t >= imprim) && (t <= tmax)
            ya += 1
            push!(televa, t)

            if fake == 1
                z_3256 = zfinal_fake(t)
                z_751 = zmeio_fake(t)
            else
                z_3256 = zoutlet(t)
                z_751 = zhistomedio(t)
            end

            push!(z751_estimado, z[idx_751])
            push!(z3256_estimado, z[nx])
            push!(z751, z_751)
            push!(z3256, z_3256)

            push!(zou, z[nx] - z_3256)
            push!(zinterior, z[idx_751] - z_751)
        end

        # --------------------------------------------------------
        # Advance time
        # --------------------------------------------------------
        t += dt

        # --------------------------------------------------------
        # Explicit update
        # --------------------------------------------------------
        for i = 1:nx

            # ----------------------------------------------------
            # Mass equation
            # ----------------------------------------------------
            if i == 1
                avx = (av[i+1] - av[i])/T(dx)
            elseif i == nx
                avx = (av[i] - av[i-1])/T(dx)
            else
                avx = (av[i+1] - av[i-1])/T(2.0*dx)
            end

            at = -avx
            anew[i] = a[i] + T(dt)*at

            if anew[i] <= amin_safe || !isfinite(anew[i])
                continuar = false
                break
            end

            # ----------------------------------------------------
            # Momentum equation
            # ----------------------------------------------------
            if i == 1
                av2x = (av[i+1]*v[i+1] - av[i]*v[i])/T(dx)
                zhatx[i] = (z[i+1] - z[i])/T(dx)

            elseif i == nx
                av2x = (av[i]*v[i] - av[i-1]*v[i-1])/T(dx)
                zhatx[i] = (z[i] - z[i-1])/T(dx)

            else
                av2x = (av[i+1]*v[i+1] - av[i-1]*v[i-1])/T(2.0*dx)
                zhatx[i] = (z[i+1] - z[i-1])/T(2.0*dx)
            end

            zhatx[i] = zhatx[i]/(one(T) + zhatx[i]^2)

            if !isfinite(zhatx[i]) || !isfinite(av2x)
                continuar = false
                break
            end

            # ----------------------------------------------------
            # Manning coefficient
            # ----------------------------------------------------
            eneg = manning_at(i)

            if eneg < 0 || !isfinite(eneg)
                continuar = false
                break
            end

            # ----------------------------------------------------
            # Hydraulic radius
            # ----------------------------------------------------
            peri = ancho[i] + T(2.0)*h[i]

            if a[i] <= amin_safe || h[i] <= hmin_safe || peri <= 0 || !isfinite(peri)
                continuar = false
                break
            end

            rh = a[i]/peri

            if rh <= 0 || !isfinite(rh)
                continuar = false
                break
            end

            rh3 = rh^(T(4.0)/T(3.0))

            if rh3 <= 0 || !isfinite(rh3)
                continuar = false
                break
            end

            # ----------------------------------------------------
            # Momentum RHS
            # ----------------------------------------------------
            avt =
                -av2x -
                T(grav)*a[i]*zhatx[i] -
                eneg^2 * av[i] * sqrt(av[i]^2 + T(1e-3))/(rh3*a[i])

            if !isfinite(avt)
                continuar = false
                break
            end

            avnew[i] = av[i] + T(dt)*avt

            if !isfinite(avnew[i])
                continuar = false
                break
            end

            # ----------------------------------------------------
            # New h and v
            # ----------------------------------------------------
            hnew[i] = anew[i]/ancho[i]

            if hnew[i] <= hmin_safe || !isfinite(hnew[i])
                continuar = false
                break
            end

            if abs(avnew[i]) == 0
                vnew[i] = zero(T)
            else
                vnew[i] = avnew[i]/anew[i]
            end

            if !isfinite(vnew[i])
                continuar = false
                break
            end
        end

        if !continuar
            break
        end

        # --------------------------------------------------------
        # Upstream boundary condition
        # --------------------------------------------------------
        avnew[1] = qinlet(t)

        if anew[1] <= amin_safe || !isfinite(anew[1]) || !isfinite(avnew[1])
            return failed_return(t)
        end

        vnew[1] = avnew[1]/anew[1]

        if !isfinite(vnew[1])
            return failed_return(t)
        end

        # --------------------------------------------------------
        # Accept new state
        # --------------------------------------------------------
        for i = 1:nx
            a[i] = anew[i]
            av[i] = avnew[i]
            h[i] = hnew[i]
            v[i] = vnew[i]
            z[i] = zb[i] + h[i]

            if a[i] <= amin_safe || h[i] <= hmin_safe ||
               !isfinite(a[i]) || !isfinite(av[i]) ||
               !isfinite(h[i]) || !isfinite(v[i]) ||
               !isfinite(z[i])
                continuar = false
                break
            end
        end

        if !continuar
            break
        end
    end

    # ------------------------------------------------------------
    # Failure
    # ------------------------------------------------------------
    
    if !continuar
        return failed_return(t)
    end

    # ------------------------------------------------------------
    # Build error vector
    # ------------------------------------------------------------
    z_error = [zou; zinterior]

    if length(z_error) != 2*nt_local
        return failed_return(t)
    end

    for val in z_error
        if !isfinite(val)
            return failed_return(t)
        end
    end

    # ------------------------------------------------------------
    # Return
    # ------------------------------------------------------------
    
    if plot_bool
        televa .= televa./(24.0*60*60)   
        pgfplotsx() 
  
        # Definir passo para amostragem
          step = 20
          
          # BOBYQA - coluna esquerda
          indices = 1:step:length(z751_estimado)
         #  Segunda linha - m=3256
          local pl2 = plot(televa, z751_estimado,
              label = "", linewidth = 2, color = :cornflowerblue, 
              xlims = (2, 31),
              ylims = (5, 9.1))
          
          plot!(televa[1:2, 1], z751_estimado[1:2],
              label = "m=751", linewidth = 2, markershape = :utriangle,
              markerstrokewidth = 0, color = :cornflowerblue, legend = (0.0, 0.6))
  
          # Marcadores para segunda linha
          scatter!(televa[indices], z751_estimado[indices],
              label = "",
              markershape = :utriangle,
              markersize = 4,
              markercolor = :cornflowerblue, markerstrokewidth = 0)
  
          # Linha sem marcadores
          plot!(televa, z3256_estimado,
              label = "",
              legend = (0.0, 0.6),
              title = "",
              ylabel = "m",
              xlabel = "t", color = :red)
  
          plot!(televa[1:2], z3256_estimado[1:2],
              label = "m=3256", linewidth = 2,
              legend = (0.0, 0.6),
              markershape = :x,
              markersize = 4,
              color = :red,
              alpha = 1.0)
  
          # Marcadores espaçados manualmente
          scatter!(televa[indices], z3256_estimado[indices],
              label = "",  # Label vazio para não duplicar
              markershape = :x,
              markersize = 4,
              markercolor = :red,
              markerstrokewidth = 1.5)
  
  
          # Dados (mantém todos os pontos)
          scatter!(televa, z3256,
              label = "Data", markersize = 1.0, color = :black, legend = (0.0, 0.6),
              )
  
          scatter!(televa, z751,
              label = "", markersize = 1.0, color = :black)


        return (
        erro = z_error,
        z = copy(z),
        a = copy(a),
        h = copy(h),
        v = copy(v),
        av = copy(av),
        t = t,
        ok = true,
        h_min = h_min_global,
        v_max = v_max_global,
        cfl_max = cfl_max_global,
        fr_max = fr_max_global,
        z751_estimado = z751_estimado,
        z3256_estimado = z3256_estimado,
        z751 = z751,
        z3256 = z3256,
        plot = pl2,
        )
    else
        return (
        erro = z_error,
        z = copy(z),
        a = copy(a),
        h = copy(h),
        v = copy(v),
        av = copy(av),
        t = t,
        ok = true,
        h_min = h_min_global,
        v_max = v_max_global,
        cfl_max = cfl_max_global,
        fr_max = fr_max_global,
        z751_estimado = z751_estimado,
        z3256_estimado = z3256_estimado,
        z751 = z751,
        z3256 = z3256
        )
    end
    
end



function plot_assimilation_profiles(history)
    if isempty(history)
        println("Nenhum estado foi salvo; gráficos de assimilação não foram gerados.")
        return nothing
    end

    plot_specs = (
        (
            nome = :av,
            serie = h -> collect(h.estado.av),
            ylabel = "Vazão av",
            title = "Vazão av final por intervalo",
            arquivo = arquivo_em_results("assimilacao_av_por_intervalo.png"),
            mensagem = "Gráfico de av por intervalo salvo em: ",
        ),
        (
            nome = :z,
            serie = h -> collect(h.estado.z),
            ylabel = "Cota z",
            title = "Cota z final por intervalo",
            arquivo = arquivo_em_results("assimilacao_z_por_intervalo.png"),
            mensagem = "Gráfico de z por intervalo salvo em: ",
        ),
        (
            nome = :manning,
            serie = h -> manning_reta_na_malha(h.X),
            ylabel = "Manning",
            title = "Manning interpolado por intervalo",
            arquivo = arquivo_em_results("assimilacao_manning_por_intervalo.png"),
            mensagem = "Gráfico de Manning por intervalo salvo em: ",
        ),
    )

    plots_salvos = []

    for spec in plot_specs
        primeira_serie = spec.serie(history[1])
        pl = plot(
            1:length(primeira_serie),
            primeira_serie,
            xlabel = "Ponto da malha",
            ylabel = spec.ylabel,
            label = "run 1: $(history[1].tbeg)-$(history[1].tend)",
            title = spec.title,
            linewidth = 2,
        )

        for i in 2:length(history)
            serie_i = spec.serie(history[i])
            plot!(
                pl,
                1:length(serie_i),
                serie_i,
                label = "run $i: $(history[i].tbeg)-$(history[i].tend)",
                linewidth = 2,
            )
        end

        savefig(pl, spec.arquivo)
        println(spec.mensagem, spec.arquivo)
        push!(plots_salvos, spec.nome => pl)
    end

    return (; plots_salvos...)
end
