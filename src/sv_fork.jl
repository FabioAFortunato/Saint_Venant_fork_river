#git init # inicia o repositório local
#git add <arquivos> <ou * para tudo.. use com responsabilidade>
#git commit -m "comentario"
#git pull # puchar alterações do repositório remoto
#git push # enviar alterações para o repositório remoto
#git remote add <link> # adiciona e configura um repositório remoto

using DelimitedFiles
using Interpolations
using LinearAlgebra
using PGFPlotsX
using Plots

include("../data/processed/dado_fork.jl")


const fake = 0 # if fake = 1 then we use pregenerated data
const nx = 101 # grid number
const tmin = 0.0 # seconds
const tmax1 = 0.0 + 60.0 * 0.0 + 60.0 * 60.0 * 0.0 + 60.0 * 60.0 * 24.0 * 31.0
const timprim = 0.0 + 60.0 * 0.0 + 60.0 * 60.0 * 4.0 + 60.0 * 60.0 * 24.0 * 0.0
const nt = round(Int32, 1 + (tmax1 - tmin - 60.0 * 60.0 * 24.0 * 3.0) / timprim)
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

## Saint-Venant com método difusivo 
function sv_fork(ng::AbstractVector{T}, tin) where T<:Real
  # variable
    local alfa = 0.99 # termo de difusão artificial -  alfa = 1 é sem difusao
    local ualfa = 1.0 - alfa 
    local xmax = 3256.0 # end point/station
    local xmin = -39.0 # initial point/station
    local dt = 1.0 #second(s)
    local dx = (xmax-xmin)/(nx-1) #space discretization
    local grav = 9.8 #gravitational constant 
    local x=0.0; zb = zeros(T, nx); z = zeros(T, nx); h = zeros(T, nx); av = zeros(T, nx)
    local ancho = zeros(T, nx); a = zeros(T, nx); v = zeros(T, nx)
    local t = tmin; ya = 0; imprim = 0.0; televa = []; zou = []; zinterior = []; zfake = []; zmeiofake = []
    local at, avx; anew = zeros(T, nx); zhatx = zeros(T, nx); av2x = 0.0; peri=0.0
    local avt = 0.0; eneg = 0.0; avnew = zeros(T, nx); vnew = zeros(T, nx); hnew = zeros(T, nx)
    local tmix_aux = 0.0 + 60.0*0.0 + 60.0*60.0*0.0 + 60.0*60.0*24.0*3.0 # seconds / min / hours / days We have data after day 3
    local continuar = true
    local tmax = 0.0 + 60.0*0.0 + 60.0*60.0*0.0 + 60.0*60.0*24.0*tin
    local n_man = length(ng)
  # variable
    for i = 1:nx 
        x = xmin + dx*(i-1)
        zb[i] = zbfork(x)
        z[i] = zfork(x)
        h[i] = z[i] - zb[i]
        av[i] = qinlet(tmin)
        ancho[i] = anchofork(x)
        a[i] = ancho[i]*h[i]
        v[i] = av[i]/a[i]
    end
    #print("t/(60*60*24) zou[ya] zinterior[ya] \n")

    while t <= tmax

      #    Smoothing
        for i = 2:nx-1 
            a[i] = alfa*a[i] + ualfa*(a[i-1]+a[i+1])/2.0
        end
    
        for i = 2:nx-1
            v[i] = alfa*v[i] + ualfa*(v[i-1]+v[i+1])/2.0
        end

        for i = 2:nx-1
            z[i] = alfa*z[i] + ualfa*(z[i-1]+z[i+1])/2.0
        end

        for i = 2:nx-1
            h[i] = alfa*h[i] + ualfa*(h[i-1]+h[i+1])/2.0
        end

        for i = 2:nx-1
            av[i] = alfa*av[i] + ualfa*(av[i-1]+av[i+1])/2.0
        end

      # End of Smoothing

      #   Writing
        imprim = tmix_aux + (ya)*timprim
        if (t >= imprim)  && (t <= tmax)
            ya = ya + 1
            push!(televa, t)
            if fake == 1
                z_3256 = zfinal_fake(t)
                z_751 = zmeio_fake(t)
                #push!(zfake, z[nx])
                #push!(zmeiofake, z[round(Int32, (751.0/3256.0)*101.0)])
                push!(zou, z[nx] - z_3256)
                push!(zinterior, z[round(Int32, (751.0/3256.0)*101.0)] - z_751)
            else
                z_3256 = zoutlet(t)
                z_751 = zhistomedio(t)
                #push!(zfake, z[nx])
                #push!(zmeiofake, z[round(Int32, (751.0/3256.0)*101.0)])
                push!(zou, z[nx] - z_3256)
                push!(zinterior, z[round(Int32, (751.0/3256.0)*101.0)] - z_751)
            end
            #@printf "%8.2f  %8.4f  %8.4f \n" t/(60*60*24) zou[ya] zinterior[ya]
        end
      # end Writing  

      t = t + dt

      for i = 1:nx 
        
        #Consider the mass conservation equation
        if i > 1 && i < nx
            avx = (av[i+1] - av[i-1])/(2.0*dx)
        end
        if i == 1
            avx = (av[i+1] - av[i])/(dx)
        end
        if i == nx 
            avx = (av[i] - av[i-1])/(dx)
        end

        at = - avx

        anew[i] = a[i] + dt*at

        if anew[i] < 0.0
            RMSD = 10e26
            continuar = false
            break
        end

        # Consider the momentum conservation equation
        if i > 1 && i < nx 
                av2x =   (av[i+1]*v[i+1] - av[i-1]*v[i-1])/(2.0*dx)
                zhatx[i] = (z[i+1]-z[i-1])/(2.0*dx)
        end
        if i == 1 
                av2x =   (av[i+1]*v[i+1] - av[i]*v[i])/(dx)
                zhatx[i] =  (z[i+1]-z[i])/(dx)
        end
        if i == nx 
                av2x =   (av[i]*v[i] - av[i-1]*v[i-1])/(dx)
                zhatx[i] = (z[i]-z[i-1])/(dx)
        end

        zhatx[i] = zhatx[i] /(1.0 + zhatx[i]^2)
        peri = ancho[i] + 2.0*h[i]

        if n_man == 2
            # Interpolação linear simples entre os dois extremos do canal
            eneg = (1 - (i-1)/(nx-1)) * ng[1] + (i-1)/(nx-1) * ng[2]

        elseif n_man == 1
            eneg = ng[1]

        elseif n_man == nx
            # Atribuição direta: cada ponto da malha tem seu próprio n de Manning
            eneg = ng[i]

        elseif 2 < n_man < nx
            # Interpolação linear segmentada (Piecewise Linear)
            # 1. Mapeia a posição do nó 'i' para o índice fracionário no vetor 'ng'
            pos_ng = 1.0 + (i - 1) * (n_man - 1) / (nx - 1)
            
            # 2. Identifica os índices vizinhos no vetor ng
            j = floor(Int, pos_ng)
            j = clamp(j, 1, n_man - 1) # Garante que j+1 não estoure o limite
            
            # 3. Calcula a fração da distância entre o ponto j e j+1
            frac = pos_ng - j
            
            # 4. Interpola entre ng[j] e ng[j+1]
            eneg = (1.0 - frac) * ng[j] + frac * ng[j+1]

        else
            println("Wrong dimension. Please choose ng with size between 2 and nx")
        end


        rh3 = (a[i]/peri)^(4.0/3.0)
        avt =-av2x-grav*a[i]*zhatx[i]-eneg^2*av[i]*sqrt(av[i]^2+1E-3)/(rh3*a[i])
        
        avnew[i]  = av[i] + dt*avt

        if isnan(avt)
            RMSD = 10e26
            continuar = false
            break
        end

        
        if avnew[i]  == 0.0
        #  If avnew = 0 and anew not equal to zero, we would have vnew = 0.
        #  This is the reason (continuity) for stating vnew=0 whenever avnew=0
            vnew[i] = 0.0
        else
            vnew[i] = avnew[i]/anew[i]
        end
        hnew[i] = anew[i]/ancho[i]

        
        
      end # end for (nx)

      if !continuar
        break  # sai do while
      end

      avnew[1] = qinlet(t)
      vnew[1] = avnew[1]/anew[1]

      for i = 1:nx
            a[i] = anew[i]
            v[i] = vnew[i]
            h[i] = hnew[i]
            av[i] = avnew[i]
            z[i] = zb[i] + h[i]
      end


    end # end while (t)

    if !continuar
        vec_aux = fill(10e26, 2*nt)
        return vec_aux
    end

    RMSD = 0.00
    for i = 1:ya
        RMSD = RMSD + (zou[i])^2
    end
    for i = 1:ya
        RMSD = RMSD + (zinterior[i])^2
    end
    RMSD = RMSD/(2*ya)
    RMSD = sqrt(RMSD)
    #print("RMSD = ", RMSD, "\n")

    #    open("dado_fake.txt", "w") do arquivo
    #        for i = 1:ya
    #            println(arquivo, televa[i]/(60*60*24), " ",zfake[i], " ", zmeiofake[i])
    #       end
    #    end 
    
    z_error = [zou; zinterior]

    if isnan(norm(z_error, 1))
        z_error = 10e26*ones(2*nt)
    end
    return z_error

end

function _pesos_manning(n_man::Integer)
    pesos = zeros(Float64, nx, n_man)

    if n_man == 1
        pesos[:, 1] .= 1.0
    elseif n_man == 2
        for i = 1:nx
            theta = (i - 1) / (nx - 1)
            pesos[i, 1] = 1.0 - theta
            pesos[i, 2] = theta
        end
    elseif n_man == nx
        for i = 1:nx
            pesos[i, i] = 1.0
        end
    elseif 2 < n_man < nx
        for i = 1:nx
            pos_ng = 1.0 + (i - 1) * (n_man - 1) / (nx - 1)
            j = floor(Int, pos_ng)
            j = clamp(j, 1, n_man - 1)
            frac = pos_ng - j
            pesos[i, j] = 1.0 - frac
            pesos[i, j + 1] = frac
        end
    else
        error("Wrong dimension. Please choose ng with size between 1 and nx")
    end

    return pesos
end

"""
    sv_fork_derivada(ng, tin)

Retorna a derivada analítica de `sv_fork(ng, tin)` em relação a `ng`.
Cada linha da matriz corresponde a uma componente de erro retornada por
`sv_fork`; cada coluna corresponde a uma componente de `ng`.
"""
function sv_fork_derivada(ng::AbstractVector{T}, tin) where T<:Real
    alfa = 0.99
    ualfa = 1.0 - alfa
    xmax = 3256.0
    xmin = -39.0
    dt = 1.0
    dx = (xmax - xmin) / (nx - 1)
    grav = 9.8
    eps_fric = 1.0e-3
    tmix_aux = 60.0 * 60.0 * 24.0 * 3.0
    tmax = 60.0 * 60.0 * 24.0 * tin
    n_man = length(ng)
    ng_float = Float64.(ng)
    pesos_manning = _pesos_manning(n_man)

    zb = zeros(Float64, nx)
    z = zeros(Float64, nx)
    h = zeros(Float64, nx)
    av = zeros(Float64, nx)
    ancho = zeros(Float64, nx)
    a = zeros(Float64, nx)
    v = zeros(Float64, nx)

    da = zeros(Float64, nx, n_man)
    dz = zeros(Float64, nx, n_man)
    dh = zeros(Float64, nx, n_man)
    dav = zeros(Float64, nx, n_man)
    dv = zeros(Float64, nx, n_man)

    anew = zeros(Float64, nx)
    zhatx = zeros(Float64, nx)
    avnew = zeros(Float64, nx)
    vnew = zeros(Float64, nx)
    hnew = zeros(Float64, nx)

    danew = zeros(Float64, nx, n_man)
    dzhatx = zeros(Float64, nx, n_man)
    davnew = zeros(Float64, nx, n_man)
    dvnew = zeros(Float64, nx, n_man)
    dhnew = zeros(Float64, nx, n_man)

    jac_zou = Vector{Vector{Float64}}()
    jac_zinterior = Vector{Vector{Float64}}()
    idx_interior = round(Int32, (751.0 / 3256.0) * 101.0)

    for i = 1:nx
        x = xmin + dx * (i - 1)
        zb[i] = zbfork(x)
        z[i] = zfork(x)
        h[i] = z[i] - zb[i]
        av[i] = qinlet(tmin)
        ancho[i] = anchofork(x)
        a[i] = ancho[i] * h[i]
        v[i] = av[i] / a[i]
    end

    t = tmin
    ya = 0
    continuar = true

    while t <= tmax
        for i = 2:nx-1
            a[i] = alfa * a[i] + ualfa * (a[i - 1] + a[i + 1]) / 2.0
            for k = 1:n_man
                da[i, k] = alfa * da[i, k] + ualfa * (da[i - 1, k] + da[i + 1, k]) / 2.0
            end
        end

        for i = 2:nx-1
            v[i] = alfa * v[i] + ualfa * (v[i - 1] + v[i + 1]) / 2.0
            for k = 1:n_man
                dv[i, k] = alfa * dv[i, k] + ualfa * (dv[i - 1, k] + dv[i + 1, k]) / 2.0
            end
        end

        for i = 2:nx-1
            z[i] = alfa * z[i] + ualfa * (z[i - 1] + z[i + 1]) / 2.0
            for k = 1:n_man
                dz[i, k] = alfa * dz[i, k] + ualfa * (dz[i - 1, k] + dz[i + 1, k]) / 2.0
            end
        end

        for i = 2:nx-1
            h[i] = alfa * h[i] + ualfa * (h[i - 1] + h[i + 1]) / 2.0
            for k = 1:n_man
                dh[i, k] = alfa * dh[i, k] + ualfa * (dh[i - 1, k] + dh[i + 1, k]) / 2.0
            end
        end

        for i = 2:nx-1
            av[i] = alfa * av[i] + ualfa * (av[i - 1] + av[i + 1]) / 2.0
            for k = 1:n_man
                dav[i, k] = alfa * dav[i, k] + ualfa * (dav[i - 1, k] + dav[i + 1, k]) / 2.0
            end
        end

        imprim = tmix_aux + ya * timprim
        if (t >= imprim) && (t <= tmax)
            ya += 1
            push!(jac_zou, collect(view(dz, nx, :)))
            push!(jac_zinterior, collect(view(dz, idx_interior, :)))
        end

        t += dt

        fill!(danew, 0.0)
        fill!(dzhatx, 0.0)
        fill!(davnew, 0.0)
        fill!(dvnew, 0.0)
        fill!(dhnew, 0.0)

        for i = 1:nx
            if i > 1 && i < nx
                avx = (av[i + 1] - av[i - 1]) / (2.0 * dx)
            elseif i == 1
                avx = (av[i + 1] - av[i]) / dx
            else
                avx = (av[i] - av[i - 1]) / dx
            end

            at = -avx
            anew[i] = a[i] + dt * at

            if anew[i] < 0.0
                continuar = false
                break
            end

            if i > 1 && i < nx
                av2x = (av[i + 1] * v[i + 1] - av[i - 1] * v[i - 1]) / (2.0 * dx)
                zhat_bruto = (z[i + 1] - z[i - 1]) / (2.0 * dx)
            elseif i == 1
                av2x = (av[i + 1] * v[i + 1] - av[i] * v[i]) / dx
                zhat_bruto = (z[i + 1] - z[i]) / dx
            else
                av2x = (av[i] * v[i] - av[i - 1] * v[i - 1]) / dx
                zhat_bruto = (z[i] - z[i - 1]) / dx
            end

            zhatx[i] = zhat_bruto / (1.0 + zhat_bruto^2)
            fator_dzhat = (1.0 - zhat_bruto^2) / (1.0 + zhat_bruto^2)^2
            peri = ancho[i] + 2.0 * h[i]
            razao_hidraulica = a[i] / peri
            rh3 = razao_hidraulica^(4.0 / 3.0)
            eneg = 0.0
            for k = 1:n_man
                eneg += pesos_manning[i, k] * ng_float[k]
            end

            s_av = sqrt(av[i]^2 + eps_fric)
            denom_fric = rh3 * a[i]
            num_fric = eneg^2 * av[i] * s_av
            fric = num_fric / denom_fric
            avt = -av2x - grav * a[i] * zhatx[i] - fric
            avnew[i] = av[i] + dt * avt

            if isnan(avt)
                continuar = false
                break
            end

            for k = 1:n_man
                if i > 1 && i < nx
                    davx_k = (dav[i + 1, k] - dav[i - 1, k]) / (2.0 * dx)
                    dav2x_k = (dav[i + 1, k] * v[i + 1] + av[i + 1] * dv[i + 1, k] -
                                dav[i - 1, k] * v[i - 1] - av[i - 1] * dv[i - 1, k]) / (2.0 * dx)
                    dzhat_bruto_k = (dz[i + 1, k] - dz[i - 1, k]) / (2.0 * dx)
                elseif i == 1
                    davx_k = (dav[i + 1, k] - dav[i, k]) / dx
                    dav2x_k = (dav[i + 1, k] * v[i + 1] + av[i + 1] * dv[i + 1, k] -
                                dav[i, k] * v[i] - av[i] * dv[i, k]) / dx
                    dzhat_bruto_k = (dz[i + 1, k] - dz[i, k]) / dx
                else
                    davx_k = (dav[i, k] - dav[i - 1, k]) / dx
                    dav2x_k = (dav[i, k] * v[i] + av[i] * dv[i, k] -
                                dav[i - 1, k] * v[i - 1] - av[i - 1] * dv[i - 1, k]) / dx
                    dzhat_bruto_k = (dz[i, k] - dz[i - 1, k]) / dx
                end

                danew[i, k] = da[i, k] - dt * davx_k
                dzhatx[i, k] = fator_dzhat * dzhat_bruto_k

                dperi_k = 2.0 * dh[i, k]
                drazao_k = (da[i, k] * peri - a[i] * dperi_k) / peri^2
                drh3_k = (4.0 / 3.0) * razao_hidraulica^(1.0 / 3.0) * drazao_k
                deneg_k = pesos_manning[i, k]
                dnum_fric_k = 2.0 * eneg * deneg_k * av[i] * s_av +
                               eneg^2 * dav[i, k] * (s_av + av[i]^2 / s_av)
                ddenom_fric_k = drh3_k * a[i] + rh3 * da[i, k]
                dfric_k = (dnum_fric_k * denom_fric - num_fric * ddenom_fric_k) / denom_fric^2
                davt_k = -dav2x_k - grav * (da[i, k] * zhatx[i] + a[i] * dzhatx[i, k]) - dfric_k
                davnew[i, k] = dav[i, k] + dt * davt_k
            end

            if avnew[i] == 0.0
                vnew[i] = 0.0
                for k = 1:n_man
                    dvnew[i, k] = 0.0
                end
            else
                vnew[i] = avnew[i] / anew[i]
                for k = 1:n_man
                    dvnew[i, k] = (davnew[i, k] * anew[i] - avnew[i] * danew[i, k]) / anew[i]^2
                end
            end

            hnew[i] = anew[i] / ancho[i]
            for k = 1:n_man
                dhnew[i, k] = danew[i, k] / ancho[i]
            end
        end

        if !continuar
            break
        end

        avnew[1] = qinlet(t)
        for k = 1:n_man
            davnew[1, k] = 0.0
        end
        vnew[1] = avnew[1] / anew[1]
        for k = 1:n_man
            dvnew[1, k] = -avnew[1] * danew[1, k] / anew[1]^2
        end

        for i = 1:nx
            a[i] = anew[i]
            v[i] = vnew[i]
            h[i] = hnew[i]
            av[i] = avnew[i]
            z[i] = zb[i] + h[i]
            for k = 1:n_man
                da[i, k] = danew[i, k]
                dv[i, k] = dvnew[i, k]
                dh[i, k] = dhnew[i, k]
                dav[i, k] = davnew[i, k]
                dz[i, k] = dh[i, k]
            end
        end
    end

    if !continuar
        return zeros(Float64, 2 * nt, n_man)
    end

    n_obs = length(jac_zou)
    jac = zeros(Float64, 2 * n_obs, n_man)
    for i = 1:n_obs
        jac[i, :] .= jac_zou[i]
        jac[n_obs + i, :] .= jac_zinterior[i]
    end
    return jac
end



function sv_fork_beta(
    ng::AbstractVector{T};
    beta = 1.0,
    tend = 5.0,
    ) where T<:Real
    # variable
      local alfa = 0.99 # termo de difusão artificial -  alfa = 1 é sem difusao
      local ualfa = 1.0 - alfa 
      local xmax = 3256.0 # end point/station
      local xmin = -39.0 # initial point/station
      local dt = 1.0 #second(s)
      local dx = (xmax-xmin)/(nx-1) #space discretization
      local grav = 9.8 #gravitational constant 
      local x=0.0; zb = zeros(T, nx); z = zeros(T, nx); h = zeros(T, nx); av = zeros(T, nx)
      local ancho = zeros(T, nx); a = zeros(T, nx); v = zeros(T, nx)
      local t = 0.0; ya = 0; imprim = 0.0; televa = []; zou = []; zinterior = []; zfake = []; zmeiofake = []
      local at, avx; anew = zeros(T, nx); zhatx = zeros(T, nx); av2x = 0.0; peri=0.0
      local avt = 0.0; eneg = 0.0; avnew = zeros(T, nx); vnew = zeros(T, nx); hnew = zeros(T, nx)
      local tmix_aux = 60.0*60.0*24.0*3.0 # seconds / min / hours / days We have data after day 3
      local continuar = true
      local tmax = 0.0 + 60.0*0.0 + 60.0*60.0*0.0 + 60.0*60.0*24.0*tend
      local n_man = length(ng)
      local idx_751 = clamp(round(Int, 1 + (751.0 - xmin)/dx), 1, nx)
        # variable


        for i = 1:nx 
            x = xmin + dx*(i-1)
            zb[i] = zbfork(x)
            z[i] = zfork(x)
            h[i] = z[i] - zb[i]
            av[i] = qinlet(tmin)
            ancho[i] = anchofork(x)
            a[i] = ancho[i]*h[i]
            v[i] = av[i]/a[i]
        end

      #print("t/(60*60*24) zou[ya] zinterior[ya] \n")
      while t <= tmax

        a_old = copy(a)
        av_old = copy(av)
        #    Smoothing 
          for i = 2:nx-1
              a[i] = alfa*a_old[i] + ualfa*(a_old[i-1]+a_old[i+1])/2.0
              av[i] = alfa*av_old[i] + ualfa*(av_old[i-1]+av_old[i+1])/2.0
          end

          for i = 1:nx
            h[i] = a[i]/ancho[i]
            z[i] = h[i]+zb[i]
            v[i] = av[i]/a[i]
          end
  
        # End of Smoothing
  
        #   Writing
          imprim = tmix_aux + (ya)*timprim
          if (t >= imprim)  && (t <= tmax)
              ya = ya + 1
              push!(televa, t)
              if fake == 1
                  z_3256 = zfinal_fake(t)
                  z_751 = zmeio_fake(t)
                  #push!(zfake, z[nx])
                  #push!(zmeiofake, z[round(Int32, (751.0/3256.0)*101.0)])
                  push!(zou, z[nx] - z_3256)
                  push!(zinterior, z[round(Int32, (751.0/3256.0)*101.0)] - z_751)
              else
                  z_3256 = zoutlet(t)
                  z_751 = zhistomedio(t)
                  #push!(zfake, z[nx])
                  #push!(zmeiofake, z[round(Int32, (751.0/3256.0)*101.0)])
                  push!(zou, z[nx] - z_3256)
                  push!(zinterior, z[idx_751] - z_751)
              end
              #@printf "%8.2f  %8.4f  %8.4f \n" t/(60*60*24) zou[ya] zinterior[ya]
          end
        # end Writing  
  
        t = t + dt
  
        for i = 1:nx 
          
          #Consider the mass conservation equation
          if i > 1 && i < nx
              avx = (av[i+1] - av[i-1])/(2.0*dx)
          end
          if i == 1
              avx = (av[i+1] - av[i])/(dx)
          end
          if i == nx 
              avx = (av[i] - av[i-1])/(dx)
          end
  
          at = - avx
  
          anew[i] = a[i] + dt*at
  
          if anew[i] < 0.0
              RMSD = 10e26
              continuar = false
              break
          end
  
          # Consider the momentum conservation equation
          if i > 1 && i < nx 
                  av2x =   (av[i+1]*v[i+1] - av[i-1]*v[i-1])/(2.0*dx)
                  zhatx[i] = (z[i+1]-z[i-1])/(2.0*dx)
          end
          if i == 1 
                  av2x =   (av[i+1]*v[i+1] - av[i]*v[i])/(dx)
                  zhatx[i] =  (z[i+1]-z[i])/(dx)
          end
          if i == nx 
                  av2x =   (av[i]*v[i] - av[i-1]*v[i-1])/(dx)
                  zhatx[i] = (z[i]-z[i-1])/(dx)
          end
  
          zhatx[i] = zhatx[i] /(1.0 + zhatx[i]^2)
          peri = ancho[i] + 2.0*h[i]
  
          if n_man == 2
              # Interpolação linear simples entre os dois extremos do canal
              eneg = (1 - (i-1)/(nx-1)) * ng[1] + (i-1)/(nx-1) * ng[2]
  
          elseif n_man == 1
                eneg = ng[1]
          elseif n_man == nx
              # Atribuição direta: cada ponto da malha tem seu próprio n de Manning
              eneg = ng[i]
  
          elseif 2 < n_man < nx
              # Interpolação linear segmentada (Piecewise Linear)
              # 1. Mapeia a posição do nó 'i' para o índice fracionário no vetor 'ng'
              pos_ng = 1.0 + (i - 1) * (n_man - 1) / (nx - 1)
              
              # 2. Identifica os índices vizinhos no vetor ng
              j = floor(Int, pos_ng)
              j = clamp(j, 1, n_man - 1) # Garante que j+1 não estoure o limite
              
              # 3. Calcula a fração da distância entre o ponto j e j+1
              frac = pos_ng - j
              
              # 4. Interpola entre ng[j] e ng[j+1]
              eneg = (1.0 - frac) * ng[j] + frac * ng[j+1]
  
          else
              println("Wrong dimension. Please choose ng with size between 2 and nx")
          end
  
  
          rh3 = (a[i]/peri)^(4.0/3.0)
          avt =-beta*av2x-grav*a[i]*zhatx[i]-eneg^2*av[i]*sqrt(av[i]^2+1E-3)/(rh3*a[i])
          
          avnew[i]  = av[i] + dt*avt
  
          if isnan(avt)
              RMSD = 10e26
              continuar = false
              break
          end
  
          
          if avnew[i]  == 0.0
          #  If avnew = 0 and anew not equal to zero, we would have vnew = 0.
          #  This is the reason (continuity) for stating vnew=0 whenever avnew=0
              vnew[i] = 0.0
          else
              vnew[i] = avnew[i]/anew[i]
          end
          hnew[i] = anew[i]/ancho[i]
  
          
          
        end # end for (nx)
  
        if !continuar
          break  # sai do while
        end
  
        avnew[1] = qinlet(t)
        vnew[1] = avnew[1]/anew[1]
  
        for i = 1:nx
              a[i] = anew[i]
              v[i] = vnew[i]
              h[i] = hnew[i]
              av[i] = avnew[i]
              z[i] = zb[i] + h[i]
        end
  
  
      end # end while (t)
  
      if !continuar
          vec_aux = fill(10e26, 2*nt)
          return (
              erro = vec_aux,
              z = copy(z),
              a = copy(a),
              h = copy(h),
              v = copy(v),
              av = copy(av),
              t = t,
              ok = false,
          )
      end
  
      RMSD = 0.00
      for i = 1:ya
          RMSD = RMSD + (zou[i])^2
      end
      for i = 1:ya
          RMSD = RMSD + (zinterior[i])^2
      end
      RMSD = RMSD/(2*ya)
      RMSD = sqrt(RMSD)
      #print("RMSD = ", RMSD, "\n")
  
      #    open("dado_fake.txt", "w") do arquivo
      #        for i = 1:ya
      #            println(arquivo, televa[i]/(60*60*24), " ",zfake[i], " ", zmeiofake[i])
      #       end
      #    end 
      
      z_error = [zou; zinterior]
  
      if isnan(norm(z_error, 1))
          z_error = 10e26*ones(2*nt)
      end
      return (
          erro = z_error,
          z = copy(z),
          a = copy(a),
          h = copy(h),
          v = copy(v),
          av = copy(av),
          t = t,
          ok = true,
      )
  
end   


  function sv_fork_assimilation(
    ng::AbstractVector{T},
    tbeg,
    tend,
    estado0 = nothing,
    ) where T<:Real
    # variable
      local alfa = 0.99 # termo de difusão artificial -  alfa = 1 é sem difusao
      local ualfa = 1.0 - alfa 
      local xmax = 3256.0 # end point/station
      local xmin = -39.0 # initial point/station
      local dt = 1.0 #second(s)
      local dx = (xmax-xmin)/(nx-1) #space discretization
      local grav = 9.8 #gravitational constant 
      local x=0.0; zb = zeros(T, nx); z = zeros(T, nx); h = zeros(T, nx); av = zeros(T, nx)
      local ancho = zeros(T, nx); a = zeros(T, nx); v = zeros(T, nx)
      local t = 60.0*60.0*24.0*tbeg; ya = 0; imprim = 0.0; televa = []; zou = []; zinterior = []; zfake = []; zmeiofake = []
      local at, avx; anew = zeros(T, nx); zhatx = zeros(T, nx); av2x = 0.0; peri=0.0
      local avt = 0.0; eneg = 0.0; avnew = zeros(T, nx); vnew = zeros(T, nx); hnew = zeros(T, nx)
      local tmix_aux = 60.0*60.0*24.0*max(3.0, tbeg) # seconds / min / hours / days We have data after day 3
      local continuar = true
      local tmax = 0.0 + 60.0*0.0 + 60.0*60.0*0.0 + 60.0*60.0*24.0*tend
      local n_man = length(ng)
      local idx_751 = clamp(round(Int, 1 + (751.0 - xmin)/dx), 1, nx)
        # variable
        for i = 1:nx
            x = xmin + dx*(i-1)
            zb[i] = zbfork(x)
            ancho[i] = anchofork(x)
        end

        if tbeg == 0.0 || estado0 === nothing
            for i = 1:nx
                z[i] = zfork(xmin + dx*(i-1))
                h[i] = z[i] - zb[i]
                a[i] = ancho[i]*h[i]
                av[i] = qinlet(t)
                v[i] = av[i]/a[i]
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
            end
        end
      #print("t/(60*60*24) zou[ya] zinterior[ya] \n")
      while t <= tmax

  
        #    Smoothing 
          for i = 2:nx-1
              h[i] = alfa*h[i] + ualfa*(h[i-1]+h[i+1])/2.0
              av[i] = alfa*av[i] + ualfa*(av[i-1]+av[i+1])/2.0
          end

          for i = 1:nx
            z[i] = h[i]+zb[i]
            a[i] = ancho[i]*h[i]
            v[i] = av[i]/a[i]
          end
  
        # End of Smoothing
  
        #   Writing
          imprim = tmix_aux + (ya)*timprim
          if (t >= imprim)  && (t <= tmax)
              ya = ya + 1
              push!(televa, t)
              if fake == 1
                  z_3256 = zfinal_fake(t)
                  z_751 = zmeio_fake(t)
                  #push!(zfake, z[nx])
                  #push!(zmeiofake, z[round(Int32, (751.0/3256.0)*101.0)])
                  push!(zou, z[nx] - z_3256)
                  push!(zinterior, z[round(Int32, (751.0/3256.0)*101.0)] - z_751)
              else
                  z_3256 = zoutlet(t)
                  z_751 = zhistomedio(t)
                  #push!(zfake, z[nx])
                  #push!(zmeiofake, z[round(Int32, (751.0/3256.0)*101.0)])
                  push!(zou, z[nx] - z_3256)
                  push!(zinterior, z[idx_751] - z_751)
              end
              #@printf "%8.2f  %8.4f  %8.4f \n" t/(60*60*24) zou[ya] zinterior[ya]
          end
        # end Writing  
  
        t = t + dt
  
        for i = 1:nx 
          
          #Consider the mass conservation equation
          if i > 1 && i < nx
              avx = (av[i+1] - av[i-1])/(2.0*dx)
          end
          if i == 1
              avx = (av[i+1] - av[i])/(dx)
          end
          if i == nx 
              avx = (av[i] - av[i-1])/(dx)
          end
  
          at = - avx
  
          anew[i] = a[i] + dt*at
  
          if anew[i] < 0.0
              RMSD = 10e26
              continuar = false
              break
          end
  
          # Consider the momentum conservation equation
          if i > 1 && i < nx 
                  av2x =   (av[i+1]*v[i+1] - av[i-1]*v[i-1])/(2.0*dx)
                  zhatx[i] = (z[i+1]-z[i-1])/(2.0*dx)
          end
          if i == 1 
                  av2x =   (av[i+1]*v[i+1] - av[i]*v[i])/(dx)
                  zhatx[i] =  (z[i+1]-z[i])/(dx)
          end
          if i == nx 
                  av2x =   (av[i]*v[i] - av[i-1]*v[i-1])/(dx)
                  zhatx[i] = (z[i]-z[i-1])/(dx)
          end
  
          zhatx[i] = zhatx[i] /(1.0 + zhatx[i]^2)
          peri = ancho[i] + 2.0*h[i]
  
          if n_man == 2
              # Interpolação linear simples entre os dois extremos do canal
              eneg = (1 - (i-1)/(nx-1)) * ng[1] + (i-1)/(nx-1) * ng[2]
  
          elseif n_man == 1
                eneg = ng[1]
          elseif n_man == nx
              # Atribuição direta: cada ponto da malha tem seu próprio n de Manning
              eneg = ng[i]
  
          elseif 2 < n_man < nx
              # Interpolação linear segmentada (Piecewise Linear)
              # 1. Mapeia a posição do nó 'i' para o índice fracionário no vetor 'ng'
              pos_ng = 1.0 + (i - 1) * (n_man - 1) / (nx - 1)
              
              # 2. Identifica os índices vizinhos no vetor ng
              j = floor(Int, pos_ng)
              j = clamp(j, 1, n_man - 1) # Garante que j+1 não estoure o limite
              
              # 3. Calcula a fração da distância entre o ponto j e j+1
              frac = pos_ng - j
              
              # 4. Interpola entre ng[j] e ng[j+1]
              eneg = (1.0 - frac) * ng[j] + frac * ng[j+1]
  
          else
              println("Wrong dimension. Please choose ng with size between 2 and nx")
          end
  
  
          rh3 = (a[i]/peri)^(4.0/3.0)
          avt =-av2x-grav*a[i]*zhatx[i]-eneg^2*av[i]*sqrt(av[i]^2+1E-3)/(rh3*a[i])
          
          avnew[i]  = av[i] + dt*avt
  
          if isnan(avt)
              RMSD = 10e26
              continuar = false
              break
          end
  
          
          if avnew[i]  == 0.0
          #  If avnew = 0 and anew not equal to zero, we would have vnew = 0.
          #  This is the reason (continuity) for stating vnew=0 whenever avnew=0
              vnew[i] = 0.0
          else
              vnew[i] = avnew[i]/anew[i]
          end
          hnew[i] = anew[i]/ancho[i]
  
          
          
        end # end for (nx)
  
        if !continuar
          break  # sai do while
        end
  
        avnew[1] = qinlet(t)
        vnew[1] = avnew[1]/anew[1]
  
        for i = 1:nx
              a[i] = anew[i]
              v[i] = vnew[i]
              h[i] = hnew[i]
              av[i] = avnew[i]
              z[i] = zb[i] + h[i]
        end
  
  
      end # end while (t)
  
      if !continuar
          vec_aux = fill(10e26, 2*nt)
          return (
              erro = vec_aux,
              z = copy(z),
              a = copy(a),
              h = copy(h),
              v = copy(v),
              av = copy(av),
              t = t,
              ok = false,
          )
      end
  
      RMSD = 0.00
      for i = 1:ya
          RMSD = RMSD + (zou[i])^2
      end
      for i = 1:ya
          RMSD = RMSD + (zinterior[i])^2
      end
      RMSD = RMSD/(2*ya)
      RMSD = sqrt(RMSD)
      #print("RMSD = ", RMSD, "\n")
  
      #    open("dado_fake.txt", "w") do arquivo
      #        for i = 1:ya
      #            println(arquivo, televa[i]/(60*60*24), " ",zfake[i], " ", zmeiofake[i])
      #       end
      #    end 
      
      z_error = [zou; zinterior]
  
      if isnan(norm(z_error, 1))
          z_error = 10e26*ones(2*nt)
      end
      return (
          erro = z_error,
          z = copy(z),
          a = copy(a),
          h = copy(h),
          v = copy(v),
          av = copy(av),
          t = t,
          ok = true,
      )
  
  end




  function sv_plot(ng::AbstractVector{T}) where T<:Real
    # variable
      local alfa = 0.99 # termo de difusão artificial -  alfa = 1 é sem difusao
      local ualfa = 1.0 - alfa 
      local xmax = 3256.0 # end point/station
      local xmin = -39.0 # initial point/station
      local dt = 1.0 #second(s)
      local dx = (xmax-xmin)/(nx-1) #space discretization
      local grav = 9.8 #gravitational constant 
      local x=0.0; zb = zeros(T, nx); z = zeros(T, nx); h = zeros(T, nx); av = zeros(T, nx)
      local ancho = zeros(T, nx); a = zeros(T, nx); v = zeros(T, nx)
      local t = tmin; ya = 0; imprim = 0.0
      local televa = []; zfinal = []; zmeio = []; zfake = []; zmeiofake = []; zdata = []; zmeiodata = []
      local at, avx; anew = zeros(T, nx); zhatx = zeros(T, nx); av2x = 0.0; peri=0.0
      local avt = 0.0; eneg = 0.0; avnew = zeros(T, nx); vnew = zeros(T, nx); hnew = zeros(T, nx)
      local tmix_aux = 0.0 + 60.0*0.0 + 60.0*60.0*0.0 + 60.0*60.0*24.0*3.0 # seconds / min / hours / days We have data after day 3
      local continuar = true
      local n_man = length(ng)
    # variable
      for i = 1:nx 
          x = xmin + dx*(i-1)
          zb[i] = zbfork(x)
          z[i] = zfork(x)
          h[i] = z[i] - zb[i]
          av[i] = qinlet(tmin)
          ancho[i] = anchofork(x)
          a[i] = ancho[i]*h[i]
          v[i] = av[i]/a[i]
      end
      #print("t/(60*60*24) zou[ya] zinterior[ya] \n")
  
      while t <= tmax1
  
        #    Smoothing
          for i = 2:nx-1 
              a[i] = alfa*a[i] + ualfa*(a[i-1]+a[i+1])/2.0
          end
      
          for i = 2:nx-1
              v[i] = alfa*v[i] + ualfa*(v[i-1]+v[i+1])/2.0
          end
  
          for i = 2:nx-1
              z[i] = alfa*z[i] + ualfa*(z[i-1]+z[i+1])/2.0
          end
  
          for i = 2:nx-1
              h[i] = alfa*h[i] + ualfa*(h[i-1]+h[i+1])/2.0
          end
  
          for i = 2:nx-1
              av[i] = alfa*av[i] + ualfa*(av[i-1]+av[i+1])/2.0
          end
  
        # End of Smoothing
  
        #   Writing
          imprim = tmix_aux + (ya)*timprim
          if (t >= imprim)  && (t <= tmax1)
              ya = ya + 1
              push!(televa, t)
              if fake == 1
                  z_3256 = zfinal_fake(t)
                  z_751 = zmeio_fake(t)
                  #push!(zfake, z[nx])
                  #push!(zmeiofake, z[round(Int32, (751.0/3256.0)*101.0)])
                  push!(zou, z[nx] - z_3256)
                  push!(zinterior, z[round(Int32, (751.0/3256.0)*101.0)] - z_751)
              else
                  z_3256 = zoutlet(t)
                  z_751 = zhistomedio(t)
                  push!(zfinal, z[nx])
                  push!(zmeio, z[round(Int32, (751.0/3256.0)*101.0)])
                  push!(zdata, z_3256)
                  push!(zmeiodata, z_751)
              end
              #@printf "%8.2f  %8.4f  %8.4f \n" t/(60*60*24) zou[ya] zinterior[ya]
          end
        # end Writing  
  
        t = t + dt
  
        for i = 1:nx 
          
          #Consider the mass conservation equation
          if i > 1 && i < nx
              avx = (av[i+1] - av[i-1])/(2.0*dx)
          end
          if i == 1
              avx = (av[i+1] - av[i])/(dx)
          end
          if i == nx 
              avx = (av[i] - av[i-1])/(dx)
          end
  
          at = - avx
  
          anew[i] = a[i] + dt*at
  
          if anew[i] < 0.0
              RMSD = 10e26
              continuar = false
              break
          end
  
          # Consider the momentum conservation equation
          if i > 1 && i < nx 
                  av2x =   (av[i+1]*v[i+1] - av[i-1]*v[i-1])/(2.0*dx)
                  zhatx[i] = (z[i+1]-z[i-1])/(2.0*dx)
          end
          if i == 1 
                  av2x =   (av[i+1]*v[i+1] - av[i]*v[i])/(dx)
                  zhatx[i] =  (z[i+1]-z[i])/(dx)
          end
          if i == nx 
                  av2x =   (av[i]*v[i] - av[i-1]*v[i-1])/(dx)
                  zhatx[i] = (z[i]-z[i-1])/(dx)
          end
  
          zhatx[i] = zhatx[i] /(1.0 + zhatx[i]^2)
          peri = ancho[i] + 2.0*h[i]
  
          if n_man == 2
              # Interpolação linear simples entre os dois extremos do canal
              eneg = (1 - (i-1)/(nx-1)) * ng[1] + (i-1)/(nx-1) * ng[2]
  
          elseif n_man == nx
              # Atribuição direta: cada ponto da malha tem seu próprio n de Manning
              eneg = ng[i]
  
          elseif 2 < n_man < nx
              # Interpolação linear segmentada (Piecewise Linear)
              # 1. Mapeia a posição do nó 'i' para o índice fracionário no vetor 'ng'
              pos_ng = 1.0 + (i - 1) * (n_man - 1) / (nx - 1)
              
              # 2. Identifica os índices vizinhos no vetor ng
              j = floor(Int, pos_ng)
              j = clamp(j, 1, n_man - 1) # Garante que j+1 não estoure o limite
              
              # 3. Calcula a fração da distância entre o ponto j e j+1
              frac = pos_ng - j
              
              # 4. Interpola entre ng[j] e ng[j+1]
              eneg = (1.0 - frac) * ng[j] + frac * ng[j+1]
  
          else
              println("Wrong dimension. Please choose ng with size between 2 and nx")
          end
  
  
          rh3 = (a[i]/peri)^(4.0/3.0)
          avt =-av2x-grav*a[i]*zhatx[i]-eneg^2*av[i]*sqrt(av[i]^2+1e-3)/(rh3*a[i])
          
          avnew[i]  = av[i] + dt*avt
  
          if isnan(avt)
              RMSD = 10e26
              continuar = false
              break
          end
  
          
          if avnew[i]  == 0.0
          #  If avnew = 0 and anew not equal to zero, we would have vnew = 0.
          #  This is the reason (continuity) for stating vnew=0 whenever avnew=0
              vnew[i] = 0.0
          else
              vnew[i] = avnew[i]/anew[i]
          end
          hnew[i] = anew[i]/ancho[i]
  
          
          
        end # end for (nx)
  
        if !continuar
          break  # sai do while
        end
  
        avnew[1] = qinlet(t)
        vnew[1] = avnew[1]/anew[1]
  
        for i = 1:nx
              a[i] = anew[i]
              v[i] = vnew[i]
              h[i] = hnew[i]
              av[i] = avnew[i]
              z[i] = zb[i] + h[i]
        end
  
  
      end # end while (t)
  
      televa .= televa./(24.0*60*60) 
      local pl1 = plot(televa, zdata, layout=(2,1), label = "Data")
      plot!(televa, zfinal, label = "S-V")
      plot!(televa, zmeiodata, subplot=2, label = "Data")
      plot!(televa, zmeio, subplot=2, label = "S-V")


      pgfplotsx() 

      # Definir passo para amostragem
        step = 20
        
        # BOBYQA - coluna esquerda
        indices = 1:step:length(zmeio)
    # Segunda linha - m=3256
        local pl2 = plot(televa, zmeio,
            label = "", linewidth = 2, color = :cornflowerblue, 
            xlims = (2, 31),
            ylims = (5, 9.1))
        
        plot!(televa[1:2, 1], zmeio[1:2],
            label = "m=751", linewidth = 2, markershape = :utriangle,
            markerstrokewidth = 0, color = :cornflowerblue, legend = (0.0, 0.6))

        # Marcadores para segunda linha
        scatter!(televa[indices], zmeio[indices],
            label = "",
            markershape = :utriangle,
            markersize = 4,
            markercolor = :cornflowerblue, markerstrokewidth = 0)

        # Linha sem marcadores
        plot!(televa, zfinal,
            label = "",
            legend = (0.0, 0.6),
            title = "",
            ylabel = "m",
            xlabel = "t", color = :red)

        plot!(televa[1:2], zfinal[1:2],
            label = "m=3256", linewidth = 2,
            legend = (0.0, 0.6),
            markershape = :x,
            markersize = 4,
            color = :red,
            alpha = 1.0)

        # Marcadores espaçados manualmente
        scatter!(televa[indices], zfinal[indices],
            label = "",  # Label vazio para não duplicar
            markershape = :x,
            markersize = 4,
            markercolor = :red,
            markerstrokewidth = 1.5)


        # Dados (mantém todos os pontos)
        scatter!(televa, zdata,
            label = "Data", markersize = 1.0, color = :black, legend = (0.0, 0.6),
            )

        scatter!(televa, zmeiodata,
            label = "", markersize = 1.0, color = :black)
  
      return pl2
      # return televa, zdata, zfinal, zmeiodata, zmeio
  
  end
