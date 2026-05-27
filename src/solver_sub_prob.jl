using ForwardDiff, LinearAlgebra

function bb_nonmonotone(
    f,
    x0::AbstractVector{T};
    max_iter=10,
    tol=1e-6,
    m=10,
    c1=1e-4,
    alpha_min=1e-5,
    alpha_max=1e5,
    ) where T<:Real
    n = length(x0)

    x_k = copy(x0)
    x_prev = copy(x_k)

    cfg = ForwardDiff.GradientConfig(f, x_k, ForwardDiff.Chunk{n}())
    g_k = ForwardDiff.gradient(f, x_k, cfg)
    f_k = f(x_k)
    g_prev = copy(g_k)
    f_hist = fill(f_k, m)
    alpha = one(T)

    for k in 1:max_iter
        if norm(g_k) < tol
            println("Convergiu na iteração $k")
            return x_k
        end

        d_k = -g_k

        if k > 1
            s = x_k - x_prev
            y = g_k - g_prev
            sy = dot(s, y)
            alpha = sy > 1e-16 ? dot(s, s) / sy : 1.0
        end

        alpha = clamp(alpha, alpha_min, alpha_max)
        f_ref = maximum(f_hist)
        alpha_k = alpha
        max_backtrack = 20
        j = 0

        while j < max_backtrack
            x_new = x_k + alpha_k * d_k
            f_new = f(x_new)

            if f_new <= f_ref + c1 * alpha_k * dot(g_k, d_k)
                break
            end

            alpha_k *= 0.5
            j += 1
        end

        if j == max_backtrack
            println("Falha no line search; usando fallback.")
            alpha_k = 1e-6
            x_new = x_k + alpha_k * d_k
            f_new = f(x_new)
        end

        f_hist = vcat(f_hist[2:end], f_new)
        x_prev = copy(x_k)
        g_prev = copy(g_k)
        x_k = x_new
        g_k = ForwardDiff.gradient(f, x_k, cfg)

        if k % 10 == 0
            println("Iter $k: f = $(round(f_k, digits=6)), ||g|| = $(round(norm(g_k), digits=6))")
        end

        f_k = f_new
    end

    println("Máximo de iterações atingido")
    return x_k
end

function spectral_corrected(
    f,
    g,
    x0::AbstractVector{T};
    max_iter=100,
    tol=1e-6,
    epsilon=1e-8,
    lb=-1000.0,
    ub=1000.0,
    lambda_min=1e-10,
    lambda_max=1e10,
    alpha_min=1e-12,
    max_backtrack=20,
    c1=1e-4,
    ) where T<:Real
    x_k = clamp.(copy(x0), lb, ub)
    x_k1 = copy(x_k)

    f_k = f(x_k)
    g_k = g(x_k)
    g_k1 = copy(g_k)
    lambda = one(T)

    for iter in 1:max_iter
        s = x_k - x_k1
        y = g_k - g_k1
        sy = dot(s, y)

        if iter > 1 && sy > 1e-16
            lambda = dot(s, s) / sy
        else
            lambda = 1.0 / max(1.0, norm(g_k))
        end

        lambda = clamp(lambda, lambda_min, lambda_max)
        d = clamp.(x_k .- lambda .* g_k, lb, ub) .- x_k
        norm_dir = norm(d)

        if norm_dir < tol
            println("Convergiu com sucesso na iteração $(iter)!")
            return x_k
        end

        alpha = one(T)
        x_new = x_k .+ alpha .* d
        f_new = f(x_new)
        j = 0

        while f_new > f_k + c1 * alpha * dot(g_k, d) && j < max_backtrack
            alpha *= T(0.5)
            if alpha < alpha_min
                break
            end

            x_new = x_k .+ alpha .* d
            f_new = f(x_new)

            if isnan(f_new)
                f_new = T(1e27)
            end
            j += 1
        end

        if j == max_backtrack || alpha < alpha_min
            println("Aviso: número máximo de backtracking atingido na iteração $iter.")
            return x_k
        end

        if iter % 10 == 0
            println("Iteração $iter: f = $(round(f_k, digits=6)), ||d_proj|| = $(round(norm_dir, digits=6))")
        end

        if abs(f_new - f_k) < epsilon
            println("Parada por estagnação na iteração $iter.")
            return x_new
        end

        x_k1 .= x_k
        x_k .= x_new
        f_k = f_new
        g_k1 .= g_k
        g_k = g(x_k)
    end

    println("Máximo de iterações atingido sem convergência estrita.")
    return x_k
end

function powell_damped_bfgs(f, x0::AbstractVector{T}; max_iter=100, tol=1e-6) where T<:Real
    n = length(x0)
    
    # InicializaÃ§Ã£o
    x_k = copy(x0)
    f_k = f(x_k)
    
    # Matriz Hessiana aproximada B (Inicia como Matriz Identidade)
    B = Matrix{T}(I, n, n)
    
    # ConfiguraÃ§Ã£o do Gradiente
    cfg = ForwardDiff.GradientConfig(nothing, x_k, ForwardDiff.Chunk{n}())
    g_k = ForwardDiff.gradient(f, x_k, cfg)
    
    for iter in 1:max_iter
        # 1. DIREÃ‡ÃƒO DE BUSCA (B * d = -g)
        # O operador \ resolve o sistema linear de forma eficiente no Julia
        d = -(B \ g_k)
        
        # 2. BUSCA LINEAR DE POWELL (Sempre tenta dar o passo inteiro primeiro)
        alpha = 1.0
        c1 = 1e-4
        max_backtrack = 20
        j = 0
        
        x_new = x_k .+ alpha .* d
        f_new = f(x_new)
        
        # Backtracking (CondiÃ§Ã£o de Armijo)
        # Reduz o passo apenas se a funÃ§Ã£o nÃ£o diminuir o suficiente
        while f_new > f_k + c1 * alpha * dot(g_k, d) && j < max_backtrack
            alpha *= 0.5 # Powell sugere cortar o passo (pode ser 0.5 ou interpolaÃ§Ã£o)
            x_new = x_k .+ alpha .* d
            f_new = f(x_new)
            
            if isnan(f_new)
                f_new = T(1e27) # ProteÃ§Ã£o contra erro na simulaÃ§Ã£o sv_fork
            end
            j += 1
        end
        
        # 3. ATUALIZAÃ‡ÃƒO DOS VETORES s E y
        s = x_new - x_k
        g_new = ForwardDiff.gradient(f, x_new, cfg)
        y = g_new - g_k
        
        # Verifica convergÃªncia antes de atualizar a matriz
        if norm(g_new) < tol || norm(s) < tol
            println("Convergiu com sucesso na iteraÃ§Ã£o $iter ")
            return x_new
        end
        
        # ==========================================================
        # 4. A MÃGICA DE POWELL: O FATOR DE AMORTECIMENTO (DAMPING)
        # ==========================================================
        Bs = B * s
        sBs = dot(s, Bs)
        sy = dot(s, y)
        
        # Regra de Powell: Verifica se a curvatura Ã© perigosa
        if sy >= 0.2 * sBs
            theta = 1.0 # Tudo seguro, usa o BFGS normal
        else
            # Calcula o amortecimento para garantir B definida positiva
            theta = (0.8 * sBs) / (sBs - sy)
        end
        
        # Cria o vetor modificado (r) misturando y e B*s
        r = theta .* y .+ (1.0 - theta) .* Bs
        
        # Atualiza a Matriz B usando a fÃ³rmula clÃ¡ssica, mas com 'r' no lugar de 'y'
        # B_{k+1} = B_k - (B_k s s^T B_k)/(s^T B_k s) + (r r^T)/(s^T r)
        B = B - (Bs * Bs') / sBs + (r * r') / dot(s, r)
        
        # ==========================================================
        
        # Atualiza variÃ¡veis para a prÃ³xima iteraÃ§Ã£o
        x_k .= x_new
        f_k = f_new
        g_k .= g_new
        
        # Monitoramento
        if iter % 1 == 0
            println("Iter $iter: f = $(round(f_k, digits=6)), ||âˆ‡f|| = $(round(norm(g_k), digits=6)), Passo Î± = $(round(alpha, digits=4)), Amortecimento Î¸ = $(round(theta, digits=4))")
        end
    end
    
    println("Aviso: MÃ¡ximo de iteraÃ§Ãµes atingido.")
    return x_k
end




function bfgs_caixa(f, gfun, x0, lb, ub; tol=1e-6, max_iter=100)
    n = length(x0)
    # Garante que o ponto inicial estÃ¡ dentro da caixa
    x = clamp.(float.(x0), lb, ub)
    
    
    # InicializaÃ§Ã£o
    H = I(n) * 1.0
    fx = f(x)  # Cache do valor inicial da funÃ§Ã£o
    g = gfun(x)
    
    for k in 1:max_iter
        # CritÃ©rio de parada: Norma do gradiente projetado
        # (Se o gradiente aponta para fora mas estamos no limite, paramos)
        gnorm = norm(g)
        print("gnorm = ", gnorm, "\n")
        if gnorm < tol
            println("ConvergÃªncia atingida na iteraÃ§Ã£o $k (norma g: $gnorm)")
            return x
        end
        
        # 1. DireÃ§Ã£o de busca
        p = -H * g
               
        # 2. Backtracking Line Search Otimizado
        alpha = 1.0
        c1 = 1e-4
        rho = 0.5
        alpha_min = 1e-5
        
        descida_esperada = c1 * (g' * p)
        
        x_proximo = clamp.(x + alpha * p, lb, ub)
        f_proximo = f(x_proximo)
        
        while f_proximo > fx + alpha * descida_esperada
            alpha *= rho
            
            # Print solicitado: alpha e o maior valor de x_proximo
            # Usamos ForwardDiff.value para garantir que o print ignore tipos Dual
            val_max_x = ForwardDiff.value(maximum(x_proximo))
            println("Busca Linear: alpha = $alpha | max(x_novo) = $val_max_x \n")
            
            if alpha < alpha_min
                alpha = 0.0
                x_proximo = x
                f_proximo = fx
                break
            end
            
            x_proximo = clamp.(x + alpha * p, lb, ub)
            f_proximo = f(x_proximo)
        end
        
        # Se nÃ£o houve movimento, encerramos
        if alpha == 0.0
            println("Busca linear falhou (passo muito pequeno) na iteraÃ§Ã£o $k")
            break
        end

        # 3. PreparaÃ§Ã£o para atualizaÃ§Ã£o BFGS
        s = x_proximo - x
        g_proximo = gfun(x_proximo)
        y = g_proximo - g
        
        # 4. AtualizaÃ§Ã£o da Inversa da Hessiana (FÃ³rmula de Sherman-Morrison)
        sy = y' * s
        if sy > 1e-10 * norm(s) * norm(y) # Estabilidade numÃ©rica
            r = 1.0 / sy
            V = I(n) - r * s * y'
            H = V * H * V' + r * s * s'
        end
        
        # 5. Atualiza variÃ¡veis para a prÃ³xima iteraÃ§Ã£o
        x = x_proximo
        fx = f_proximo  # O valor calculado no line search vira o novo fx
        g = g_proximo   # O gradiente calculado vira o novo g
    end
    
    return x
end

_valor_finito(v; limite=1e26) = begin
    vv = ForwardDiff.value(v)
    isfinite(vv) && abs(vv) < limite
end

_gradiente_projetado_norma(x, g, lb, ub) = norm(x .- clamp.(x .- g, lb, ub))

function powell_bfgs(f, gfun, x0::AbstractVector{T};
                     tol=1e-6,
                     max_iter=100,
                     c1=1e-4,
                     c2=0.9,
                     alpha0=1.0,
                     alpha_min=1e-12,
                     alpha_max=1e6,
                     max_linesearch=30,
                     damping_delta=0.2,
                     penalidade=1e26,
                     verbose=true) where T<:Real
    n = length(x0)
    x = copy(float.(x0))
    fx = f(x)
    g = gfun(x)
    B = Matrix{eltype(x)}(I, n, n)

    if !_valor_finito(fx; limite=penalidade)
        error("powell_bfgs: ponto inicial tem valor invalido: $fx")
    end

    for k in 1:max_iter
        gnorm = norm(g)
        if verbose
            println("Powell-BFGS iter $k | f = $(ForwardDiff.value(fx)) | ||g|| = $gnorm")
        end
        if gnorm < tol
            println("Convergencia atingida na iteracao $k")
            return x
        end

        # Direcao de Newton aproximada. Se perder descida, reinicia a metrica.
        d = -(B \ g)
        gtd = dot(g, d)
        if !isfinite(ForwardDiff.value(gtd)) || gtd >= 0
            B .= Matrix{eltype(x)}(I, n, n)
            d = -g
            gtd = dot(g, d)
        end

        # Busca linear inexata do tipo Wolfe-Powell:
        # Armijo para descida suficiente e Wolfe fraca para curvatura.
        alpha = alpha0
        alpha_low = zero(alpha)
        alpha_high = Inf
        x_new = similar(x)
        f_new = fx
        g_new = similar(g)
        aceito = false

        for ls in 1:max_linesearch
            x_trial = x .+ alpha .* d
            f_trial = f(x_trial)

            if !_valor_finito(f_trial; limite=penalidade) ||
               f_trial > fx + c1 * alpha * gtd
                alpha_high = alpha
                alpha = isfinite(alpha_high) ? 0.5 * (alpha_low + alpha_high) : 0.5 * alpha
            else
                g_trial = gfun(x_trial)
                if !_valor_finito(norm(g_trial); limite=penalidade)
                    alpha_high = alpha
                    alpha = isfinite(alpha_high) ? 0.5 * (alpha_low + alpha_high) : 0.5 * alpha
                elseif dot(g_trial, d) >= c2 * gtd
                    x_new .= x_trial
                    f_new = f_trial
                    g_new .= g_trial
                    aceito = true
                    break
                else
                    alpha_low = alpha
                    alpha = isfinite(alpha_high) ? 0.5 * (alpha_low + alpha_high) : min(2.0 * alpha, alpha_max)
                end
            end

            if alpha < alpha_min || alpha > alpha_max
                break
            end
        end

        if !aceito
            println("Busca linear Wolfe-Powell falhou na iteracao $k; retornando ultimo ponto valido.")
            return x
        end

        s = x_new - x
        y = g_new - g

        # Amortecimento de Powell: troca y por r quando a curvatura e fraca.
        Bs = B * s
        sBs = dot(s, Bs)
        sy = dot(s, y)

        if sBs <= eps(eltype(x)) * max(1.0, norm(s)^2)
            B .= Matrix{eltype(x)}(I, n, n)
        else
            theta = sy >= damping_delta * sBs ? one(eltype(x)) :
                    ((1.0 - damping_delta) * sBs) / (sBs - sy)
            r = theta .* y .+ (1.0 - theta) .* Bs
            sr = dot(s, r)

            if sr > eps(eltype(x)) * norm(s) * norm(r)
                B .= B .- (Bs * Bs') ./ sBs .+ (r * r') ./ sr
            else
                B .= Matrix{eltype(x)}(I, n, n)
            end
        end

        x .= x_new
        fx = f_new
        g .= g_new
    end

    println("Maximo de iteracoes atingido no Powell-BFGS.")
    return x
end


# # FunÃ§Ãµes de teste
# function quadratic(x)
#     return x[1]^2 + x[2]^2
# end

# function rosenbrock(x)
#     return (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
# end

# # Teste 1: FunÃ§Ã£o quadrÃ¡tica simples
# println("=== Teste 1: FunÃ§Ã£o quadrÃ¡tica ===")
# x0 = [1.0, 1.0]
# result1 = spectral_corrected(quadratic, x0, max_iter=50)
# println("Ponto inicial: $x0")
# println("MÃ­nimo encontrado: $result1")
# println("Valor da funÃ§Ã£o: $(quadratic(result1))")
# println()

# # Teste 2: FunÃ§Ã£o Rosenbrock
# println("=== Teste 2: FunÃ§Ã£o Rosenbrock ===")
# x0_ros = [0.0, 0.0]
# result2 = spectral_corrected(rosenbrock, x0_ros, max_iter=200)
# println("Ponto inicial: $x0_ros")
# println("MÃ­nimo encontrado: $result2")
# println("Valor da funÃ§Ã£o: $(rosenbrock(result2))")
