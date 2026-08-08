using ForwardDiff
using LinearAlgebra
using Printf

# ==============================================================================
# Tradução para Julia de `sr1gemini.for` e `bfgsgemini.for`: métodos quase-
# Newton SR1 e BFGS com busca linear de Armijo (backtracking) e teste de
# direção de descida (substitui a direção por -grad quando ela não é de
# descida). A tradução é literal: mantém a mesma sequência de operações,
# incluindo o critério de parada por número de avaliações da função.
#
# Nota: no Fortran original (`sr1gemini.for`), o teste de limite de
# avaliações da função dentro do backtracking do SR1 compara `nef` com
# `maxit` (e não com `maxnef`, como faz a versão BFGS). Esse comportamento é
# preservado abaixo por fidelidade à tradução.
# ==============================================================================

# ------------------------------------------------------------------------------
# Função objetivo e gradiente de teste (Rosenbrock 2D)
# ------------------------------------------------------------------------------

rosenbrock(x::AbstractVector) = 100.0 * (x[2] - x[1]^2)^2 + (1.0 - x[1])^2

function rosenbrock_grad(x::AbstractVector)
    g1 = -400.0 * x[1] * (x[2] - x[1]^2) - 2.0 * (1.0 - x[1])
    g2 = 200.0 * (x[2] - x[1]^2)
    return [g1, g2]
end

# ------------------------------------------------------------------------------
# Busca linear alternativa: backtracking com interpolação quadrática.
#
# Não é usada por `sr1_backtracking`/`bfgs_backtracking` abaixo — esses
# fazem o backtracking geométrico simples do Fortran original (alpha *= 0.5
# a cada falha do teste de Armijo). Fica guardada aqui, autocontida, para uso
# futuro em outros métodos.
# ------------------------------------------------------------------------------

"""
    quadratic_step_alpha(alpha0, phi0, dphi0, phi_alpha0; rho_lo=0.1, rho_hi=0.5)

Dado que o passo `alpha0` já foi tentado (`phi_alpha0 = ϕ(alpha0) = f(x + alpha0*p)`)
e que se conhecem `ϕ(0) = phi0` e `ϕ'(0) = dphi0 = grad f(x)ᵀp`, interpola
esses três valores pela parábola

    q(α) = ϕ(0) + ϕ'(0)·α + c·α², com c = (ϕ(alpha0) - ϕ(0) - alpha0·ϕ'(0)) / alpha0²,

e devolve o minimizador dessa parábola,

    α_quad = -ϕ'(0)·alpha0² / (2·(ϕ(alpha0) - ϕ(0) - alpha0·ϕ'(0))).

Esse é o mesmo princípio da primeira interpolação de
`LineSearches.BackTracking(order = 3)` (ou de `QuadraticBacktracking` em
`aux_func.jl`): em vez de simplesmente reduzir `alpha0` por um fator fixo,
usa-se a curvatura aparente de `ϕ` para estimar diretamente onde ela é
mínima. O resultado é limitado a `[rho_lo, rho_hi] .* alpha0` (mesma
salvaguarda do `LineSearches.jl`) para não gerar passos próximos demais de
`0` ou de `alpha0`. Devolve `nothing` quando a parábola não é utilizável
(denominador não positivo — o que não deveria ocorrer quando `dphi0 < 0` e o
teste de Armijo falhou em `alpha0` — ou resultado não finito); nesse caso o
chamador deve recorrer ao backtracking geométrico simples.
"""
function quadratic_step_alpha(
    alpha0::Real, phi0::Real, dphi0::Real, phi_alpha0::Real;
    rho_lo::Real = 0.1, rho_hi::Real = 0.5,
)
    denom = phi_alpha0 - phi0 - dphi0 * alpha0
    denom > 0 || return nothing
    alpha_quad = -dphi0 * alpha0^2 / (2 * denom)
    isfinite(alpha_quad) || return nothing
    return clamp(alpha_quad, rho_lo * alpha0, rho_hi * alpha0)
end

"""
    quadratic_backtracking(f, x, p, fval, gdotp; alpha0=1.0, c1=1e-4, rho=0.5, quad_fmax=1000.0, quad_rho_lo=0.1, quad_rho_hi=0.5, min_alpha=1e-12, max_backtracks=50)

Busca linear por backtracking de Armijo com interpolação quadrática no
primeiro retrocesso, usando [`quadratic_step_alpha`](@ref). A partir de `x`,
direção `p`, `fval = f(x)` e `gdotp = grad f(x)ᵀp`, tenta `alpha = alpha0`;
se o teste de Armijo falhar, o primeiro retrocesso escolhe o próximo `alpha`
pela parábola interpolada — mas somente quando `f(x + alpha0*p)` é
"aceitável" (finito e menor que `quad_fmax`); caso contrário, e em
retrocessos subsequentes, `alpha` é simplesmente multiplicado por `rho`.

Retorna `(; alpha, xnext, fnext, nef)`, onde `nef` é o número de avaliações
de `f` realizadas. Se `max_backtracks` for atingido ou `alpha` cair abaixo de
`min_alpha` sem satisfazer Armijo, devolve mesmo assim o último par
`(alpha, xnext, fnext)` tentado — cabe ao chamador decidir como tratar essa
falha.
"""
function quadratic_backtracking(
    f, x::AbstractVector, p::AbstractVector, fval::Real, gdotp::Real;
    alpha0::Real = 1.0,
    c1::Real = 1.0e-4,
    rho::Real = 0.5,
    quad_fmax::Real = 1000.0,
    quad_rho_lo::Real = 0.1,
    quad_rho_hi::Real = 0.5,
    min_alpha::Real = 1.0e-12,
    max_backtracks::Integer = 50,
)
    alpha = float(alpha0)
    nef = 0
    xnext = similar(x)
    fnext = fval
    first_backtrack = true

    for _ in 1:max_backtracks
        xnext = x .+ alpha .* p
        fnext = f(xnext)
        nef += 1

        if fnext <= fval + c1 * alpha * gdotp
            return (; alpha, xnext, fnext, nef)
        end

        alpha_quad = first_backtrack && isfinite(fnext) && fnext < quad_fmax ?
            quadratic_step_alpha(alpha, fval, gdotp, fnext; rho_lo = quad_rho_lo, rho_hi = quad_rho_hi) :
            nothing
        alpha = alpha_quad === nothing ? alpha * rho : alpha_quad
        first_backtrack = false

        alpha > min_alpha || break
    end

    return (; alpha, xnext, fnext, nef)
end

# ==============================================================================
# SR1 com backtracking (Armijo) e teste de direção de descida
# ==============================================================================

"""
    sr1_backtracking(f, grad, x0, tol, maxit, maxnef; show_trace=false)

Tradução da subrotina Fortran `SR1_BACKTRACKING`. `f(x)` retorna o valor da
função objetivo e `grad(x)` o vetor gradiente. `H` (inversa aproximada da
Hessiana) é iniciada como a identidade e atualizada pela fórmula SR1, com
salvaguarda `|yᵀu| ≥ r·‖y‖·‖u‖`.

Se `show_trace = true`, imprime a cada iteração o número da iteração, `f`,
`‖grad f‖`, o passo `alpha` aceito pelo backtracking e se a atualização SR1
de `H` foi aceita pela salvaguarda ou rejeitada.

Retorna `(x, info, kon, nef, neg, notdes)`, onde `info` vale `0` se convergiu
(`‖grad f‖ < tol`), `1` se atingiu `maxit` iterações, ou `2` se atingiu o
limite de avaliações da função dentro do backtracking.
"""
function sr1_backtracking(
    f, grad, x0::AbstractVector, tol::Real, maxit::Integer, maxnef::Integer;
    show_trace::Bool = false,
)
    n = length(x0)
    x = collect(float.(x0))
    H = Matrix{Float64}(I, n, n)

    c1 = 1.0e-4
    rho = 0.5
    r = 1.0e-8

    kon = 0
    nef = 0
    neg = 0
    notdes = 0

    fval = 0.0
    fnext = 0.0
    g = zeros(n)
    gnext = zeros(n)

    for _ in 1:maxit
        kon += 1
        if kon == 1
            fval = f(x)
            nef += 1
            g = grad(x)
            neg += 1
        else
            fval = fnext
            g = copy(gnext)
        end

        gnorm = norm(g)
        if gnorm < tol
            show_trace && @printf(
                "iter %4d | f = %.6e | ||g|| = %.3e | convergiu (||g|| < tol)\n",
                kon, fval, gnorm,
            )
            return (x = x, info = 0, kon = kon, nef = nef, neg = neg, notdes = notdes)
        end

        # Direção quasi-Newton: p = -H*g
        p = -H * g

        # Teste de direção de descida: g'*p < 0
        gdotp = dot(g, p)
        if gdotp >= 0.0
            p = -g
            H = Matrix{Float64}(I, n, n)
            notdes += 1
            gdotp = -gnorm^2
        end

        # Busca linear (Armijo backtracking)
        alpha = 1.0
        xnext = similar(x)
        while true
            xnext = x .+ alpha .* p
            fnext = f(xnext)
            nef += 1
            if nef >= maxit
                show_trace && @printf(
                    "iter %4d | f = %.6e | ||g|| = %.3e | alpha = %.3e | limite de avaliações (nef >= maxit) atingido\n",
                    kon, fval, gnorm, alpha,
                )
                return (x = x, info = 2, kon = kon, nef = nef, neg = neg, notdes = notdes)
            end
            if fnext > fval + c1 * alpha * gdotp
                alpha *= rho
                alpha > 1.0e-12 && continue
            end
            break
        end

        # s = alpha*p, y = g_next - g
        s = alpha .* p
        x = xnext
        gnext = grad(x)
        neg += 1
        y = gnext .- g

        # Atualização SR1: H_next = H + (u*u')/(y'*u), u = s - H*y
        Hy = H * y
        u = s .- Hy

        ysu = dot(y, u)
        su_norm = norm(u)
        y_norm = norm(y)

        aceito = abs(ysu) >= r * y_norm * su_norm
        if aceito
            H = H .+ (u * u') ./ ysu
        end

        show_trace && @printf(
            "iter %4d | f = %.6e | ||g|| = %.3e | alpha = %.3e | atualização SR1: %s\n",
            kon, fval, gnorm, alpha, aceito ? "aceita" : "rejeitada",
        )
    end

    return (x = x, info = 1, kon = kon, nef = nef, neg = neg, notdes = notdes)
end

# ==============================================================================
# BFGS com backtracking (Armijo) e teste de direção de descida
# ==============================================================================

"""
    bfgs_backtracking(f, grad, x0, tol, maxit, maxnef; show_trace=false)

Tradução da subrotina Fortran `BFGS_BACKTRACKING`. `f(x)` retorna o valor da
função objetivo e `grad(x)` o vetor gradiente. `H` (inversa aproximada da
Hessiana) é iniciada como a identidade e atualizada pela fórmula BFGS direta
para `H`, com salvaguarda `yᵀs > 1e-10`.

Se `show_trace = true`, imprime a cada iteração o número da iteração, `f`,
`‖grad f‖`, o passo `alpha` aceito pelo backtracking e se a atualização BFGS
de `H` foi aceita pela salvaguarda ou rejeitada.

Retorna `(x, info, kon, nef, neg, notdes)`, onde `info` vale `0` se convergiu
(`‖grad f‖ < tol`), `1` se atingiu `maxit` iterações, ou `2` se atingiu
`maxnef` avaliações da função.
"""
function bfgs_backtracking(
    f, grad, x0::AbstractVector, tol::Real, maxit::Integer, maxnef::Integer;
    show_trace::Bool = false,
)
    n = length(x0)
    x = collect(float.(x0))
    H = Matrix{Float64}(I, n, n)

    c1 = 1.0e-4
    rho = 0.5

    kon = 0
    nef = 0
    neg = 0
    notdes = 0

    fval = 0.0
    g = zeros(n)

    for _ in 1:maxit
        kon += 1
        if kon == 1
            g = grad(x)
            neg += 1
            fval = f(x)
            nef += 1
        end

        gnorm = norm(g)
        if gnorm < tol
            show_trace && @printf(
                "iter %4d | f = %.6e | ||g|| = %.3e | convergiu (||g|| < tol)\n",
                kon, fval, gnorm,
            )
            return (x = x, info = 0, kon = kon, nef = nef, neg = neg, notdes = notdes)
        end
        fval_start = fval

        # Direção p = -H*g
        p = -H * g

        # Teste de direção de descida: g'*p < 0
        gdotp = dot(g, p)
        if gdotp >= 0.0
            notdes += 1
            p = -g
            H = Matrix{Float64}(I, n, n)
            gdotp = -gnorm^2
        end

        # Busca linear (Armijo backtracking)
        alpha = 1.0
        fnext = 0.0
        xnext = similar(x)
        while true
            xnext = x .+ alpha .* p
            fnext = f(xnext)
            nef += 1
            if nef >= maxnef
                show_trace && @printf(
                    "iter %4d | f = %.6e | ||g|| = %.3e | alpha = %.3e | limite de avaliações (nef >= maxnef) atingido\n",
                    kon, fval_start, gnorm, alpha,
                )
                return (x = x, info = 2, kon = kon, nef = nef, neg = neg, notdes = notdes)
            end
            if fnext > fval + c1 * alpha * gdotp
                alpha *= rho
                alpha > 1.0e-12 && continue
            end
            break
        end

        fval = fnext
        # s = alpha*p
        s = alpha .* p
        x = xnext

        gnext = grad(x)
        neg += 1
        y = gnext .- g
        g = gnext

        # Atualização BFGS direta de H:
        # H = H + r*(1 + r*(y'*H*y))*(s*s') - r*(s*(H*y)' + (H*y)*s'), r = 1/(y'*s)
        ys = dot(y, s)
        aceito = ys > 1.0e-10
        if aceito
            r = 1.0 / ys
            Hy = H * y
            dotp = dot(y, Hy)
            H = H .+ (r * (1.0 + r * dotp)) .* (s * s') .- r .* (s * Hy' .+ Hy * s')
        end

        show_trace && @printf(
            "iter %4d | f = %.6e | ||g|| = %.3e | alpha = %.3e | atualização BFGS: %s\n",
            kon, fval_start, gnorm, alpha, aceito ? "aceita" : "rejeitada",
        )
    end

    return (x = x, info = 1, kon = kon, nef = nef, neg = neg, notdes = notdes)
end

# ==============================================================================
# Demonstração equivalente aos PROGRAM TEST_SR1 / TEST_BFGS originais
# ==============================================================================

"""
    teste_sr1_bfgs(; show_trace=true)

Executa SR1 e BFGS com backtracking sobre a função de Rosenbrock 2D a partir
de `x0 = (-1.2, 1.0)`, imprimindo um resumo equivalente aos programas
Fortran `sr1gemini.for` e `bfgsgemini.for`.
"""
function teste_sr1_bfgs(; show_trace::Bool = true)
    x0 = [-1.2, 1.0]
    tol = 1.0e-6
    maxit = 500
    maxnef = 10 * maxit

    println("=========================================")
    println("   TESTE SR1 + BACKTRACKING             ")
    println("=========================================")
    resultado_sr1 = sr1_backtracking(rosenbrock, rosenbrock_grad, x0, tol, maxit, maxnef; show_trace)
    println("Solução: x = ", resultado_sr1.x, ", f(x) = ", rosenbrock(resultado_sr1.x))
    println("info = ", resultado_sr1.info, ", iterações = ", resultado_sr1.kon,
        ", avaliações de f = ", resultado_sr1.nef, ", avaliações de grad = ", resultado_sr1.neg,
        ", substituições por -g = ", resultado_sr1.notdes)

    println("=========================================")
    println("   TESTE BFGS + BACKTRACKING             ")
    println("=========================================")
    resultado_bfgs = bfgs_backtracking(rosenbrock, rosenbrock_grad, x0, tol, maxit, maxnef; show_trace)
    println("Solução: x = ", resultado_bfgs.x, ", f(x) = ", rosenbrock(resultado_bfgs.x))
    println("info = ", resultado_bfgs.info, ", iterações = ", resultado_bfgs.kon,
        ", avaliações de f = ", resultado_bfgs.nef, ", avaliações de grad = ", resultado_bfgs.neg,
        ", substituições por -g = ", resultado_bfgs.notdes)

    return (sr1 = resultado_sr1, bfgs = resultado_bfgs)
end

# ==============================================================================
# Aplicação ao problema de calibração Saint-Venant (F de ffjm2.jl)
# ==============================================================================

"""
    sv_residual(x)

Resíduo `F(x) = sv_fork_assimilation(x, 0.0, 31.0, nothing).erro`, o mesmo
usado em `ffjm2.jl` (seção "Exemplo de resíduos e ponto inicial do problema
Saint-Venant").
"""
sv_residual(x) = sv_fork_assimilation(x, 0.0, 31.0, nothing).erro

function sv_box_penalty(x::AbstractVector, lower::AbstractVector, upper::AbstractVector, penalty_weight::Real)
    penalty = zero(eltype(x))
    for i in eachindex(x)
        below = max(zero(x[i]), lower[i] - x[i])
        above = max(zero(x[i]), x[i] - upper[i])
        penalty += below^2 + above^2
    end
    return penalty_weight * penalty
end

"""
    sv_objective(F, x; penalty_weight=1e6, lower=zeros(length(x)), upper=fill(0.5, length(x)))

Função objetivo escalar `sum(F(x).^2) + penalidade externa de caixa`, a
partir de um resíduo vetorial `F` (por padrão [`sv_residual`](@ref)), no
mesmo esquema usado por `bfgs_puro_penalizado` (ver `ffjm2.jl`). Chama `F`
uma única vez; não calcula gradiente algum.
"""
function sv_objective(
    F, x::AbstractVector;
    penalty_weight::Real = 1e6,
    lower::AbstractVector = zeros(length(x)),
    upper::AbstractVector = fill(0.5, length(x)),
)
    value = sum(abs2, F(x))
    return isnan(value) ? oftype(value, 1e26) : value
end

"""
    sv_gradient(F, x; penalty_weight=1e6, lower=zeros(length(x)), upper=fill(0.5, length(x)))

Gradiente de [`sv_objective`](@ref) por `ForwardDiff.gradient`. Como o
gradiente é obtido por diferenciação automática direta (`ForwardDiff.Chunk`
com todas as `length(x)` variáveis juntas), avalia `F` com números duais uma
única vez — não chama `sv_objective` separadamente antes.
"""
function sv_gradient(
    F, x::AbstractVector;
    penalty_weight::Real = 1e6,
    lower::AbstractVector = zeros(length(x)),
    upper::AbstractVector = fill(0.5, length(x)),
)
    obj(z) = sv_objective(F, z; penalty_weight, lower, upper)
    return ForwardDiff.gradient(obj, x)
end

"""
    teste_sr1_bfgs_sv(; kwargs...)

Executa `sr1_backtracking` e `bfgs_backtracking` sobre a função objetivo
penalizada de [`sv_residual`](@ref) (calibração dos coeficientes de
rugosidade do modelo Saint-Venant), a partir de `x0 = fill(0.09, 3)`.

Como `sv_residual` envolve uma simulação completa do modelo, os limites
padrão de iterações/avaliações são os mesmos usados em `bfgs_puro_penalizado`
(`ffjm2.jl`), bem mais modestos que os do teste de Rosenbrock. Note que, por
fidelidade ao Fortran original, `sr1_backtracking` limita o número de
avaliações da função pelo próprio `maxit` (não por `maxnef`).
"""
function teste_sr1_bfgs_sv(;
    x0::AbstractVector = fill(0.09, 3),
    tol::Real = 1e-3,
    maxit::Integer = 100,
    maxnef::Integer = 200,
    penalty_weight::Real = 1e6,
    lower::AbstractVector = zeros(length(x0)),
    upper::AbstractVector = fill(0.5, length(x0)),
    show_trace::Bool = true,
)
    objective(x) = sv_objective(sv_residual, x; penalty_weight, lower, upper)
    gradient(x) = sv_gradient(sv_residual, x; penalty_weight, lower, upper)

    println("=========================================")
    println("   SR1 + BACKTRACKING (Saint-Venant)     ")
    println("=========================================")
    resultado_sr1 = sr1_backtracking(objective, gradient, x0, tol, maxit, maxnef; show_trace)
    println("x = ", resultado_sr1.x, ", f(x) = ", objective(resultado_sr1.x))
    println("info = ", resultado_sr1.info, ", iterações = ", resultado_sr1.kon,
        ", avaliações de f = ", resultado_sr1.nef, ", avaliações de grad = ", resultado_sr1.neg,
        ", substituições por -g = ", resultado_sr1.notdes)

    println("=========================================")
    println("   BFGS + BACKTRACKING (Saint-Venant)    ")
    println("=========================================")
    resultado_bfgs = bfgs_backtracking(objective, gradient, x0, tol, maxit, maxnef; show_trace)
    println("x = ", resultado_bfgs.x, ", f(x) = ", objective(resultado_bfgs.x))
    println("info = ", resultado_bfgs.info, ", iterações = ", resultado_bfgs.kon,
        ", avaliações de f = ", resultado_bfgs.nef, ", avaliações de grad = ", resultado_bfgs.neg,
        ", substituições por -g = ", resultado_bfgs.notdes)

    return (sr1 = resultado_sr1, bfgs = resultado_bfgs)
end
