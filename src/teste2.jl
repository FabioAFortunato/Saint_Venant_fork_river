using LinearAlgebra
using Printf

# ==============================================================================
# SPG (gradiente espectral projetado, Birgin-Martínez-Raydan) com restrição de
# caixa. Mesmo estilo de sr1_backtracking/bfgs_backtracking em
# `sr1_bfgs_backtracking.jl`: busca linear geométrica simples (`lambda *=
# 0.5`), teste de direção de descida e tratamento explícito de NaN/Inf no
# teste de Armijo.
# ==============================================================================

rosenbrock(x::AbstractVector) = 100.0 * (x[2] - x[1]^2)^2 + (1.0 - x[1])^2

function rosenbrock_grad(x::AbstractVector)
    g1 = -400.0 * x[1] * (x[2] - x[1]^2) - 2.0 * (1.0 - x[1])
    g2 = 200.0 * (x[2] - x[1]^2)
    return [g1, g2]
end

"""
    project_box(x, lower, upper)

Projeção Euclidiana de `x` na caixa `[lower, upper]`.
"""
project_box(x::AbstractVector, lower::AbstractVector, upper::AbstractVector) =
    clamp.(x, lower, upper)

"""
    spg_box(f, grad, x0, lower, upper; kwargs...)

Gradiente espectral projetado (SPG, Birgin, Martínez & Raydan, *SIAM J.
Optim.* 2000) para

    min f(x) sujeito a lower <= x <= upper.

A cada iteração:
1. Direção `d = P(x - alpha*g) - x` (`P` = projeção na caixa), com `alpha` o
   passo espectral de Barzilai-Borwein.
2. Busca linear não-monótona de Grippo-Lampariello-Lucidi (GLL): aceita o
   maior `lambda = (1/2)^k ∈ (0, 1]` tal que
   `f(x + lambda*d) <= max(últimos M valores aceitos de f) + c1*lambda*g'd`.
   O backtracking em si é geométrico simples (`lambda *= 0.5`), como em
   `sr1_backtracking`/`bfgs_backtracking`.
3. Passo espectral seguinte: `alpha = clamp(s's / s'y, alpha_min, alpha_max)`
   se `s'y > 0` (`s = x_next - x`, `y = g_next - g`), senão `alpha_max`.

Convergência: `‖P(x - g) - x‖_∞ < gtol` — gradiente projetado com passo
unitário, a medida de estacionariedade padrão do SPG, independente do passo
espectral `alpha` usado para gerar a direção `d`.

Retorna `(x, fval, info, kon, nef, neg)`, onde `fval` é o valor de `f` no
`x` devolvido e `info` vale `0` se convergiu, `1`
se atingiu `maxit` iterações, `2` se atingiu `maxnef` avaliações da função,
`3` se a busca linear falhou (`lambda` caiu a `min_lambda` ou menos sem
satisfazer o teste GLL) ou a direção projetada não foi de descida (`g'd`
não finito ou não negativo — não deveria ocorrer em aritmética exata, já
que a projeção euclidiana é não expansiva, mas é verificado por segurança
numérica), ou `4` se o gradiente projetado ficou nulo (`‖P(x-g)-x‖_∞ <
gtol`) num ponto onde `f(x)` não é finito.

O caso `info = 4` existe porque `f`/`grad` normalmente vêm de
[`sv_objective`](@ref)/[`sv_gradient!`](@ref) (`sr1_bfgs_backtracking.jl`),
cujo núcleo (`sv_objective_from_residual`) satura para `Inf` em pontos
inválidos (fora do domínio físico da simulação, ou onde a penalidade de
caixa domina) — e o `ForwardDiff` devolve gradiente identicamente nulo
nesse platô saturado (`oftype(value, Inf)` descarta as partials). Sem essa
distinção, um ponto assim seria reportado como `info = 0` (convergência):
gradiente nulo passa no teste `‖P(x-g)-x‖_∞ < gtol` mesmo estando preso num
ponto penalizado, não num mínimo de verdade. Exigir `isfinite(fval)` para
aceitar a convergência resolve isso sem precisar mexer na função objetivo:
`sv_objective_from_residual` satura exatamente para `Inf` (não um número
finito grande), então a checagem detecta com precisão os pontos saturados.
"""
function spg_box(
    f, grad, x0::AbstractVector, lower::AbstractVector, upper::AbstractVector;
    gtol::Real = 1.0e-6,
    maxit::Integer = 1000,
    maxnef::Integer = 10_000,
    M::Integer = 10,
    alpha_min::Real = 1.0e-10,
    alpha_max::Real = 1.0e10,
    c1::Real = 1.0e-4,
    min_lambda::Real = 1.0e-12,
    show_trace::Bool = false,
)
    n = length(x0)
    length(lower) == n || throw(DimensionMismatch("lower e x0 devem ter o mesmo tamanho"))
    length(upper) == n || throw(DimensionMismatch("upper e x0 devem ter o mesmo tamanho"))
    all(lower .<= upper) || throw(ArgumentError("lower deve ser <= upper em cada componente"))

    x = project_box(collect(float.(x0)), lower, upper)
    fval = f(x)
    g = grad(x)
    nef = 1
    neg = 1
    kon = 0

    # Passo espectral inicial: escala pelo gradiente em x0 (Birgin-Martínez-Raydan).
    gnorm_inf = norm(g, Inf)
    alpha = gnorm_inf > 0 ? clamp(1.0 / gnorm_inf, alpha_min, alpha_max) : 1.0

    f_history = fill(fval, M)

    for _ in 1:maxit
        kon += 1

        proj_grad = project_box(x .- g, lower, upper) .- x
        proj_grad_inf = norm(proj_grad, Inf)
        if proj_grad_inf < gtol
            if isfinite(fval)
                show_trace && @printf(
                    "iter %4d | f = %.6e | ||P(x-g)-x|| = %.3e | convergiu\n",
                    kon, fval, proj_grad_inf,
                )
                return (x = x, fval, info = 0, kon = kon, nef = nef, neg = neg)
            end
            # Gradiente projetado nulo, mas f(x) não é finito: não é um
            # mínimo real, é um ponto saturado/penalizado onde o gradiente
            # foi zerado artificialmente (ver docstring). Não reporta
            # convergência falsa.
            show_trace && @printf(
                "iter %4d | f = %.6e | ||P(x-g)-x|| = %.3e | gradiente nulo em ponto inválido (f não finito) — não é convergência\n",
                kon, fval, proj_grad_inf,
            )
            return (x = x, fval, info = 4, kon = kon, nef = nef, neg = neg)
        end

        d = project_box(x .- alpha .* g, lower, upper) .- x
        gdotd = dot(g, d)
        if !(gdotd < 0.0)
            show_trace && @printf(
                "iter %4d | f = %.6e | direção projetada não é de descida (g'd = %.3e)\n",
                kon, fval, gdotd,
            )
            return (x = x, fval, info = 3, kon = kon, nef = nef, neg = neg)
        end

        # Busca linear não-monótona (GLL): compara com o maior f dos últimos
        # M iterandos aceitos, não só com fval — permite passos que piorem f
        # momentaneamente, evitando o efeito "vale estreito" do Armijo
        # monótono padrão.
        fmax = maximum(f_history)
        lambda = 1.0
        xnext = similar(x)
        fnext = fval
        armijo_satisfeito = false
        while true
            xnext = x .+ lambda .* d
            fnext = f(xnext)
            nef += 1
            if nef >= maxnef
                show_trace && @printf(
                    "iter %4d | f = %.6e | limite de avaliações (nef >= maxnef) atingido\n",
                    kon, fval,
                )
                return (x = x, fval, info = 2, kon = kon, nef = nef, neg = neg)
            end
            if isfinite(fnext) && fnext <= fmax + c1 * lambda * gdotd
                armijo_satisfeito = true
                break
            end
            lambda *= 0.5
            lambda > min_lambda || break
        end

        if !armijo_satisfeito
            show_trace && @printf(
                "iter %4d | f = %.6e | lambda = %.3e | busca linear falhou (GLL não satisfeito)\n",
                kon, fval, lambda,
            )
            return (x = x, fval, info = 3, kon = kon, nef = nef, neg = neg)
        end

        gnext = grad(xnext)
        neg += 1
        s = xnext .- x
        y = gnext .- g
        sy = dot(s, y)
        alpha = sy > 0 ? clamp(dot(s, s) / sy, alpha_min, alpha_max) : alpha_max

        x = xnext
        fval = fnext
        g = gnext
        f_history[mod1(kon, M)] = fval

        show_trace && @printf(
            "iter %4d | f = %.6e | ||P(x-g)-x|| = %.3e | lambda = %.3e | alpha = %.3e\n",
            kon, fval, proj_grad_inf, lambda, alpha,
        )
    end

    return (x = x, fval, info = 1, kon = kon, nef = nef, neg = neg)
end

# ==============================================================================
# Teste: Rosenbrock 2D com caixa
# ==============================================================================

"""
    teste_spg(; show_trace=true)

Executa `spg_box` na função de Rosenbrock 2D a partir de `x0 = (-1.2, 1.0)`
em dois cenários:

- Caixa larga `[-2, 2]²`, que não fica ativa no mínimo `(1, 1)` — valida a
  convergência ao ótimo irrestrito conhecido.
- Caixa apertada `[-2, 0.5]²`, que exclui o mínimo do Rosenbrock — a solução
  deve ficar na fronteira `x = (0.5, 0.5)`, testando a projeção de fato.
"""
function teste_spg(; show_trace::Bool = true)
    x0 = [-1.2, 1.0]

    println("=========================================")
    println("   TESTE SPG (caixa larga, [-2,2]^2)    ")
    println("=========================================")
    resultado_larga = spg_box(
        rosenbrock, rosenbrock_grad, x0, [-2.0, -2.0], [2.0, 2.0]; show_trace,
    )
    println("Solução: x = ", resultado_larga.x, ", f(x) = ", rosenbrock(resultado_larga.x))
    println("info = ", resultado_larga.info, ", iterações = ", resultado_larga.kon,
        ", avaliações de f = ", resultado_larga.nef, ", avaliações de grad = ", resultado_larga.neg)

    println("=========================================")
    println("   TESTE SPG (caixa apertada, [-2,0.5]^2)")
    println("=========================================")
    resultado_apertada = spg_box(
        rosenbrock, rosenbrock_grad, x0, [-2.0, -2.0], [0.5, 0.5]; show_trace,
    )
    println("Solução: x = ", resultado_apertada.x, ", f(x) = ", rosenbrock(resultado_apertada.x))
    println("info = ", resultado_apertada.info, ", iterações = ", resultado_apertada.kon,
        ", avaliações de f = ", resultado_apertada.nef, ", avaliações de grad = ", resultado_apertada.neg)

    return (larga = resultado_larga, apertada = resultado_apertada)
end
