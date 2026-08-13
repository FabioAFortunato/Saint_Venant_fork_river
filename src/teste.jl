using ForwardDiff
using LinearAlgebra
using LineSearches
using Optim
using Printf

include("ffjm2.jl")

# `DynamicBackTracking` mora em ffjm2.jl (seção 11) — teste.jl inclui ffjm2.jl
# acima, então já está disponível aqui. Ficou lá porque `comparar_backtracking`
# (também em ffjm2.jl) passou a usá-la.

# ==============================================================================
# Demonstração sobre a função de Rosenbrock 2D
# ==============================================================================

rosenbrock(x::AbstractVector) = 100.0 * (x[2] - x[1]^2)^2 + (1.0 - x[1])^2

function rosenbrock_grad!(g::AbstractVector, x::AbstractVector)
    g[1] = -400.0 * x[1] * (x[2] - x[1]^2) - 2.0 * (1.0 - x[1])
    g[2] = 200.0 * (x[2] - x[1]^2)
    return g
end

function teste_dynamic_backtracking(; show_trace::Bool = true)
    x0 = [-1.2, 1.0]
    options = Optim.Options(show_trace = show_trace, store_trace = true)
    method = Optim.BFGS(linesearch = DynamicBackTracking())
    result = Optim.optimize(rosenbrock, rosenbrock_grad!, x0, method, options)

    println("x = ", Optim.minimizer(result), ", f(x) = ", Optim.minimum(result))
    println(
        "iterações = ", Optim.iterations(result),
        ", avaliações de f = ", Optim.f_calls(result),
        ", avaliações de grad = ", Optim.g_calls(result),
    )
    return result
end

# ==============================================================================
# Função objetivo e gradiente compartilhadas por todos os testes de BFGS sobre
# o problema real de calibração Saint-Venant
#
# Definidas uma única vez, acima de qualquer teste, para que todo teste BFGS
# sobre esse problema otimize exatamente a mesma superfície (mesma penalidade
# de caixa, mesmos limites, mesma configuração do ForwardDiff) em vez de cada
# teste reconstruir sua própria closure — o que já causou divergência sutil de
# comportamento entre testes.
#
# Mesmo problema penalizado de `bfgs_puro_penalizado`/`comparar_backtracking`
# (ffjm2.jl): resíduo = sv_fork_assimilation(x, 0.0, tend, nothing).erro,
# dim = 10, x0 = fill(0.09, dim).
#
# `sv_objective` satura para 1e26 quando o valor não é finito. Isso cobre dois
# casos: (a) `sv_fork_assimilation` já sinaliza internamente uma simulação que
# divergiu devolvendo um resíduo grande porém finito (`RMSD = 10e26` em
# `sv_fork.jl`), e (b) o quadrado desse resíduo, somado à penalidade de caixa,
# pode estourar para `Inf`. Sem essa saturação, um `Inf`/`NaN` na função
# objetivo se propaga: a busca linear nunca encontra `alpha` que satisfaça
# Armijo (a comparação com `NaN`/`Inf` falha para qualquer `alpha`), esgota o
# backtracking e lança `LineSearchException` — foi exatamente isso que
# acontecia antes desta correção.
# ==============================================================================

const sv_tend = 31.0
const sv_dim = 10
const sv_lower = zeros(sv_dim)
const sv_upper = fill(0.5, sv_dim)
const sv_penalty_weight = 1e6

sv_residual(x::AbstractVector) = sv_fork_assimilation(x, 0.0, sv_tend, nothing).erro

function sv_box_penalty(x::AbstractVector)
    penalty = zero(eltype(x))
    for i in eachindex(x)
        below = max(zero(x[i]), sv_lower[i] - x[i])
        above = max(zero(x[i]), x[i] - sv_upper[i])
        penalty += below^2 + above^2
    end
    return sv_penalty_weight * penalty
end

function sv_objective(x::AbstractVector)
    value = sum(abs2, sv_residual(x)) + sv_box_penalty(x)
    return isfinite(value) ? value : oftype(value, 1e26)
end

const _sv_gradient_config = ForwardDiff.GradientConfig(
    sv_objective, fill(0.09, sv_dim), ForwardDiff.Chunk{sv_dim}(),
)

function sv_gradient!(G::AbstractVector, x::AbstractVector)
    ForwardDiff.gradient!(G, sv_objective, x, _sv_gradient_config)
    return G
end

# ==============================================================================
# Demonstração sobre o objetivo real de calibração Saint-Venant
#
# Usa sempre `sv_objective`/`sv_gradient!` acima; varia apenas `linesearch` e
# as opções do `Optim.BFGS`.
# ==============================================================================

"""
    teste_dynamic_backtracking_sv(; x0=fill(0.09, sv_dim), linesearch=DynamicBackTracking(), show_trace=true, optim_options=(;))

Roda `Optim.BFGS(; linesearch)` sobre `sv_objective`/`sv_gradient!`.
`optim_options` é repassado a `Optim.Options` (ex.: `optim_options=(iterations=50, g_abstol=1e-3)`),
não confundir com as opções de `bfgs_puro_penalizado`, que este teste não usa mais.
"""
function teste_dynamic_backtracking_sv(;
    x0::AbstractVector = fill(0.09, sv_dim),
    linesearch = DynamicBackTracking(),
    show_trace::Bool = true,
    optim_options = (;),
)
    options = Optim.Options(; merge((; show_trace, store_trace = true), optim_options)...)
    method = Optim.BFGS(; linesearch)
    result = Optim.optimize(sv_objective, sv_gradient!, x0, method, options)

    minimizer = Optim.minimizer(result)
    residual = sv_residual(minimizer)
    sse = sum(abs2, residual)
    rmsd = sqrt(sse / length(residual))
    gradient_norm = norm(sv_gradient!(similar(minimizer), minimizer))

    println("x = ", minimizer)
    println("SSE = ", sse, ", RMSD = ", rmsd, ", ||grad|| = ", gradient_norm)
    println(
        "iterações = ", Optim.iterations(result),
        ", avaliações de f = ", Optim.f_calls(result),
        ", avaliações de grad = ", Optim.g_calls(result),
        ", convergiu = ", Optim.converged(result),
    )
    return result
end


