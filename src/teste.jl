using ForwardDiff
using JuMP
using LinearAlgebra
using LineSearches
using NLPModels
using NLPModelsJuMP
using NLSProblems
using Optim
using Printf
using Statistics

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

# ==============================================================================
# Testes de ffjm2 nos problemas de mínimos quadrados de Moré–Garbow–Hillstrom
# (MGH), a mesma coleção incorporada em CUTEst, via NLSProblems.jl.
#
# NLSProblems.jl monta cada problema como um `MathOptNLSModel` (JuMP por
# baixo); seu `residual` não aceita números `Dual` do ForwardDiff, então aqui
# `ffjm2` (e o BFGS de comparação) recebem a Jacobiana analítica via
# `jac_residual` em vez de deixar o ForwardDiff diferenciar `F` sozinho.
#
# `testar_ffjm2_mgh` e `testar_bfgs_backtracking_mgh` devolvem NamedTuples com
# os mesmos nomes de campo (`minimum`, `gradient`, `iterations`,
# `function_evaluations`, `gradient_evaluations`, `execution_time_seconds`,
# `converged`, `status`, `minimizer`) justamente para que `_mgh_csv_row`
# consiga montar a linha do CSV de forma genérica, sem saber qual dos dois
# métodos produziu o resultado.
# ==============================================================================

const MGH_PROBLEM_NAMES = [Symbol("mgh", lpad(i, 2, "0")) for i in 1:34]

# ==============================================================================
# `NLSProblems.mgh11` (Gulf research and development function) tem um bug: a
# linha `abs(y[i] * m * i * x[2])^x[3]` deveria ser `abs(y[i] - x[2])^x[3]`
# (ver `~/.julia/packages/NLSProblems/*/src/mgh11.jl`). Por causa disso, o
# mínimo global conhecido do problema, x* = (50, 25, 1.5) com f(x*) = 0
# (conferido contra o problema `GULF` do CUTEst: f(x*) ≈ 9e-14), não é mais o
# mínimo do resíduo que o pacote instalado calcula. `mgh11_corrigido` reproduz
# a mesma construção via JuMP/`MathOptNLSModel` que `NLSProblems.jl` usa, só
# com a fórmula certa.
# ==============================================================================

function mgh11_corrigido(; m::Int = 100)
    t = (1:m) ./ 100
    y = 25 .+ (-50 .* log.(t)) .^ (2 / 3)
    model = Model()
    @variable(model, x[1:3])
    set_start_value.(x, [5.00; 2.50; 0.15])
    @NLexpression(model, F[i = 1:m], exp(-abs(y[i] - x[2])^x[3] / x[1]) - t[i])
    return MathOptNLSModel(model, F, name = "mgh11")
end

# Ponto único de resolução de um nome de problema MGH: usa a versão corrigida
# acima para `:mgh11` e delega a `NLSProblems.jl` para os outros 33.
_mgh_problem(nome::Symbol) = nome === :mgh11 ? mgh11_corrigido() : getfield(NLSProblems, nome)()

const _MGH_CSV_HEADER = "problem,n,m,method,f,sse,rmsd,gradient_norm,iterations,function_evaluations,gradient_evaluations,execution_time_seconds,converged,status,x0,solution\n"

function _mgh_csv_row(nome::AbstractString, n::Integer, m::Integer, x0::AbstractVector, metodo::AbstractString, resultado)
    sse = 2 * resultado.minimum
    return (;
        problem = nome,
        n,
        m,
        method = metodo,
        f = resultado.minimum,
        sse,
        rmsd = sqrt(sse / m),
        gradient_norm = norm(resultado.gradient),
        iterations = resultado.iterations,
        function_evaluations = resultado.function_evaluations,
        gradient_evaluations = resultado.gradient_evaluations,
        execution_time_seconds = resultado.execution_time_seconds,
        converged = resultado.converged,
        status = string(resultado.status),
        x0 = collect(x0),
        solution = collect(resultado.minimizer),
    )
end

function _write_mgh_csv_row(io, row)
    values = (
        row.problem, row.n, row.m, row.method, row.f, row.sse, row.rmsd,
        row.gradient_norm, row.iterations, row.function_evaluations,
        row.gradient_evaluations, row.execution_time_seconds, row.converged,
        row.status, "\"" * repr(row.x0) * "\"", "\"" * repr(row.solution) * "\"",
    )
    write(io, join(values, ','), '\n')
end

"""
    testar_ffjm2_mgh(nls_ou_nome; update=:bfgs, show_trace=false, kwargs...)

Roda `ffjm2` sobre um problema MGH de `NLSProblems.jl`, aceito tanto como
modelo já construído (`nls::NLPModels.AbstractNLSModel`) quanto pelo nome
(`nome::Symbol`, ex.: `:mgh01`). `kwargs` é repassado a `ffjm2` (ex.:
`model_multistart=10` para acelerar testes exploratórios). Imprime um resumo
de uma linha e devolve o `NamedTuple` de `ffjm2`.
"""
function testar_ffjm2_mgh(nls::NLPModels.AbstractNLSModel; update::Union{Symbol,Tuple{Vararg{Symbol}}} = :bfgs, show_trace::Bool = true, kwargs...)
    F(x) = residual(nls, x)
    jacobiana(x) = jac_residual(nls, x)

    resultado = ffjm2(F, nls.meta.x0; jacobian = jacobiana, update, show_trace, kwargs...)

    @printf(
        "%-7s n=%-3d m=%-3d f=%.6e ||grad||=%.3e iter=%-4d status=%-22s convergiu=%s\n",
        nls.meta.name, nls.meta.nvar, nls.nls_meta.nequ,
        resultado.minimum, norm(resultado.gradient), resultado.iterations,
        resultado.status, resultado.converged,
    )

    return resultado
end

testar_ffjm2_mgh(nome::Symbol; kwargs...) =
    testar_ffjm2_mgh(_mgh_problem(nome); kwargs...)

# ==============================================================================
# Mudança de variável diagonal x = D·y, D = diag(d), para problemas com chute
# inicial desbalanceado (ex. mgh10: x0 = (0.02, 4000, 250), 6 ordens de
# grandeza de diferença — ver [[project_mgh10_hard_case]]).
#
#   G(y) = F(D·y)              (mesmo valor de f, só muda a coordenada)
#   J_G(y) = J_F(x)·D          (regra da cadeia: coluna j escalada por d_j)
#
# `ffjm2` roda inteiro sobre (G, y0 = x0./d) sem saber que houve reescala —
# H_i, Δ, gnorm etc. passam a ser construídos/medidos em unidades de y. No
# fim, desfaz-se com x* = D·y* e g(x*) = g̃(y*)./d (pois g̃ = D·g(x)).
#
# `d = max.(|x0|, scale_floor)` por padrão: usa a ordem de grandeza do
# próprio chute inicial como escala de cada variável.
# ==============================================================================

"""
    testar_ffjm2_mgh_normalizado(nls_ou_nome; update=:bfgs, scale=nothing, scale_floor=1e-8, show_trace=false, kwargs...)

Igual a [`testar_ffjm2_mgh`](@ref), mas roda `ffjm2` sobre a mudança de
variável `x = D·y` (`D = diag(scale)`) em vez de `x` diretamente — normaliza
componentes de escala muito diferente no chute inicial (ex. `mgh10`,
`mgh17`) antes de otimizar. `scale` por padrão é `max.(abs.(x0), scale_floor)`
(a ordem de grandeza do próprio `x0`); passe um vetor explícito para usar
outra escala. `kwargs` é repassado a `ffjm2`. O `NamedTuple` devolvido tem os
mesmos campos de `ffjm2`, com `minimizer`/`gradient` já desfeitos de volta
para as unidades originais de `x` (não de `y`), mais um campo extra `scale`
com o `D` usado.
"""
function testar_ffjm2_mgh_normalizado(
    nls::NLPModels.AbstractNLSModel;
    update::Union{Symbol,Tuple{Vararg{Symbol}}} = :bfgs,
    scale::Union{Nothing,AbstractVector} = nothing,
    scale_floor::Real = 1e-8,
    show_trace::Bool = true,
    kwargs...,
)
    x0 = nls.meta.x0
    d = scale === nothing ? max.(abs.(x0), scale_floor) : collect(float.(scale))
    length(d) == length(x0) ||
        throw(DimensionMismatch("scale e x0 devem ter o mesmo tamanho"))
    any(d .<= 0) && throw(ArgumentError("scale deve ser estritamente positivo"))

    G(y) = residual(nls, d .* y)
    jacobiana_y(y) = jac_residual(nls, d .* y) * Diagonal(d)
    y0 = x0 ./ d

    resultado_y = ffjm2(G, y0; jacobian = jacobiana_y, update, show_trace, kwargs...)

    minimizer = d .* resultado_y.minimizer
    gradient = resultado_y.gradient ./ d
    resultado = merge(resultado_y, (; minimizer, gradient, scale = d))

    @printf(
        "%-7s n=%-3d m=%-3d f=%.6e ||grad||=%.3e iter=%-4d status=%-22s convergiu=%s (normalizado)\n",
        nls.meta.name, nls.meta.nvar, nls.nls_meta.nequ,
        resultado.minimum, norm(resultado.gradient), resultado.iterations,
        resultado.status, resultado.converged,
    )

    return resultado
end

testar_ffjm2_mgh_normalizado(nome::Symbol; kwargs...) =
    testar_ffjm2_mgh_normalizado(_mgh_problem(nome); kwargs...)

"""
    testar_bfgs_backtracking_mgh(nls_ou_nome; order=2, show_trace=false, optim_options=(;))

Roda `Optim.BFGS(linesearch=LineSearches.BackTracking(order=order))` sobre o
mesmo objetivo `0.5*||F(x)||²` que `ffjm2` minimiza, usando gradiente
analítico (`Jᵀr`, via `jac_residual`) pelo mesmo motivo de `testar_ffjm2_mgh`.
`order=2` é interpolação quadrática, `order=3` cúbica (ver `LineSearches.jl`).
Devolve um `NamedTuple` com os mesmos nomes de campo relevantes de `ffjm2`
(`minimum`, `gradient`, `iterations`, `function_evaluations`,
`gradient_evaluations`, `execution_time_seconds`, `converged`, `status`,
`minimizer`), para poder ser comparado linha a linha com o resultado de
`testar_ffjm2_mgh`.
"""
function testar_bfgs_backtracking_mgh(
    nls::NLPModels.AbstractNLSModel;
    order::Integer = 2,
    show_trace::Bool = false,
    optim_options = (;),
)
    x0 = collect(nls.meta.x0)
    objetivo(x) = 0.5 * sum(abs2, residual(nls, x))
    function gradiente!(g, x)
        mul!(g, jac_residual(nls, x)', residual(nls, x))
        return g
    end

    options = Optim.Options(; merge((; show_trace, store_trace = true), optim_options)...)
    method = Optim.BFGS(linesearch = LineSearches.BackTracking(order = order))

    start_ns = time_ns()
    result = Optim.optimize(objetivo, gradiente!, x0, method, options)
    execution_time_seconds = (time_ns() - start_ns) / 1e9

    minimizer = Optim.minimizer(result)
    gradient = gradiente!(similar(minimizer), minimizer)
    converged = Optim.converged(result)
    status = if Optim.g_converged(result)
        "gradient_converged"
    elseif Optim.x_converged(result)
        "step_converged"
    elseif Optim.f_converged(result)
        "function_converged"
    elseif Optim.iteration_limit_reached(result)
        "maximum_iterations"
    else
        "not_converged"
    end

    @printf(
        "%-7s n=%-3d m=%-3d f=%.6e ||grad||=%.3e iter=%-4d status=%-22s convergiu=%s\n",
        nls.meta.name, nls.meta.nvar, nls.nls_meta.nequ,
        Optim.minimum(result), norm(gradient), Optim.iterations(result),
        status, converged,
    )

    return (;
        minimizer,
        minimum = Optim.minimum(result),
        gradient,
        iterations = Optim.iterations(result),
        function_evaluations = Optim.f_calls(result),
        gradient_evaluations = Optim.g_calls(result),
        execution_time_seconds,
        converged,
        status,
        result,
    )
end

testar_bfgs_backtracking_mgh(nome::Symbol; kwargs...) =
    testar_bfgs_backtracking_mgh(_mgh_problem(nome); kwargs...)

"""
    testar_ffjm2_mgh_todos(; nomes=MGH_PROBLEM_NAMES, update=:bfgs,
                              output="results/ffjm2_mgh.csv", kwargs...)

Roda `testar_ffjm2_mgh` sobre todos os problemas em `nomes` (por padrão os 34
problemas MGH), salvando uma linha por problema em `output` (CSV; passe
`output=nothing` para não salvar). Problemas que lançarem exceção são
reportados e pulados sem interromper o laço.
"""
function testar_ffjm2_mgh_todos(;
    nomes::AbstractVector{Symbol} = MGH_PROBLEM_NAMES,
    update::Union{Symbol,Tuple{Vararg{Symbol}}} = :sr1,
    output::Union{Nothing,AbstractString} = "results/ffjm2_mgh.csv",
    kwargs...,
)
    resultados = Dict{Symbol,Any}()
    io = nothing
    if output !== nothing
        mkpath(dirname(output))
        io = open(output, "w")
        write(io, _MGH_CSV_HEADER)
    end

    try
        for nome in nomes
            nls = _mgh_problem(nome)
            try
                resultado = testar_ffjm2_mgh(nls; update, kwargs...)
                resultados[nome] = resultado
                if io !== nothing
                    row = _mgh_csv_row(
                        String(nome), nls.meta.nvar, nls.nls_meta.nequ, nls.meta.x0,
                        "ffjm2_" * (update isa Symbol ? String(update) : join(String.(update), "+")),
                        resultado,
                    )
                    _write_mgh_csv_row(io, row)
                    flush(io)
                end
            catch e
                println("$(nome): ERRO - $(sprint(showerror, e))")
                resultados[nome] = e
            end
        end
    finally
        io !== nothing && close(io)
    end

    output !== nothing && println("Resultados salvos em: $output")
    return resultados
end

"""
    comparar_ffjm2_bfgs_mgh(; nomes=MGH_PROBLEM_NAMES,
                               output="results/comparacao_ffjm2_bfgs_mgh.csv",
                               update=:bfgs, backtracking_order=2,
                               ffjm2_options=(;), bfgs_options=(;), show_trace=false)

Roda, para cada problema MGH em `nomes`, tanto `ffjm2` (atualização `update`)
quanto `Optim.BFGS` com `LineSearches.BackTracking(order=backtracking_order)`,
salvando duas linhas por problema (uma por método) em `output` (CSV). Um
problema que lançar exceção em um dos dois métodos é reportado e pulado sem
impedir a linha do outro método nem interromper o laço.
"""
function comparar_ffjm2_bfgs_mgh(;
    nomes::AbstractVector{Symbol} = MGH_PROBLEM_NAMES,
    output::AbstractString = "results/sr1_comparacao_ffjm2_bfgs_mgh.csv",
    update::Union{Symbol,Tuple{Vararg{Symbol}}} = :psb,
    backtracking_order::Integer = 2,
    ffjm2_options = (;),
    bfgs_options = (;),
    show_trace::Bool = false,
)
    mkpath(dirname(output))
    rows = NamedTuple[]
    metodo_ffjm2 = "ffjm2_" * (update isa Symbol ? String(update) : join(String.(update), "+"))
    metodo_bfgs = "bfgs_backtracking$(backtracking_order)"

    open(output, "w") do io
        write(io, _MGH_CSV_HEADER)

        for nome in nomes
            nls = _mgh_problem(nome)
            n, m, x0 = nls.meta.nvar, nls.nls_meta.nequ, nls.meta.x0

            try
                resultado = testar_ffjm2_mgh(nls; update, show_trace, ffjm2_options...)
                row = _mgh_csv_row(String(nome), n, m, x0, metodo_ffjm2, resultado)
                push!(rows, row)
                _write_mgh_csv_row(io, row)
            catch e
                println("$(nome) ($(metodo_ffjm2)): ERRO - $(sprint(showerror, e))")
            end

            try
                resultado = testar_bfgs_backtracking_mgh(nls; order = backtracking_order, show_trace, bfgs_options...)
                row = _mgh_csv_row(String(nome), n, m, x0, metodo_bfgs, resultado)
                push!(rows, row)
                _write_mgh_csv_row(io, row)
            catch e
                println("$(nome) ($(metodo_bfgs)): ERRO - $(sprint(showerror, e))")
            end

            flush(io)
        end
    end

    println("Comparação MGH (ffjm2 vs $(metodo_bfgs)) salva em: $output")
    return (; rows, output)
end

# Média e desvio padrão de um vetor de métricas; devolve (NaN, NaN) se vazio
# e (valor, NaN) se houver só uma amostra (std de 1 ponto não é definido).
function _mean_std(xs::AbstractVector{<:Real})
    v = Float64.(xs)
    isempty(v) && return (NaN, NaN)
    length(v) == 1 && return (v[1], NaN)
    return (mean(v), std(v))
end

"""
    comparar_ffjm2_modelos_mgh(; nomes=MGH_PROBLEM_NAMES, modelos=FFJM2_ALL_MODELS,
                                  output="results/ffjm2_modelos_mgh.csv",
                                  resumo_output="results/ffjm2_modelos_mgh_resumo.csv",
                                  ffjm2_options=(;), show_trace=false)

Roda `ffjm2` separadamente com cada um dos `modelos` (por padrão os 6 de
`FFJM2_ALL_MODELS`, um de cada vez — não em conjunto) sobre todos os
problemas MGH em `nomes`, pra comparar qual `update` individual funciona
melhor. Salva uma linha por (problema, modelo) em `output` (mesmo formato de
`comparar_ffjm2_bfgs_mgh`), e um resumo — uma linha por modelo — em
`resumo_output`, com `n_problems` (quantos problemas rodaram sem lançar
exceção), `n_gradient_converged` (quantos pararam com
`status = :gradient_converged`, ou seja, resolvidos pela norma do
gradiente) e média/desvio padrão (`mean_*`/`std_*`) de `f`, `rmsd`,
`gradient_norm`, `iterations`, `function_evaluations`,
`gradient_evaluations` e `execution_time_seconds` sobre os problemas que não
lançaram exceção. Um problema que lançar exceção num modelo é reportado e
pulado sem interromper o laço nem entrar nas estatísticas desse modelo.
"""
function comparar_ffjm2_modelos_mgh(;
    nomes::AbstractVector{Symbol} = MGH_PROBLEM_NAMES,
    modelos::Tuple{Vararg{Symbol}} = FFJM2_ALL_MODELS,
    output::AbstractString = "results/ffjm2_modelos_mgh.csv",
    resumo_output::AbstractString = "results/ffjm2_modelos_mgh_resumo.csv",
    ffjm2_options = (;),
    show_trace::Bool = false,
)
    mkpath(dirname(output))
    mkpath(dirname(resumo_output))
    rows = NamedTuple[]

    open(output, "w") do io
        write(io, _MGH_CSV_HEADER)
        for modelo in modelos, nome in nomes
            nls = _mgh_problem(nome)
            n, m, x0 = nls.meta.nvar, nls.nls_meta.nequ, nls.meta.x0
            metodo = "ffjm2_" * String(modelo)
            try
                resultado = testar_ffjm2_mgh(nls; update = modelo, show_trace, ffjm2_options...)
                row = _mgh_csv_row(String(nome), n, m, x0, metodo, resultado)
                push!(rows, row)
                _write_mgh_csv_row(io, row)
            catch e
                println("$(nome) ($(metodo)): ERRO - $(sprint(showerror, e))")
            end
            flush(io)
        end
    end
    println("Resultados por modelo (ffjm2) salvos em: $output")

    resumo_header = "model,n_problems,n_gradient_converged,mean_f,std_f,mean_rmsd,std_rmsd," *
        "mean_gradient_norm,std_gradient_norm,mean_iterations,std_iterations," *
        "mean_function_evaluations,std_function_evaluations," *
        "mean_gradient_evaluations,std_gradient_evaluations," *
        "mean_execution_time_seconds,std_execution_time_seconds\n"

    open(resumo_output, "w") do io
        write(io, resumo_header)
        for modelo in modelos
            metodo = "ffjm2_" * String(modelo)
            rs = filter(r -> r.method == metodo, rows)
            isempty(rs) && continue
            n_grad = count(r -> r.status == "gradient_converged", rs)
            mean_f, std_f = _mean_std(getproperty.(rs, :f))
            mean_rmsd, std_rmsd = _mean_std(getproperty.(rs, :rmsd))
            mean_gn, std_gn = _mean_std(getproperty.(rs, :gradient_norm))
            mean_it, std_it = _mean_std(getproperty.(rs, :iterations))
            mean_fe, std_fe = _mean_std(getproperty.(rs, :function_evaluations))
            mean_ge, std_ge = _mean_std(getproperty.(rs, :gradient_evaluations))
            mean_t, std_t = _mean_std(getproperty.(rs, :execution_time_seconds))
            values = (
                metodo, length(rs), n_grad,
                mean_f, std_f, mean_rmsd, std_rmsd, mean_gn, std_gn,
                mean_it, std_it, mean_fe, std_fe, mean_ge, std_ge,
                mean_t, std_t,
            )
            write(io, join(values, ','), '\n')
        end
    end
    println("Resumo por modelo (média/desvio padrão) salvo em: $resumo_output")

    return (; rows, output, resumo_output)
end
