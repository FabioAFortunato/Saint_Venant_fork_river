using HypothesisTests
using PGFPlotsX
using Plots
using Statistics

gr()

# ==============================================================================
# Boxplot de três painéis (função objetivo, avaliações de função, avaliações
# de gradiente) comparando dois métodos nos problemas MGH. Por padrão
# restrito às execuções com `status == "gradient_converged"` — a mesma
# figura gerada anteriormente para comparar `ffjm2` com BFGS
# (`results/boxplot_gradient_converged_ffjm2_psb_bfgs_mgh.png`) — mas o novo
# argumento `status_filter` (ver docstring de `boxplot_ffjm2_vs_bfgs_mgh`)
# permite incluir também os problemas que pararam por Step/Function/Stalled,
# ou seja, TODOS os 34 problemas MGH, não só o subconjunto onde os dois
# métodos convergem por gradiente.
#
# Não recomputa nada: lê resultados já calculados, de um CSV no formato
# escrito por `comparar_ffjm2_bfgs_mgh`/`comparar_ffjm2_modelos_mgh`
# (`src/teste.jl`, cabeçalho `_MGH_CSV_HEADER`) via `csv_path=`, ou recebe
# diretamente um vetor de `NamedTuple`s (`rows=`, ex.:
# `comparar_ffjm2_bfgs_mgh(; update=:psb).rows` de uma corrida já feita na
# sessão). Sem StatsPlots disponível no projeto, a caixa (Q1–Q3), a mediana,
# os whiskers (regra de Tukey, 1.5×IQR) e os outliers são desenhados
# manualmente com primitivas do Plots.jl.
# ==============================================================================

# Lê de volta um CSV no formato de `_MGH_CSV_HEADER` (teste.jl). Só as
# colunas escalares (1 a 14) são interpretadas — `x0`/`solution` (15 e 16)
# vêm entre aspas com vírgulas internas e não são necessárias aqui, então são
# ignoradas em vez de parseadas.
function _read_mgh_csv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("CSV vazio: $path")
    header = split(lines[1], ',')
    idx = Dict(name => i for (i, name) in enumerate(header))

    rows = NamedTuple[]
    for line in lines[2:end]
        isempty(line) && continue
        fields = split(line, ',')
        push!(rows, (;
            problem = fields[idx["problem"]],
            n = parse(Int, fields[idx["n"]]),
            m = parse(Int, fields[idx["m"]]),
            method = fields[idx["method"]],
            f = parse(Float64, fields[idx["f"]]),
            sse = parse(Float64, fields[idx["sse"]]),
            rmsd = parse(Float64, fields[idx["rmsd"]]),
            gradient_norm = parse(Float64, fields[idx["gradient_norm"]]),
            iterations = parse(Int, fields[idx["iterations"]]),
            function_evaluations = parse(Int, fields[idx["function_evaluations"]]),
            gradient_evaluations = parse(Int, fields[idx["gradient_evaluations"]]),
            execution_time_seconds = parse(Float64, fields[idx["execution_time_seconds"]]),
            converged = fields[idx["converged"]] == "true",
            status = fields[idx["status"]],
        ))
    end
    return rows
end

# Resumo de cinco números (Tukey) de um vetor de dados: quartis, whiskers
# (ponto mais extremo dentro de 1.5*IQR da caixa) e outliers (fora disso).
function _tukey_summary(data::AbstractVector{<:Real})
    v = sort(Float64.(data))
    q1, q2, q3 = quantile(v, 0.25), quantile(v, 0.50), quantile(v, 0.75)
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = filter(x -> lo_fence <= x <= hi_fence, v)
    whisker_lo = isempty(inside) ? q1 : minimum(inside)
    whisker_hi = isempty(inside) ? q3 : maximum(inside)
    outliers = filter(x -> x < whisker_lo || x > whisker_hi, v)
    return (; q1, q2, q3, whisker_lo, whisker_hi, outliers)
end

# Desenha uma caixa (Tukey) na posição `x` do painel `p`: retângulo Q1–Q3,
# linha de mediana, whiskers com "tampa" e outliers como pontos dispersos.
function _draw_box!(p, x::Real, stats; width::Real = 0.5, color = 1)
    half = width / 2
    cap = width / 4

    box = Shape([x - half, x + half, x + half, x - half], [stats.q1, stats.q1, stats.q3, stats.q3])
    plot!(p, box; fillcolor = color, fillalpha = 0.85, linecolor = :black, label = "")
    plot!(p, [x - half, x + half], [stats.q2, stats.q2]; linecolor = :black, linewidth = 2, label = "")
    plot!(p, [x, x], [stats.q3, stats.whisker_hi]; linecolor = :black, label = "")
    plot!(p, [x, x], [stats.q1, stats.whisker_lo]; linecolor = :black, label = "")
    plot!(p, [x - cap, x + cap], [stats.whisker_hi, stats.whisker_hi]; linecolor = :black, label = "")
    plot!(p, [x - cap, x + cap], [stats.whisker_lo, stats.whisker_lo]; linecolor = :black, label = "")

    if !isempty(stats.outliers)
        scatter!(
            p, fill(x, length(stats.outliers)), stats.outliers;
            markercolor = color, markerstrokecolor = :black, markersize = 5, label = "",
        )
    end
    return p
end

# Estrelas de significância (teste de Mann–Whitney U, `HypothesisTests.jl`)
# para o contraste entre os dois grupos; `nothing` se p >= 0.05.
function _significance_stars(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    p = pvalue(MannWhitneyUTest(Float64.(a), Float64.(b)))
    p < 0.001 && return "***"
    p < 0.01 && return "**"
    p < 0.05 && return "*"
    return nothing
end

# Chave (bracket) horizontal com as estrelas de significância, desenhada
# acima do maior valor (`y_top`) das duas caixas do painel `p`.
function _draw_significance!(p, x1::Real, x2::Real, y_top::Real, stars::AbstractString)
    y_tick = y_top * 1.08
    y_bar = y_top * 1.22
    plot!(p, [x1, x1, x2, x2], [y_tick, y_bar, y_bar, y_tick]; linecolor = :black, label = "")
    annotate!(p, (x1 + x2) / 2, y_bar * 1.15, text(stars, 11))
    return p
end

# Um painel do boxplot: duas caixas (grupo `a`, grupo `b`) em escala log10,
# com estrelas de significância quando p < 0.05.
function _boxplot_panel(
    a::AbstractVector{<:Real}, b::AbstractVector{<:Real};
    title::AbstractString, ylabel::AbstractString,
    labels::Tuple{AbstractString,AbstractString},
)
    stats_a, stats_b = _tukey_summary(a), _tukey_summary(b)
    lo = min(minimum(a), minimum(b))
    hi = max(maximum(a), maximum(b))
    stars = _significance_stars(a, b)
    headroom = stars === nothing ? 1.5 : 4.0

    p = plot(;
        title, ylabel, yscale = :log10, ylims = (lo / 3, hi * headroom),
        xlims = (0.3, 2.7), xticks = ([1, 2], collect(labels)),
        legend = false, framestyle = :box,
    )
    _draw_box!(p, 1, stats_a; color = 1)
    _draw_box!(p, 2, stats_b; color = 2)
    stars !== nothing && _draw_significance!(p, 1, 2, hi, stars)
    return p
end

"""
    boxplot_ffjm2_vs_bfgs_mgh(; rows=nothing, csv_path=nothing,
                                 method_a="ffjm2_psb", method_b="bfgs_backtracking2",
                                 label_a="FFJM2 (PSB)", label_b="BFGS (backtracking)",
                                 status_filter="gradient_converged",
                                 output="results/boxplot_gradient_converged_ffjm2_vs_bfgs_mgh.png")

Reproduz o boxplot de três painéis (função objetivo, avaliações de função,
avaliações de gradiente) comparando dois métodos (colunas `method` de um CSV
MGH: `method_a` vs `method_b`) nos problemas MGH.

Por padrão (`status_filter = "gradient_converged"`), restringe às execuções
com `status == "gradient_converged"` em ambos — reproduzindo a figura
original, que por isso NÃO considera todos os 34 problemas: qualquer
problema em que `method_a` ou `method_b` tenha parado por Step, Function ou
Stalled (ex.: MGH10, MGH16, MGH31 do lado do FFJM2) fica de fora dos dois
grupos. Passe `status_filter=nothing` para desligar esse filtro e incluir
TODOS os problemas presentes no CSV para cada método, com qualquer status
final — é o jeito de ver as duas figuras lado a lado e conferir a
diferença que a restrição por convergência de gradiente faz. Também aceita
uma string diferente (ex.: `"step"`) para restringir a outro status
específico.

Não computa nada: os dados vêm de resultados já calculados, informados por
`rows` (vetor de `NamedTuple`s, ex.: o campo `rows` devolvido por
`comparar_ffjm2_bfgs_mgh`/`comparar_ffjm2_modelos_mgh` numa corrida já
feita) ou por `csv_path` (caminho de um CSV no formato escrito por essas
duas funções, ex.: `"results/ffjm2_modelos_mgh.csv"` ou
`"results/sr1_comparacao_ffjm2_bfgs_mgh.csv"`); é erro não passar nenhum
dos dois. Os nomes em `method_a`/`method_b` precisam bater com os valores
da coluna `method` desse CSV (ex.: `"ffjm2_psb"`, `"ffjm2_sr1"`,
`"bfgs_backtracking2"`).

Textos do gráfico (títulos, eixos, legendas dos grupos) em inglês. Salva a
figura em `output` (`nothing` para não salvar) e devolve `(; plot, rows)`.
"""
function boxplot_ffjm2_vs_bfgs_mgh(;
    rows = nothing,
    csv_path="results/sr1_comparacao_ffjm2_bfgs_mgh.csv",
    method_a::AbstractString = "ffjm2_psb",
    method_b::AbstractString = "bfgs_backtracking2",
    label_a::AbstractString = "FFJM2 (PSB)",
    label_b::AbstractString = "BFGS (backtracking)",
    status_filter::Union{AbstractString,Nothing} = "gradient_converged",
    output::Union{AbstractString,Nothing} = "results/boxplot_gradient_converged_ffjm2_vs_bfgs_mgh.png",
)
    if rows === nothing
        csv_path === nothing && error("Informe `rows` (resultados já calculados) ou `csv_path` (CSV com esses resultados).")
        rows = _read_mgh_csv(csv_path)
    end

    selected = status_filter === nothing ? rows : filter(r -> r.status == status_filter, rows)
    rows_a = filter(r -> r.method == method_a, selected)
    rows_b = filter(r -> r.method == method_b, selected)

    status_desc = status_filter === nothing ? "." : " com status $status_filter."
    isempty(rows_a) && error("Nenhuma linha de method == \"$method_a\"$status_desc")
    isempty(rows_b) && error("Nenhuma linha de method == \"$method_b\"$status_desc")

    labels = (label_a, label_b)

    p1 = _boxplot_panel(
        getproperty.(rows_a, :f), getproperty.(rows_b, :f);
        title = "Objective function", ylabel = "f(x)", labels,
    )
    p2 = _boxplot_panel(
        Float64.(getproperty.(rows_a, :function_evaluations)),
        Float64.(getproperty.(rows_b, :function_evaluations));
        title = "Function evaluations", ylabel = "Function evaluations", labels,
    )
    p3 = _boxplot_panel(
        Float64.(getproperty.(rows_a, :gradient_evaluations)),
        Float64.(getproperty.(rows_b, :gradient_evaluations));
        title = "Gradient evaluations", ylabel = "Gradient evaluations", labels,
    )

    fig = plot(p1, p2, p3; layout = (1, 3), size = (1500, 450), margin = 5Plots.mm)

    if output !== nothing
        mkpath(dirname(output))
        savefig(fig, output)
        println("Boxplot salvo em: $output")
    end

    return (; plot = fig, rows)
end

# ==============================================================================
# Performance profile (Dolan & Moré, 2002) comparando dois métodos nos
# problemas MGH — mesma fonte de dados/formato de CSV de
# `boxplot_ffjm2_vs_bfgs_mgh` (`_read_mgh_csv`, colunas de
# `comparar_ffjm2_bfgs_mgh`/`comparar_ffjm2_modelos_mgh`, `src/teste.jl`).
#
# Para cada problema p, t_{p,s} é a métrica de custo (`metric`, padrão
# `function_evaluations`) do método s; problemas em que o método não
# convergiu (`converged == false`) contam como falha (t_{p,s} = Inf, nunca
# entra no perfil). r_{p,s} = t_{p,s} / min(t_{p,BFGS}, t_{p,ffjm2}) é a
# razão pro melhor dos dois métodos naquele problema (sempre ≥ 1). O perfil
# ρ_s(τ) = fração dos problemas com r_{p,s} ≤ τ é uma função em escada:
# ρ_s(1) é a fração de problemas em que s foi o melhor (ou empatou), e
# ρ_s(τ) no τ máximo tende à taxa de convergência de s. Sem pacote externo
# de performance profile no projeto, a curva é construída manualmente
# amostrando ρ_s num grid fino de τ (log-espaçado) e desenhando como função
# em escada (`seriestype=:steppost`), igual à abordagem "sem StatsPlots"
# já usada em `_boxplot_panel`.
# ==============================================================================

# Núcleo do cálculo de um performance profile para UMA métrica: monta t_a/t_b
# (custo por problema, Inf se o método não "convergiu" segundo `status_filter`),
# a razão r = t/melhor (r=1 quando os dois empatam em 0, Inf se o vencedor
# "ganhou por zero" e o outro não — caso raro mas possível com `metric=:f`,
# onde o custo pode ser exatamente 0), e amostra ρ_s(τ) num grid log-espaçado
# de τ. Compartilhado por `performance_profile_ffjm2_vs_bfgs_mgh` (um painel)
# e `performance_profiles_ffjm2_vs_bfgs_mgh` (três painéis, uma métrica cada).
#
# `status_filter` decide o que conta como sucesso: uma string (ex.
# `"gradient_converged"`, o padrão, igual a `boxplot_ffjm2_vs_bfgs_mgh`)
# exige esse `status` exato; `nothing` aceita qualquer `converged == true`.
# A diferença importa de verdade: no CSV padrão, BFGS tem `converged=true`
# em 34/34 problemas, mas só 32/34 têm `status == "gradient_converged"` —
# MGH10 parou por `step_converged` e MGH16 por `function_converged`, nenhum
# dos dois uma certificação real de estacionariedade. Contar os três como
# sucesso infla artificialmente a taxa de convergência "de verdade" do
# BFGS no perfil.
function _performance_profile_curve(
    dict_a, dict_b, problems, metric::Symbol, tau_max::Real,
    status_filter::Union{AbstractString,Nothing},
)
    success(row) = status_filter === nothing ? row.converged : row.status == status_filter
    cost(row) = success(row) ? Float64(getproperty(row, metric)) : Inf

    t_a = [cost(dict_a[p]) for p in problems]
    t_b = [cost(dict_b[p]) for p in problems]
    best = min.(t_a, t_b)

    # Problema em que nenhum dos dois teve sucesso (best=Inf): falha mútua,
    # os dois ficam com razão infinita (nunca cruzam nenhum τ finito), em vez
    # de erro — o problema continua contando no denominador de ambos os
    # perfis (convenção usual de Dolan-Moré: o total de problemas é fixo).
    ratio(t, b) = !isfinite(b) ? Inf : (b == 0 ? (t == 0 ? 1.0 : Inf) : t / b)
    r_a = ratio.(t_a, best)
    r_b = ratio.(t_b, best)

    rho(r, taus) = [count(x -> isfinite(x) && x <= tau, r) / length(r) for tau in taus]
    taus = exp.(range(0.0, log(tau_max), length = 400))
    log2_taus = log2.(taus)

    return (; log2_taus, rho_a = rho(r_a, taus), rho_b = rho(r_b, taus), r_a, r_b)
end

"""
    performance_profile_ffjm2_vs_bfgs_mgh(; rows=nothing, csv_path="results/sr1_comparacao_ffjm2_bfgs_mgh.csv",
                                              method_a="ffjm2_psb", method_b="bfgs_backtracking2",
                                              label_a="FFJM2 (PSB)", label_b="BFGS (backtracking)",
                                              metric=:function_evaluations, tau_max=32.0,
                                              output="results/performance_profile_ffjm2_vs_bfgs_mgh.png")

Performance profile de Dolan e Moré comparando `method_a` e `method_b`
(colunas `method` do CSV MGH) nos problemas em comum entre os dois. `metric`
é o campo usado como custo por problema — qualquer coluna numérica de
`_read_mgh_csv` (`:function_evaluations` por padrão; também úteis:
`:gradient_evaluations`, `:iterations`, `:execution_time_seconds`, ou `:f`
para comparar a qualidade da solução em vez do custo — ver
[`performance_profiles_ffjm2_vs_bfgs_mgh`](@ref) para função/gradiente de uma vez).
`status_filter` (padrão `"gradient_converged"`, igual a
[`boxplot_ffjm2_vs_bfgs_mgh`](@ref)) decide o que conta como sucesso: uma
`String` exige esse `status` exato; `nothing` aceita qualquer `converged ==
true`, mais permissivo (ex.: BFGS pode ter `converged=true` com `status ==
"step_converged"`/`"function_converged"`, que não certificam
estacionariedade — usar `status_filter=nothing` conta esses como sucesso).
Um problema em que o método não teve sucesso conta como falha (custo
infinito) nesse problema — nunca é o melhor, e nunca ultrapassa `τ_max` no
perfil, mesmo com `metric=:f` (um `f` pequeno de uma execução sem sucesso
não conta). Um problema em que NENHUM dos dois teve sucesso (possível com
`status_filter` estrito: no CSV padrão, com `"gradient_converged"`, isso
acontece em MGH10 e MGH16, onde os dois só passam por critérios mais
fracos) conta como falha mútua para os dois, sem gerar erro — o problema
continua no denominador de ambos os perfis (convenção usual de Dolan-Moré),
só nunca é ultrapassado por nenhum dos dois em nenhum `τ` finito.

O eixo horizontal é `log2(τ)` (convenção original de Dolan-Moré), não `τ`
em escala log via `xscale` do Plots.jl, para não depender de suporte a
escala log2 de um backend específico. `τ_max` (padrão `32`, i.e.
`log2(τ_max)=5`) deve ser grande o suficiente para os dois métodos
atingirem seu patamar (a fração de problemas convergidos); aumente se as
curvas ainda estiverem subindo na borda direita do gráfico.

Não computa nada: os dados vêm de `rows` (vetor de `NamedTuple`s) ou
`csv_path` (mesmo formato de [`boxplot_ffjm2_vs_bfgs_mgh`](@ref); é erro não
passar nenhum dos dois). Salva a figura em `output` (`nothing` para não
salvar) e devolve `(; plot, taus, rho_a, rho_b, r_a, r_b, problems, metric)`.
"""
function performance_profile_ffjm2_vs_bfgs_mgh(;
    rows = nothing,
    csv_path::Union{AbstractString,Nothing} = "results/sr1_comparacao_ffjm2_bfgs_mgh.csv",
    method_a::AbstractString = "ffjm2_psb",
    method_b::AbstractString = "bfgs_backtracking2",
    label_a::AbstractString = "FFJM2 (PSB)",
    label_b::AbstractString = "BFGS (backtracking)",
    metric::Symbol = :function_evaluations,
    status_filter::Union{AbstractString,Nothing} = "gradient_converged",
    tau_max::Real = 32.0,
    output::Union{AbstractString,Nothing} = "results/performance_profile_ffjm2_vs_bfgs_mgh.png",
)
    if rows === nothing
        csv_path === nothing && error("Informe `rows` (resultados já calculados) ou `csv_path` (CSV com esses resultados).")
        rows = _read_mgh_csv(csv_path)
    end

    rows_a = filter(r -> r.method == method_a, rows)
    rows_b = filter(r -> r.method == method_b, rows)
    isempty(rows_a) && error("Nenhuma linha de method == \"$method_a\"")
    isempty(rows_b) && error("Nenhuma linha de method == \"$method_b\"")

    dict_a = Dict(r.problem => r for r in rows_a)
    dict_b = Dict(r.problem => r for r in rows_b)
    problems = sort(collect(intersect(Set(keys(dict_a)), Set(keys(dict_b)))))
    isempty(problems) && error("Nenhum problema em comum entre \"$method_a\" e \"$method_b\".")

    curve = _performance_profile_curve(dict_a, dict_b, problems, metric, tau_max, status_filter)

    p = plot(
        curve.log2_taus, curve.rho_a;
        seriestype = :steppost,
        xlabel = "log2(τ)",
        ylabel = "P(r ≤ τ)",
        label = label_a,
        linewidth = 2,
        legend = :bottomright,
        ylims = (0, 1.02),
    )
    plot!(p, curve.log2_taus, curve.rho_b; seriestype = :steppost, label = label_b, linewidth = 2)

    if output !== nothing
        mkpath(dirname(output))
        savefig(p, output)
        println("Performance profile salvo em: $output")
    end

    return (; plot = p, taus = 2.0 .^ curve.log2_taus, curve.rho_a, curve.rho_b, curve.r_a, curve.r_b, problems, metric)
end

"""
    performance_profiles_ffjm2_vs_bfgs_mgh(; rows=nothing, csv_path="results/sr1_comparacao_ffjm2_bfgs_mgh.csv",
                                               method_a="ffjm2_psb", method_b="bfgs_backtracking2",
                                               label_a="FFJM2 (PSB)", label_b="BFGS (backtracking)",
                                               tau_max=32.0,
                                               output="results/performance_profiles_ffjm2_vs_bfgs_mgh.png")

Mesma ideia de [`performance_profile_ffjm2_vs_bfgs_mgh`](@ref) (mesmo
`status_filter`, padrão `"gradient_converged"` — ver sua docstring), mas
gera as três curvas de uma vez, uma por painel (`layout=(1,3)`, mesmo
esquema de `boxplot_ffjm2_vs_bfgs_mgh`): valor final de `f` (qualidade da
solução, não custo), avaliações de função e avaliações de gradiente,
nessa ordem. `τ_max` é compartilhado pelas três; o painel de `f` tende a
não atingir seu patamar com o `τ_max` padrão (`32`), já que os valores de
`f` convergidos variam em ordens de grandeza muito maiores do que
contagens de avaliações (ver a nota na docstring da função de painel
único) — aumente `τ_max` ou chame `performance_profile_ffjm2_vs_bfgs_mgh(metric=:f, tau_max=...)`
isoladamente se quiser ver esse painel saturar.

Devolve `(; plot, curves)`, onde `curves` é um `NamedTuple` com uma entrada
por métrica (`f`, `function_evaluations`, `gradient_evaluations`), cada
uma com os mesmos campos de retorno da função de painel único (exceto
`plot`).
"""
function performance_profiles_ffjm2_vs_bfgs_mgh(;
    rows = nothing,
    csv_path::Union{AbstractString,Nothing} = "results/sr1_comparacao_ffjm2_bfgs_mgh.csv",
    method_a::AbstractString = "ffjm2_psb",
    method_b::AbstractString = "bfgs_backtracking2",
    label_a::AbstractString = "FFJM2 (PSB)",
    label_b::AbstractString = "BFGS (backtracking)",
    status_filter::Union{AbstractString,Nothing} = "gradient_converged",
    tau_max::Real = 32.0,
    output::Union{AbstractString,Nothing} = "results/performance_profiles_ffjm2_vs_bfgs_mgh.png",
)
    if rows === nothing
        csv_path === nothing && error("Informe `rows` (resultados já calculados) ou `csv_path` (CSV com esses resultados).")
        rows = _read_mgh_csv(csv_path)
    end

    rows_a = filter(r -> r.method == method_a, rows)
    rows_b = filter(r -> r.method == method_b, rows)
    isempty(rows_a) && error("Nenhuma linha de method == \"$method_a\"")
    isempty(rows_b) && error("Nenhuma linha de method == \"$method_b\"")

    dict_a = Dict(r.problem => r for r in rows_a)
    dict_b = Dict(r.problem => r for r in rows_b)
    problems = sort(collect(intersect(Set(keys(dict_a)), Set(keys(dict_b)))))
    isempty(problems) && error("Nenhum problema em comum entre \"$method_a\" e \"$method_b\".")

    metrics = (:f, :function_evaluations, :gradient_evaluations)
    titles = ("Objective value f", "Function evaluations", "Gradient evaluations")

    curves = NamedTuple[]
    panels = map(zip(metrics, titles)) do (metric, title)
        curve = _performance_profile_curve(dict_a, dict_b, problems, metric, tau_max, status_filter)
        push!(curves, (; metric, curve...))

        panel = plot(
            curve.log2_taus, curve.rho_a;
            seriestype = :steppost, title, xlabel = "log2(τ)", ylabel = "P(r ≤ τ)",
            label = label_a, linewidth = 2, legend = :bottomright, ylims = (0, 1.02),
        )
        plot!(panel, curve.log2_taus, curve.rho_b; seriestype = :steppost, label = label_b, linewidth = 2)
        panel
    end

    fig = plot(panels...; layout = (1, 3), size = (1500, 450), margin = 5Plots.mm)

    if output !== nothing
        mkpath(dirname(output))
        savefig(fig, output)
        println("Performance profiles salvos em: $output")
    end

    return (; plot = fig, curves = Tuple(curves), problems)
end

# ==============================================================================
# Sobrepõe, no heatmap de RMSD de `plot_assimilacao_heatmap_tend_31_latex`
# (`scripts/main.jl` — precisa estar carregado para as duas funções abaixo
# funcionarem, já que constrói o fundo heatmap/curvas de nível), o caminho de
# pontos aceitos de cada solver (BFGS, BOBYQA, ffjm2), lido do CSV
# `method,point_index,x` gerado por `_comparar_solvers`/`comparar_solvers_real`
# (`sv_teste_pregenered.jl`) — mesma ideia de
# `plot_assimilacao_heatmap_tend_31_latex_com_bfgs_aceitos`, mas generalizada
# para os três solvers de uma vez, direto do CSV em vez de logs de texto.
# Assume dimensão 2 (`x = [n1, n2]`), como o heatmap de fundo.
# ==============================================================================

function le_pontos_aceitos_solvers(pontos_csv = "results/comparacao_solvers_real_dim2_pontos.csv")
    linhas = readlines(pontos_csv)
    isempty(linhas) && error("Arquivo vazio: $pontos_csv")

    pontos_por_metodo = Dict{String,Vector{Vector{Float64}}}()
    ordem_metodos = String[]

    for linha in linhas[2:end]
        isempty(strip(linha)) && continue
        m = match(r"^([^,]+),(\d+),\"?(\[[^\]]*\])\"?$", linha)
        m === nothing && error("Linha invalida em $pontos_csv: $linha")

        metodo = String(m.captures[1])
        x = parse.(Float64, split(strip(m.captures[3], ['[', ']']), ','))

        if !haskey(pontos_por_metodo, metodo)
            pontos_por_metodo[metodo] = Vector{Vector{Float64}}()
            push!(ordem_metodos, metodo)
        end
        push!(pontos_por_metodo[metodo], x)
    end

    return (; pontos_csv, pontos_por_metodo, metodos = ordem_metodos)
end

"""
    plot_assimilacao_heatmap_tend_31_latex_com_solvers(; pontos_csv, output, kwargs...)

Plota o caminho de pontos aceitos de cada solver (BFGS, BOBYQA, ffjm2) sobre
o heatmap/curvas de nível de RMSD de
`plot_assimilacao_heatmap_tend_31_latex` (`scripts/main.jl`, precisa estar
carregado), lidos de `pontos_csv` (formato `method,point_index,x` gerado por
`comparar_solvers_real`, `sv_teste_pregenered.jl`, tipicamente
`results/comparacao_solvers_real_dim2_pontos.csv`). Salva em `output`
(padrão `results/plot_calor_novo.pdf`). `kwargs...` são repassados a
`plot_assimilacao_heatmap_tend_31_latex` (`matrix_output`, `rmsd_max`,
`tamanho`).
"""
function plot_assimilacao_heatmap_tend_31_latex_com_solvers(;
    pontos_csv = "results/comparacao_solvers_real_dim2_pontos.csv",
    output = "results/plot_calor_novo.pdf",
    kwargs...,
)
    base = plot_assimilacao_heatmap_tend_31_latex(; output, kwargs...)
    lidos = le_pontos_aceitos_solvers(pontos_csv)
    # `base.plot` já vem com as curvas de nível desenhadas (`contourf`
    # dentro de `plot_assimilacao_heatmap_tend_31_latex`) — nenhum
    # `contour!` adicional é necessário aqui.

    estilos = Dict(
        "BFGS" => (cor = :red, marcador = :circle),
        "BOBYQA" => (cor = :blue, marcador = :diamond),
        "ffjm2" => (cor = :green, marcador = :utriangle),
    )
    cores_extra = [:orange, :purple, :brown, :gray]

    caminhos = NamedTuple[]
    for (j, metodo) in enumerate(lidos.metodos)
        pontos = lidos.pontos_por_metodo[metodo]
        if isempty(pontos)
            @warn "Sem pontos aceitos para $metodo; ignorando." pontos_csv
            continue
        end

        estilo = get(estilos, metodo, (cor = cores_extra[mod1(j, length(cores_extra))], marcador = :hexagon))
        xs = [p[1] for p in pontos]
        ys = [p[2] for p in pontos]

        plot!(
            base.plot,
            xs,
            ys;
            color = estilo.cor,
            linewidth = 2,
            marker = estilo.marcador,
            markersize = 4,
            markercolor = estilo.cor,
            markerstrokecolor = :white,
            label = "$metodo path",
        )

        scatter!(
            base.plot,
            [xs[1]],
            [ys[1]];
            marker = :diamond,
            markersize = 6,
            markercolor = :white,
            markerstrokecolor = estilo.cor,
            markerstrokewidth = 1.5,
            label = false,
        )

        scatter!(
            base.plot,
            [xs[end]],
            [ys[end]];
            marker = :star5,
            markersize = 8,
            markercolor = estilo.cor,
            markerstrokecolor = :white,
            markerstrokewidth = 1.5,
            label = false,
        )

        push!(caminhos, (; metodo, x = xs, y = ys))
    end

    isempty(caminhos) && @warn "Nenhum caminho de solver foi plotado." pontos_csv

    mkpath(dirname(output))
    savefig(base.plot, output)

    return merge(base, (; output, pontos_csv, caminhos))
end

# ==============================================================================
# Só o plot do heatmap "a priori" do experimento gêmeo (dimensão 2), a
# partir do CSV já gerado por `assimilation_rmsd_heatmap_pregerado`
# (`sv_teste_pregenered.jl`) — sem recalcular a grade (`le_assimilation_heatmap_matrix`,
# `assimilacao.jl`, precisa estar carregado). Mesmo estilo visual de
# `plot_assimilacao_heatmap_tend_31_latex` (`scripts/main.jl`), mas marca o
# `x_otimo` VERDADEIRO com a estrela, já que no experimento gêmeo ele é
# conhecido por construção — ao contrário do heatmap de dados reais, cujo
# mínimo marcado é só o empírico da grade.
# ==============================================================================

# `_desenha_curvas_nivel!` (curvas de nível calculadas "na mão" via
# `Contour.jl`) foi removida junto com o backend `pgfplotsx()`: era só um
# workaround para o recipe `contour!`/`contourf` do Plots.jl não renderizar
# linhas de forma confiável nesse backend. No `gr()` atual, `contour!`
# nativo funciona direito — ver `plot_assimilation_rmsd_heatmap_pregerado`
# e `plot_assimilacao_heatmap_tend_31_latex` (`scripts/main.jl`) abaixo.

"""
    plot_assimilation_rmsd_heatmap_pregerado(; matrix_output="results/assimilacao_heatmap_pregerado_tend_31.csv",
                                                 x_otimo, output="results/assimilacao_heatmap_pregerado_tend_31.pdf",
                                                 rmsd_max=3.0, tamanho=(700,600))

Plota o heatmap/curvas de nível de RMSD do CSV já gerado por
`assimilation_rmsd_heatmap_pregerado`, sem rodar a varredura de novo.
`x_otimo` é o ótimo verdadeiro usado para gerar aquele CSV (precisa ser
passado aqui de novo, já que só o RMSD é salvo no CSV, não `x_otimo`) e é
marcado com uma estrela branca. Salva em `output` e devolve `(; plot,
output, data, x_otimo)`.
"""
function plot_assimilation_rmsd_heatmap_pregerado(;
    matrix_output::AbstractString = "results/assimilacao_heatmap_pregerado_tend_31.csv",
    x_otimo::AbstractVector,
    output::AbstractString = "results/assimilacao_heatmap_pregerado_tend_31.pdf",
    rmsd_max::Real = 2.0,
    tamanho = (700, 600),
)
    length(x_otimo) == 2 ||
        throw(ArgumentError("x_otimo deve ter dimensão 2 (grade 2D de n1 × n2)"))

    heat = le_assimilation_heatmap_matrix(matrix_output)
    Z_plot = map(v -> isfinite(v) && v <= rmsd_max ? v : NaN, heat.RMSD)

    # Backend `gr()` (global, topo do arquivo): `contour!` nativo do
    # Plots.jl funciona direito aqui (ao contrário do `pgfplotsx()`, usado
    # antes, cujo recipe de contorno não renderizava as linhas de forma
    # confiável).
    p = heatmap(
        heat.n1,
        heat.n2,
        Z_plot;
        xlabel = "Manning coefficient x1",
        ylabel = "Manning coefficient x2",
        colorbar_title = "RMSD",
        clim = (0.0, rmsd_max),
        aspect_ratio = :equal,
        xlims = (minimum(heat.n1), maximum(heat.n1)),
        ylims = (minimum(heat.n2), maximum(heat.n2)),
        color = :viridis,
        background_color = :white,
        background_color_inside = :gold,
        size = tamanho,
        legend = :topleft,
    )

    scatter!(
        p,
        [NaN],
        [NaN];
        markershape = :rect,
        markersize = 14,
        markercolor = :gold,
        markerstrokecolor = :black,
        label = "Yellow region = NaN",
    )

    contour!(
        p,
        heat.n1,
        heat.n2,
        Z_plot;
        levels = range(0.0, rmsd_max, length = 13),
        linecolor = :black,
        linewidth = 1.0,
        colorbar_entry = false,
    )

    scatter!(
        p,
        [x_otimo[1]],
        [x_otimo[2]];
        marker = :star5,
        markersize = 10,
        markercolor = :white,
        markerstrokecolor = :black,
        markerstrokewidth = 1.5,
        label = "x* = ($(round(x_otimo[1], digits=4)), $(round(x_otimo[2], digits=4)))",
    )

    mkpath(dirname(output))
    savefig(p, output)
    println("Heatmap do experimento gêmeo salvo em: $output")

    return (; plot = p, output, data = heat, x_otimo)
end

# ==============================================================================
# Igual a `plot_assimilacao_heatmap_tend_31_latex_com_solvers` acima, mas para
# o experimento gêmeo (dados sintéticos): sobrepõe, no heatmap de
# `plot_assimilation_rmsd_heatmap_pregerado`, o caminho de pontos aceitos de
# cada solver (BFGS, BOBYQA, ffjm2), lido do CSV `method,point_index,x`
# gerado por `comparar_solvers_pregerado`/`comparar_solvers_twin_dim2`
# (`sv_teste_pregenered.jl`). Assume dimensão 2, como o heatmap de fundo.
# ==============================================================================

"""
    plot_assimilation_rmsd_heatmap_pregerado_com_solvers(; x_otimo, pontos_csv, output, kwargs...)

Plota o caminho de pontos aceitos de cada solver (BFGS, BOBYQA, ffjm2) sobre
o heatmap/curvas de nível de RMSD do experimento gêmeo de
`plot_assimilation_rmsd_heatmap_pregerado`, lidos de `pontos_csv` (formato
`method,point_index,x` gerado por `comparar_solvers_pregerado`,
`sv_teste_pregenered.jl`, tipicamente
`results/comparacao_solvers_twin_dim2_pontos.csv`). `x_otimo` é o ótimo
verdadeiro usado para gerar o heatmap/os dados sintéticos (mesmo valor
passado a `assimilation_rmsd_heatmap_pregerado`/`comparar_solvers_twin_dim2`)
e é marcado com uma estrela branca pelo heatmap de base. Salva em `output`
(padrão `results/plot_calor_pregerado_com_solvers.pdf`). `kwargs...` são
repassados a `plot_assimilation_rmsd_heatmap_pregerado` (`matrix_output`,
`rmsd_max`, `tamanho`).
"""
function plot_assimilation_rmsd_heatmap_pregerado_com_solvers(;
    x_otimo::AbstractVector,
    pontos_csv = "results/comparacao_solvers_twin_dim2_pontos.csv",
    output = "results/plot_calor_pregerado_com_solvers.pdf",
    kwargs...,
)
    base = plot_assimilation_rmsd_heatmap_pregerado(; x_otimo, output, kwargs...)
    lidos = le_pontos_aceitos_solvers(pontos_csv)
    # `base.plot` já vem com as curvas de nível desenhadas (dentro de
    # `plot_assimilation_rmsd_heatmap_pregerado`) — nenhum `contour!`
    # adicional é necessário aqui.

    estilos = Dict(
        "BFGS" => (cor = :red, marcador = :circle),
        "BOBYQA" => (cor = :blue, marcador = :diamond),
        "ffjm2" => (cor = :green, marcador = :utriangle),
    )
    cores_extra = [:orange, :purple, :brown, :gray]

    caminhos = NamedTuple[]
    for (j, metodo) in enumerate(lidos.metodos)
        pontos = lidos.pontos_por_metodo[metodo]
        if isempty(pontos)
            @warn "Sem pontos aceitos para $metodo; ignorando." pontos_csv
            continue
        end

        estilo = get(estilos, metodo, (cor = cores_extra[mod1(j, length(cores_extra))], marcador = :hexagon))
        xs = [p[1] for p in pontos]
        ys = [p[2] for p in pontos]

        plot!(
            base.plot,
            xs,
            ys;
            color = estilo.cor,
            linewidth = 2,
            marker = estilo.marcador,
            markersize = 4,
            markercolor = estilo.cor,
            markerstrokecolor = :white,
            label = "$metodo",
        )

        scatter!(
            base.plot,
            [xs[1]],
            [ys[1]];
            marker = :diamond,
            markersize = 6,
            markercolor = :white,
            markerstrokecolor = estilo.cor,
            markerstrokewidth = 1.5,
            label = false,
        )

        scatter!(
            base.plot,
            [xs[end]],
            [ys[end]];
            marker = :star5,
            markersize = 8,
            markercolor = estilo.cor,
            markerstrokecolor = :white,
            markerstrokewidth = 1.5,
            label = false,
        )

        push!(caminhos, (; metodo, x = xs, y = ys))
    end

    isempty(caminhos) && @warn "Nenhum caminho de solver foi plotado." pontos_csv

    mkpath(dirname(output))
    savefig(base.plot, output)

    return merge(base, (; output, pontos_csv, caminhos))
end

# ==============================================================================
# Custo computacional (segundos) de avaliar o gradiente e a Hessiana de
# F_residual por ForwardDiff, em função da dimensão de entrada — dados
# gerados por `benchmark_forwarddiff_sv` (`ffjm2.jl`), CSV com colunas
# `dim,metric,seconds,value` (`metric` ∈ `"f"`, `"gradient"`, `"hessian"`).
# ==============================================================================

function _read_benchmark_forwarddiff_csv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("CSV vazio: $path")
    header = split(lines[1], ',')
    idx = Dict(name => i for (i, name) in enumerate(header))

    rows = NamedTuple[]
    for line in lines[2:end]
        isempty(line) && continue
        fields = split(line, ',')
        push!(rows, (;
            dim = parse(Int, fields[idx["dim"]]),
            metric = fields[idx["metric"]],
            seconds = parse(Float64, fields[idx["seconds"]]),
            value = parse(Float64, fields[idx["value"]]),
        ))
    end
    return rows
end

"""
    plot_benchmark_forwarddiff_tempo(; data_output="results/benchmark_forwarddiff_sv.csv",
                                         output="results/benchmark_forwarddiff_sv_tempo.pdf",
                                         f_cost_seconds=12.0)

Plota o custo (escala log) de avaliar o gradiente e a Hessiana de
`F_residual` por `ForwardDiff`, em função da dimensão de entrada, a partir
do CSV escrito por `benchmark_forwarddiff_sv` (`ffjm2.jl`, colunas
`dim,metric,seconds,value`). Em vez do tempo em segundos, o eixo vertical é
o custo relativo a uma avaliação de `f` (`seconds / f_cost_seconds`,
"avaliações de f equivalentes"): `f_cost_seconds` (padrão `12.0`) é o custo
de uma avaliação de `F_residual`, medido nas próprias linhas `metric ==
"f"` do CSV (≈11.8–11.9s no benchmark salvo). Essa normalização é mais
direta de interpretar do que segundos brutos, já que o custo de otimizar é
naturalmente contado em número de avaliações de função.

Cada combinação (dimensão, métrica) naquele CSV é uma única avaliação (sem
repetições), o benchmark não mede variância, então o gráfico é uma curva
simples por métrica, não um boxplot. Salva a figura em `output` e devolve
`(; plot, rows, dims_grad, custo_grad, dims_hess, custo_hess)`.
"""
function plot_benchmark_forwarddiff_tempo(;
    data_output::AbstractString = "results/benchmark_forwarddiff_sv.csv",
    output::AbstractString = "results/benchmark_forwarddiff_sv_tempo.pdf",
    f_cost_seconds::Real = 12.0,
)
    rows = _read_benchmark_forwarddiff_csv(data_output)

    rows_grad = filter(r -> r.metric == "gradient", rows)
    rows_hess = filter(r -> r.metric == "hessian", rows)
    isempty(rows_grad) && error("Nenhuma linha com metric == \"gradient\" em $data_output")
    isempty(rows_hess) && error("Nenhuma linha com metric == \"hessian\" em $data_output")

    ord_grad = sortperm(getproperty.(rows_grad, :dim))
    ord_hess = sortperm(getproperty.(rows_hess, :dim))

    dims_grad = getproperty.(rows_grad, :dim)[ord_grad]
    custo_grad = getproperty.(rows_grad, :seconds)[ord_grad] ./ f_cost_seconds
    dims_hess = getproperty.(rows_hess, :dim)[ord_hess]
    custo_hess = getproperty.(rows_hess, :seconds)[ord_hess] ./ f_cost_seconds

    p = plot(
        dims_grad,
        custo_grad;
        xlabel = "Dimension",
        ylabel = "Cost (equivalent f evaluations)",
        label = "Gradient (ForwardDiff)",
        marker = :circle,
        linewidth = 2,
        yscale = :log10,
        legend = :topleft,
    )

    plot!(
        p,
        dims_hess,
        custo_hess;
        label = "Hessian (ForwardDiff)",
        marker = :square,
        linewidth = 2,
    )

    mkpath(dirname(output))
    savefig(p, output)
    println("Custo de derivadas por ForwardDiff (em avaliações de f) salvo em: $output")

    return (; plot = p, rows, dims_grad, custo_grad, dims_hess, custo_hess)
end

# ==============================================================================
# Ajuste do modelo aos dados reais nas duas estações de medição (x=751 e
# x=3256), no estilo de `_research/ffjmfig12.png`, para o minimizador de
# cada método salvo por `comparar_solvers_real`/`comparar_solvers_real_dim10`
# (`sv_teste_pregenered.jl`, CSV com colunas `method,...,minimizer`).
#
# A simulação em si (`sv_fork_estacoes`, `sv_fork.jl` — precisa estar
# carregado, ex.: via `sv_teste_pregenered.jl`) roda o MESMO solver usado
# para gerar esses resultados (`sv_fork_assimilation`, chamado
# posicionalmente `(ng, tbeg, tend, nothing)` em `sv_fork.jl`, não a versão
# de `sv_fork_new.jl`/`assimilacao.jl`, que usa argumentos nomeados e não é
# a que `comparar_solvers_real` de fato invoca) e devolve as séries brutas
# simulada/observada em cada estação, em vez do resíduo já diferenciado
# usado para otimização.
# ==============================================================================

function _read_comparacao_solvers_csv(path::AbstractString)
    linhas = readlines(path)
    isempty(linhas) && error("CSV vazio: $path")

    rows = NamedTuple[]
    for linha in linhas[2:end]
        isempty(strip(linha)) && continue
        m_method = match(r"^([^,]+),", linha)
        m_vec = match(r"\[[^\]]*\]", linha)
        (m_method === nothing || m_vec === nothing) && continue

        metodo = String(m_method.captures[1])
        minimizer = parse.(Float64, split(strip(m_vec.match, ['[', ']']), ','))
        push!(rows, (; method = metodo, minimizer))
    end
    isempty(rows) && error("Nenhuma linha \"method,...,minimizer\" reconhecida em $path")
    return rows
end

"""
    plot_ajuste_estacoes_solvers(; csv_path="results/comparacao_solvers_real_dim10.csv",
                                     tbeg=0.0, tend=31.0,
                                     output_prefix="results/ajuste_estacoes_real_dim10",
                                     step=20)

Para cada método (linha) do CSV escrito por `comparar_solvers_real`/
`comparar_solvers_real_dim10`/`comparar_solvers_real_dim2`
(`sv_teste_pregenered.jl`), roda a simulação real (`sv_fork_estacoes`,
`sv_fork.jl`) no minimizador daquele método e plota, numa única figura, o
ajuste do modelo aos dados observados nas duas estações de medição
(`x=751`, `x=3256`) — mesmo estilo de `_research/ffjmfig12.png`: triângulos
azuis conectados por linha para a série simulada em `x=751` (rótulo
`m=751`), x's vermelhos conectados por linha para `x=3256` (`m=3256`), e
pontos pretos para os dados observados nas duas estações (`step` controla
o espaçamento dos marcadores, a linha em si usa todos os pontos).

Salva uma figura por método, em `"\$(output_prefix)_\$(method).pdf"`.
Emite um aviso (mas não falha) se a simulação de algum método divergiu
antes de `tend` (`sim.ok == false`), caso em que a série daquele método
pode estar incompleta. Devolve um `Vector` de `NamedTuple`s `(; method,
minimizer, plot, output, t, z751_sim, z751_obs, z3256_sim, z3256_obs)`.
"""
function plot_ajuste_estacoes_solvers(;
    csv_path::AbstractString = "results/comparacao_solvers_real_dim10.csv",
    tbeg::Real = 0.0,
    tend::Real = 31.0,
    output_prefix::AbstractString = "results/ajuste_estacoes_real_dim10",
    step::Integer = 20,
)
    rows = _read_comparacao_solvers_csv(csv_path)

    resultados = NamedTuple[]
    for row in rows
        println("Rodando simulação real para $(row.method) (dim=$(length(row.minimizer)))...")
        sim = sv_fork_estacoes(row.minimizer, tbeg, tend)
        sim.ok || @warn "Simulação divergiu antes de tend=$tend para $(row.method); série pode estar incompleta." csv_path

        n = length(sim.t)
        if n < 2
            @warn "Simulação de $(row.method) não produziu pontos suficientes para plotar (n=$n); pulando." csv_path
            push!(resultados, (;
                method = row.method, minimizer = row.minimizer, plot = nothing, output = nothing,
                t = sim.t, z751_sim = sim.z751_sim, z751_obs = sim.z751_obs,
                z3256_sim = sim.z3256_sim, z3256_obs = sim.z3256_obs,
            ))
            continue
        end
        indices = 1:step:n
        head = 1:min(2, n)

        p = plot(
            sim.t, sim.z751_sim;
            label = "", linewidth = 2, color = :cornflowerblue,
            xlims = (tbeg, tend),
        )
        plot!(
            p, sim.t[head], sim.z751_sim[head];
            label = "m=751", linewidth = 2, markershape = :utriangle,
            markerstrokewidth = 0, color = :cornflowerblue, legend = (0.0, 0.6),
        )
        scatter!(
            p, sim.t[indices], sim.z751_sim[indices];
            label = "", markershape = :utriangle, markersize = 4,
            markercolor = :cornflowerblue, markerstrokewidth = 0,
        )

        plot!(p, sim.t, sim.z3256_sim; label = "", ylabel = "m", xlabel = "t", color = :red)
        plot!(
            p, sim.t[head], sim.z3256_sim[head];
            label = "m=3256", linewidth = 2, markershape = :x,
            markersize = 4, color = :red,
        )
        scatter!(
            p, sim.t[indices], sim.z3256_sim[indices];
            label = "", markershape = :x, markersize = 4,
            markercolor = :red, markerstrokewidth = 1.5,
        )

        scatter!(p, sim.t, sim.z3256_obs; label = "Data", markersize = 1.0, color = :black)
        scatter!(p, sim.t, sim.z751_obs; label = "", markersize = 1.0, color = :black)

        output = "$(output_prefix)_$(row.method).pdf"
        mkpath(dirname(output))
        savefig(p, output)
        println("Ajuste de $(row.method) salvo em: $output")

        push!(resultados, (;
            method = row.method, minimizer = row.minimizer, plot = p, output,
            t = sim.t, z751_sim = sim.z751_sim, z751_obs = sim.z751_obs,
            z3256_sim = sim.z3256_sim, z3256_obs = sim.z3256_obs,
        ))
    end

    return resultados
end
