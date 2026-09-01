using HypothesisTests
using PGFPlotsX
using Plots
using Statistics

pgfplotsx()

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
