
function busca_perturbacao_segura(
    f,
    x_atual;
    taxa_inicial=0.05,
    taxa_min=0.001,
    tentativas_por_taxa=500,
    lb=0.0,
    ub=1.0,
)
    n = length(x_atual)
    taxa = taxa_inicial
    limite_inferior = lb .* ones(n)
    limite_superior = ub .* ones(n)

    while taxa >= taxa_min
        println("Tentando perturbacoes com taxa = ", taxa)

        for tentativa in 1:tentativas_por_taxa
            perturbacao = (2.0 .* rand(n) .- 1.0) .* taxa
            x_temp = clamp.(x_atual .+ perturbacao, limite_inferior, limite_superior)
            f_val = f(x_temp)

            if isfinite(f_val) && !isnan(f_val) && f_val <= 1000.0
                println("Perturbacao segura encontrada na tentativa ", tentativa, " | f = ", f_val)
                return x_temp
            end
        end

        taxa /= 2.0
    end

    println("Aviso: Nenhuma perturbacao segura encontrada. Retornando ponto do BFGS.")
    return x_atual
end




function arquivo_em_results(arquivo)
    mkpath(RESULTS_DIR)
    return joinpath(RESULTS_DIR, basename(String(arquivo)))
end

nome = arquivo_em_results("teste_bfgs_bobyqa_L_sv_5.txt")

function label_tmax(tmax)
    tmax_float = Float64(tmax)
    return isinteger(tmax_float) ? string(Int(round(tmax_float))) : replace(string(tmax_float), "." => "_")
end


function expande_dimensao(x_antigo, novo_n_man)
    n_antigo = length(x_antigo) - 1
    
    # Separar a parte espacial do Ãºltimo parÃ¢metro
    espaco_antigo = x_antigo[1:n_antigo]
    parametro_final = x_antigo[end]
    
    # Criar malhas virtuais (de 0.0 a 1.0) para mapear o rio
    malha_antiga = range(0.0, 1.0, length=n_antigo)
    malha_nova = range(0.0, 1.0, length=novo_n_man)
    
    # Fazer a interpolaÃ§Ã£o linear
    itp = LinearInterpolation(malha_antiga, espaco_antigo)
    espaco_novo = itp.(malha_nova)
    
    # Juntar a nova parte espacial com o parÃ¢metro final
    return vcat(espaco_novo, parametro_final)
end


# ng = fill(0.08, n_man + 1)
# run_L_sv_bfgs_bobyqa_tmax = true
# result = run_L_sv_bfgs_bobyqa_tmax ? testar_L_sv_bfgs_bobyqa_tmax(ng) : ng
