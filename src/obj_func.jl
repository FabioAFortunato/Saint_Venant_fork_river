
using ForwardDiff

include("sv_fork.jl")
function soma_desvio_quadratico(x::AbstractVector{T}) where T<:Real
    n_man = length(x) - 1
    x_ref = x[n_man + 1]
    return sum((x[i] - x_ref)^2 for i in 1:n_man)
end

function quad_fun(x::AbstractVector{T}; tmax = 5.0) where T<:Real
    global n_calfun, nome, inicio_s

    # Detecta se é avaliação com derivada (ForwardDiff)
    is_dual = T <: ForwardDiff.Dual

    # PRINT LOGO NO COMEÇO
    if is_dual
        println(">>> Avaliação com DERIVADA (ForwardDiff)")
    else
        println(">>> Avaliação NORMAL (sem derivada)")
    end

    # 1. Ajuste de dimensão
    n_man = length(x)
    
    # 2. Chamada da simulação
    local z_erro = sv_fork(x[1:n_man], tmax)
    
    # Só incrementa contador em avaliação real
    if !is_dual
        n_calfun += 1
    end
    
    # Proteção contra NaN
    if any(isnan, z_erro)
        return T(1e27)
    end

    # Cálculo do erro quadrático e do RMSD correspondente
    erro_quadratico_total = sum(abs2, z_erro)
    RMSD = sqrt(erro_quadratico_total / length(z_erro))
    val_RMSD = ForwardDiff.value(RMSD)

    # Logs
    if !is_dual
        tempo_s = time() - inicio_s
        println("RMSD = $val_RMSD")

        open(nome, "a") do file
            write(file, "$(round(tempo_s, digits=3))\t$n_calfun\t$val_RMSD\t$erro_quadratico_total\n")
        end
    end

    return erro_quadratico_total
end



"""
Função Lagrangeana Aumentada L(x; lambda, rho)
Modificada para receber os multiplicadores e o peso da penalidade do loop externo.
"""
function L_sv(x::AbstractVector{T}, lambda::AbstractVector{Float64}, rho::Float64, tmax) where T<:Real
    global n_calfun, nome, inicio_s
    
    # 1. n_man dinâmico baseado no vetor de decisão
    n_man = length(x) - 1 
    
    # 2. Chamada da função de restrição (sv_fork deve aceitar Dual se x for Dual)
    local z_erro = sv_fork(x[1:n_man], tmax)
    
    # Verifica se estamos num cálculo de gradiente via ForwardDiff
    is_dual = T <: ForwardDiff.Dual
    
    # Só incrementa o contador global em avaliações de ponto real
    if !is_dual
        n_calfun += 1
    end
    
    # Se houver erro numérico nas restrições, retorna um valor de penalidade alto
    if any(isnan, z_erro)
        return T(1e27) 
    end

    # --- PARTE 1: Função Objetivo Original f(x) ---
    # f = sum (x_i - x_ref)^2
    f_obj = soma_desvio_quadratico(x)

    # --- PARTE 2: Termo de Penalidade e Multiplicadores ---
    # A fórmula do Lagrangeano Aumentado para c(x) = 0 é:
    # L = f(x) + Σ [λ_i * c_i(x) + (ρ/2) * c_i(x)^2]
    # Que pode ser escrita como: f(x) + (ρ/2) * Σ [c_i(x) + λ_i/ρ]^2
    
    termo_lagrange = 0.0
    ya = length(z_erro) ÷ 2 # Ajuste conforme a estrutura do seu sv_fork
    
    for i = 1:length(z_erro)
        # Usamos a forma compacta da parábola deslocada
        ajuste = z_erro[i] + (lambda[i] / rho)
        termo_lagrange += ajuste * ajuste
    end

    # Valor total da Lagrangeana
    L_total = f_obj + (rho / 2.0 * termo_lagrange)

    # --- LOGS E MONITORAMENTO ---
    if !is_dual
        tempo_s = time() - inicio_s
        RMSD = sqrt(sum(abs2, z_erro) / length(z_erro))
        val_RMSD = ForwardDiff.value(RMSD) 
        val_f_obj = ForwardDiff.value(f_obj)
        
        # Opcional: imprimir progresso
        println("Iter: $n_calfun | RMSD: $val_RMSD | rho: $rho")
        
        open(nome, "a") do file
            write(file, "$(round(tempo_s, digits=3))\t$n_calfun\t$val_RMSD\t$val_f_obj\n")
        end
    end

    return L_total
end

