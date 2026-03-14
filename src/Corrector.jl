# src/Corrector.jl

function DivergenceCorrector(
    sys_params::SystemParameters{T}, 
    chain_params::ChainParameters{T}, 
    grid::RadialGrid{T}, 
    sim_dist::T=10.0
) where {T}

    N_sites = sys_params.N_sites
    begin_idx = zeros(Int, N_sites, N_sites)
    end_idx = zeros(Int, N_sites, N_sites)
    
    for i in 1:N_sites, j in 1:N_sites
        σ_ij = 0.5 * (chain_params.σ[i] + chain_params.σ[j])
        begin_idx[i, j] = round(Int, σ_ij / grid.Δr) 
        end_idx[i, j]   = round(Int, sim_dist * σ_ij / grid.Δr)
    end
    
    sum_rules = [
        T[1.0 -1.0; -1.0 1.0], 
        T[1.0  0.0; -1.0 0.0],
        T[1.0 -1.0;  0.0 0.0],
        T[1.0  0.0;  0.0 -1.0]
    ]
    sum_orders =[2, 0, 0, 0]
    
    q_len = 0
    for i in 1:N_sites, j in 1:N_sites
        q_len += (end_idx[i, j] - begin_idx[i, j] + 1)
    end
    
    q_matrix = zeros(T, length(sum_orders), q_len)
    
    for g in 1:length(sum_orders)
        n = sum_orders[g]
        sign_val = n == 2 ? T(-1.0) : T(1.0)
        prefactor = sign_val * (T(4.0) * T(π) * grid.Δr) / (n + 1) 
        
        col_offset = 0
        for i in 1:N_sites, j in 1:N_sites
            len = end_idx[i, j] - begin_idx[i, j] + 1
            for k in 1:len
                r_idx = begin_idx[i, j] + k - 1
                r_val = grid.r[r_idx]
                q_matrix[g, col_offset + k] = prefactor * (r_val^(n + 2)) * sum_rules[g][i, j]
            end
            col_offset += len
        end
    end
    
    println("Precomputing Moore-Penrose Pseudoinverse for Corrector...")
    pinv_q = pinv(q_matrix)
    
    return DivergenceCorrector{T}(pinv_q, sum_rules, sum_orders, begin_idx, end_idx)
end

function correct_h!(
    h_fixed::Array{T,3}, 
    h_sim::Array{T,3}, 
    corrector::DivergenceCorrector{T}, 
    grid::RadialGrid{T}
) where {T}

    N_sites = size(h_sim, 1)
    num_rules = length(corrector.sum_orders)
    RHS = zeros(T, num_rules)
    
    for g in 1:num_rules
        n = corrector.sum_orders[g]
        for i in 1:N_sites, j in 1:N_sites
            h_data = @view h_sim[i, j, :]
            moment_val = h_moment(h_data, n, grid)
            RHS[g] += -(corrector.sum_rules[g][i, j] * moment_val)
        end
    end
    
    q_vec = corrector.pinv_q * RHS
    h_fixed .= h_sim
    
    col_offset = 0
    for i in 1:N_sites, j in 1:N_sites
        len = corrector.end_idx[i, j] - corrector.begin_idx[i, j] + 1
        for k in 1:len
            r_idx = corrector.begin_idx[i, j] + k - 1
            h_fixed[i, j, r_idx] += q_vec[col_offset + k]
        end
        col_offset += len
    end
end