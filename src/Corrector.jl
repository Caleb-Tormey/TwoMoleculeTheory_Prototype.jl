# src/Corrector.jl

# src/Corrector.jl

function DivergenceCorrector(
    sys_params::SystemParameters{T}, 
    chain_params::ChainParameters{T}, 
    grid::RadialGrid{T}, 
    z_max::T # NEW: We pass the exact dynamic sampling boundary
) where {T}

    N_sites = sys_params.N_sites
    begin_idx = zeros(Int, N_sites, N_sites)
    end_idx = zeros(Int, N_sites, N_sites)
    
    for i in 1:N_sites, j in 1:N_sites
        # Start correction just inside the core
        begin_idx[i, j] = max(1, floor(Int, chain_params.σ[1] / grid.Δr)) 
        # End correction EXACTLY where direct sampling stops
        end_idx[i, j]   = min(grid.N, ceil(Int, z_max / grid.Δr))
    end
    
    sum_rules =[
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
    
    # We build the WEIGHTED projection matrix A'
    q_matrix_weighted = zeros(T, length(sum_orders), q_len)
    
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
                
                # --- NEW: Spatial Weighting (1 / r^4) ---
                # This guarantees the pseudoinverse will decay as 1/r^4
                weight_inv = T(1.0) / (r_val^4)
                
                q_matrix_weighted[g, col_offset + k] = prefactor * (r_val^(n + 2)) * sum_rules[g][i, j] * weight_inv
            end
            col_offset += len
        end
    end
    
    println("Precomputing Spatially-Weighted Pseudoinverse for Corrector...")
    pinv_q = pinv(q_matrix_weighted)
    
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
    
    # This solves for the transformed variable x
    x_vec = corrector.pinv_q * RHS
    
    h_fixed .= h_sim
    
    col_offset = 0
    for i in 1:N_sites, j in 1:N_sites
        len = corrector.end_idx[i, j] - corrector.begin_idx[i, j] + 1
        for k in 1:len
            r_idx = corrector.begin_idx[i, j] + k - 1
            r_val = grid.r[r_idx]
            
            # --- NEW: Transform x back to q using the spatial decay ---
            # q_i = x_i / r^4. This ensures the correction smoothly hits 0.0!
            q_val = x_vec[col_offset + k] / (r_val^4)
            
            h_fixed[i, j, r_idx] += q_val
        end
        col_offset += len
    end
end