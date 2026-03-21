# src/Solver.jl

function compute_omega!(
    Ω_k::Array{T,3}, configs::Vector{Molecule{T}}, grid::RadialGrid{T}, 
    sys_params::SystemParameters{T}, chain_params::ChainParameters{T} # Added chain_params!
) where {T}
    N_sites = sys_params.N_sites
    N_monomers = sys_params.N_monomers
    Ω_r_accum = zeros(T, N_sites, N_sites, grid.N)
    
    for mol in configs
        for i in 1:N_monomers, j in 1:N_monomers
            if i != j
                dist = norm(mol[i] - mol[j])
                idx = round(Int, dist / grid.Δr)
                if 1 <= idx <= grid.N
                    # BUG FIXED: Actually map the correct site types!
                    s1 = chain_params.site_types[i]
                    s2 = chain_params.site_types[j]
                    Ω_r_accum[s1, s2, idx] += 1.0
                end
            end
        end
    end
    
    norm_factor = T(3.0) / (T(4.0) * T(π) * (grid.Δr^3) * T(12.0))
    Ω_r = zeros(T, N_sites, N_sites, grid.N)
    for idx in 1:grid.N
        shell_vol = T(3.0) * idx^2 + T(3.0) * idx + T(1.0)
        for i in 1:N_sites, j in 1:N_sites
            Ω_r[i, j, idx] = (norm_factor * Ω_r_accum[i, j, idx]) / (length(configs) * shell_vol)
        end
    end
    
    fst!(Ω_k, Ω_r, grid)
    for idx in 1:grid.N, i in 1:N_sites
        Ω_k[i, i, idx] += T(1.0)
    end
end

function solve_prism_kspace!(
    Δ_k::Array{T,3}, W_solv::Array{T,3}, C_k::Array{T,3}, Ω_k::Array{T,3}, 
    grid::RadialGrid{T}, sys_params::SystemParameters{T}
) where {T}
    N_sites = sys_params.N_sites
    I_mat = Matrix{T}(I, N_sites, N_sites)
    
    ρ_mat = I_mat .* (sys_params.ρ / N_sites)
    
    W_k = zeros(T, N_sites, N_sites, grid.N)
    
    for i in 1:grid.N
        C_mat = C_k[:, :, i]
        Ω_mat = Ω_k[:, :, i]
        
        inv_term = inv(I_mat - C_mat * ρ_mat * Ω_mat)
        Δ_mat = -1.0 * inv_term * C_mat
        Δ_k[:, :, i] .= Δ_mat
        
        kT = sys_params.k_B * sys_params.T_sys
        W_k[:, :, i] .= kT .* (C_mat + Δ_mat)
    end
    ifst!(W_solv, W_k, grid)
end

function solve_two_molecule_theory!(
    sys_params::SystemParameters{T}, chain_params::ChainParameters{T}, grid::RadialGrid{T};
    max_outer::Int = 3, max_inner::Int = 20, mix_inner::T = T(0.05), mix_outer::T = T(0.25)
) where {T}
    println("\n==================================================")
    println("   INITIALIZING TWO-MOLECULE THEORY SOLVER")
    println("==================================================")
    
    N_sites = sys_params.N_sites
    W_solv     = zeros(T, N_sites, N_sites, grid.N)
    W_solv_old = zeros(T, N_sites, N_sites, grid.N) # NEW: To track W(r) for mixing
    C_k        = zeros(T, N_sites, N_sites, grid.N)
    Ω_k        = zeros(T, N_sites, N_sites, grid.N)
    Δ_PRISM    = zeros(T, N_sites, N_sites, grid.N)
    Δ_Two      = zeros(T, N_sites, N_sites, grid.N)
    h_sim      = zeros(T, N_sites, N_sites, grid.N)
    h_fixed    = zeros(T, N_sites, N_sites, grid.N)
    H_k        = zeros(T, N_sites, N_sites, grid.N)
    
    gen = PivotGenerator(2500, 400) 
    corrector = DivergenceCorrector(sys_params, chain_params, grid)
    
    start_n = 33
    stop_n  = 600
    
    local configs
    
    for outer_iter in 1:max_outer
        @printf("\n==================================================\n")
        @printf(">>> OUTER ITERATION %d <<<\n", outer_iter)
        @printf("==================================================\n")
        
        # --- NEW: Save the solvation potential before inner loop overwrites it ---
        W_solv_old .= W_solv 
        
        println("Generating Single Chains in current Solvation Field...")
        configs = generate_configs!(gen, chain_params, sys_params, W_solv, grid)
        
        compute_omega!(Ω_k, configs, grid, sys_params, chain_params) 
        
        MC_sweeps = 50 * length(configs) 
        
        for inner_iter in 1:max_inner
            @printf("\n  --- Inner Iteration %d ---\n", inner_iter)
            
            # This overwrites W_solv with the new calculated W_calc(r)
            solve_prism_kspace!(Δ_PRISM, W_solv, C_k, Ω_k, grid, sys_params)
            
            h_sim .= 0.0
            sample_direct!(h_sim, configs, MC_sweeps, start_n, stop_n, chain_params, sys_params, W_solv, grid)
            
            # --- TRUE TAIL SPLICING ---
            H_PRISM_k = zeros(T, N_sites, N_sites, grid.N)
            for i in 1:grid.N
                Ω_mat = Ω_k[:, :, i]
                H_PRISM_k[:, :, i] .= -1.0 .* (Ω_mat * Δ_PRISM[:, :, i] * Ω_mat)
            end
            h_PRISM_r = zeros(T, N_sites, N_sites, grid.N)
            ifst!(h_PRISM_r, H_PRISM_k, grid)
            
            splice_n = round(Int, 10.0 * chain_params.σ[1] / grid.Δr)
            
            for i in 1:N_sites, j in 1:N_sites
                h_sim[i, j, splice_n+1:end] .= h_PRISM_r[i, j, splice_n+1:end]
            end
            
            correct_h!(h_fixed, h_sim, corrector, grid)
            fst!(H_k, h_fixed, grid)
            
            for i in 1:grid.N
                Ω_mat = Ω_k[:, :, i]
                H_mat = H_k[:, :, i]
                Ω_inv = inv(Ω_mat)
                Δ_Two[:, :, i] .= -1.0 .* (Ω_inv * H_mat * Ω_inv)
            end
            
            δC = Δ_PRISM .- Δ_Two
            err = sqrt(sum(δC.^2) / length(δC))
            @printf("  Convergence Error ||δC|| : %.6e\n", err)
            
            max_step = T(1.0)
            if err > max_step || isnan(err)
                println("    -> WARNING: Large step detected! Clamping δC.")
                δC .*= (max_step / err)
            end
            
            # Update C(k)
            C_k .+= mix_inner .* δC
            
            if err < 1e-5
                println("\n  *** INNER LOOP CONVERGED! ***")
                break
            end
        end
        
        # --- NEW: Outer Loop Mixing for W(r) ---
        # Mix the W(r) we started with (W_solv_old) and the W(r) the inner loop just converged to (W_solv)
        W_solv .= (T(1.0) - mix_outer) .* W_solv_old .+ mix_outer .* W_solv
        println("  -> Mixed Outer Solvation Potential W(r)")
    end
    
    return C_k, W_solv, h_fixed, configs
end