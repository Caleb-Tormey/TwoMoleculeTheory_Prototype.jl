# src/Solver.jl

function compute_omega!(
    Ω_k::Array{T,3}, configs::Vector{Molecule{T}}, grid::RadialGrid{T}, 
    sys_params::SystemParameters{T}, chain_params::ChainParameters{T} 
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
    max_outer::Int = 10, max_inner::Int = 20, mix_inner::T = T(0.05), mix_outer::T = T(0.25),
    burn_in_outer::Int = 2, burn_in_inner::Int = 2,
    initial_W::Union{Array{T,3}, Nothing} = nothing,
    out_dir::String = "output" # NEW: Directory output parameter
) where {T}
    println("\n==================================================")
    println("   INITIALIZING TWO-MOLECULE THEORY SOLVER")
    println("==================================================")
    
    # --- NEW: Safely create the subfolders ---
    mkpath(joinpath(out_dir, "Wr"))
    mkpath(joinpath(out_dir, "Ck"))
    mkpath(joinpath(out_dir, "hr_fixed"))
    
    N_sites = sys_params.N_sites
    W_solv = zeros(T, N_sites, N_sites, grid.N)
    
    if initial_W !== nothing
        println("  -> Loading initial W(r) from checkpoint...")
        W_solv .= initial_W
    end
    
    W_solv_old = zeros(T, N_sites, N_sites, grid.N)
    C_k        = zeros(T, N_sites, N_sites, grid.N)
    Ω_k        = zeros(T, N_sites, N_sites, grid.N)
    Δ_PRISM    = zeros(T, N_sites, N_sites, grid.N)
    Δ_Two      = zeros(T, N_sites, N_sites, grid.N)
    h_sim      = zeros(T, N_sites, N_sites, grid.N)
    h_fixed    = zeros(T, N_sites, N_sites, grid.N)
    H_k        = zeros(T, N_sites, N_sites, grid.N)
    
    gen = PivotGenerator(2500, 400) 
    corrector = DivergenceCorrector(sys_params, chain_params, grid)
    
    dims = (N_sites, N_sites, grid.N)
    inner_mdiis = MDIIS_State(5, dims, T)
    outer_mdiis = MDIIS_State(5, dims, T) 
    
    start_n = 33
    stop_n  = 600
    
    W_err_list = T[]
    C_err_history = Vector{T}[]
    δC_step_history = Vector{T}[]
    
    local configs
    
    for outer_iter in 1:max_outer
        @printf("\n==================================================\n")
        @printf(">>> OUTER ITERATION %d <<<\n", outer_iter)
        @printf("==================================================\n")
        
        W_solv_old .= W_solv 
        reset!(inner_mdiis)
        
        println("Generating Single Chains in current Solvation Field...")
        configs = generate_configs!(gen, chain_params, sys_params, W_solv, grid)
        
        compute_omega!(Ω_k, configs, grid, sys_params, chain_params) 
        
        MC_sweeps = 50 * length(configs) 
        
        inner_C_errs = T[]
        inner_δC_steps = T[]
        last_inner_err = T(Inf) 
        
        for inner_iter in 1:max_inner
            @printf("\n  --- Inner Iteration %d ---\n", inner_iter)
            
            C_k_old = copy(C_k) 
            solve_prism_kspace!(Δ_PRISM, W_solv, C_k, Ω_k, grid, sys_params)
            
            h_sim .= 0.0
            sample_direct!(h_sim, configs, MC_sweeps, start_n, stop_n, chain_params, sys_params, W_solv, grid)
            
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
            err_step = sqrt(sum(δC.^2) / length(δC))
            push!(inner_δC_steps, err_step)
            
            if err_step > last_inner_err && inner_iter > burn_in_inner + 1
                println("    -> WARNING: Error increased! Flushing Inner MDIIS history.")
                reset!(inner_mdiis)
            end
            last_inner_err = err_step
            
            max_step = T(1.0)
            if err_step > max_step || isnan(err_step)
                println("    -> WARNING: Large step detected! Clamping δC.")
                δC .*= (max_step / err_step)
            end
            
            if inner_iter <= burn_in_inner
                C_k .+= mix_inner .* δC
                println("    -> Picard Mixed C(k) (Burn-in)")
            else
                update_MDIIS!(C_k, δC, inner_mdiis, mix_inner)
            end
            
            C_err = T(0.0)
            for i in 1:N_sites, j in 1:N_sites
                diff_sq = (C_k[i, j, :] .- C_k_old[i, j, :]).^2
                C_err += trap_integrate(diff_sq, grid.Δk)
            end
            C_err /= (N_sites * N_sites) 
            push!(inner_C_errs, C_err)
            
            @printf("  Convergence ||δC|| Step: %.6e | ∫(ΔC_k)² dk: %.6e\n", err_step, C_err)
            
            if err_step < 1e-5
                println("\n  *** INNER LOOP CONVERGED! ***")
                break
            end
        end
        
        push!(C_err_history, inner_C_errs)
        push!(δC_step_history, inner_δC_steps)
        
        δW = W_solv .- W_solv_old
        
        W_err = T(0.0)
        for i in 1:N_sites, j in 1:N_sites
            diff_sq = (δW[i, j, :]).^2
            W_err += trap_integrate(diff_sq, grid.Δr)
        end
        W_err /= (N_sites * N_sites)
        push!(W_err_list, W_err)
        
        @printf("\n  Outer Solvation Error ∫(ΔW_r)² dr : %.6e\n", W_err)
        
        mdiis_threshold = T(1.0) 
        if outer_iter <= burn_in_outer || W_err > mdiis_threshold
            W_solv .= W_solv_old .+ mix_outer .* δW
            if outer_iter > burn_in_outer
                println("  -> Picard Mixed W(r) (Error above threshold, holding off MDIIS)")
            else
                println("  -> Picard Mixed W(r) (Burn-in)")
            end
            reset!(outer_mdiis)
        else
            W_solv .= W_solv_old
            update_MDIIS!(W_solv, δW, outer_mdiis, mix_outer)
            println("  -> MDIIS Updated Outer Solvation Potential W(r)!")
        end
        
        # --- NEW: Save directly into the subfolders! ---
        save_to_csv(joinpath(out_dir, "Wr", @sprintf("W_solv_outer_%02d.csv", outer_iter)), grid.r, W_solv)
        save_to_csv(joinpath(out_dir, "Ck", @sprintf("C_k_outer_%02d.csv", outer_iter)), grid.k, C_k)
        save_to_csv(joinpath(out_dir, "hr_fixed", @sprintf("h_r_fixed_outer_%02d.csv", outer_iter)), grid.r, h_fixed)
    end
    
    return C_k, W_solv, h_fixed, configs, W_err_list, C_err_history, δC_step_history
end