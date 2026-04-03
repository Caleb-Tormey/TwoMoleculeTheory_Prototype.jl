# scripts/run_simulation.jl

include("../src/TwoMoleculeTheory.jl")
using StaticArrays
using Printf
using Dates # NEW: For timestamping our runs

function save_to_csv(filename::String, grid_vals::Vector{Float64}, data::Array{Float64, 3})
    open(filename, "w") do io
        println(io, "x, 11, 12, 21, 22")
        for i in 1:length(grid_vals)
            @printf(io, "%.6f, %.6e, %.6e, %.6e, %.6e\n", 
                    grid_vals[i], data[1,1,i], data[1,2,i], data[2,1,i], data[2,2,i])
        end
    end
    println("  -> Saved: $filename")
end

function save_convergence_history(filename::String, W_err_list, C_err_hist, dC_hist)
    open(filename, "w") do io
        println(io, "Outer_Iter,Inner_Iter,deltaC_Step,C_k_Error,W_r_Error")
        for out_it in 1:length(C_err_hist)
            num_inner = length(C_err_hist[out_it])
            for in_it in 1:num_inner
                dc_val = dC_hist[out_it][in_it]
                c_val = C_err_hist[out_it][in_it]
                
                if in_it == num_inner && out_it <= length(W_err_list)
                    w_val = W_err_list[out_it]
                    @printf(io, "%d,%d,%.6e,%.6e,%.6e\n", out_it, in_it, dc_val, c_val, w_val)
                else
                    @printf(io, "%d,%d,%.6e,%.6e,NaN\n", out_it, in_it, dc_val, c_val)
                end
            end
        end
    end
    println("  -> Saved Convergence History to: $filename")
end

function main()
    sys = SystemParameters(405.0, 0.001985875, 0.03123, 2, 24)
    ch_params = ChainParameters(
        1.54, 124.18, 114.0 * π / 180.0, 
        SVector(2.007, 4.012, 0.271, -6.290), 
        SVector(3.93, 3.93), SVector(0.07398, 0.07398), 
        1.1225 * 3.93, 0.0,[i % 2 == 1 ? 1 : 2 for i in 1:24] 
    )

    grid = RadialGrid(2048, 0.1)

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    out_dir = joinpath("output", "run_$timestamp")
    println("\n[!] All output for this simulation will be saved to: $out_dir")
    mkpath(out_dir) 
    
    results = solve_two_molecule_theory!(
        sys, ch_params, grid, 
        max_outer = 1,       
        max_inner = 2,      
        mix_inner = 0.05,    
        mix_outer = 0.25,
        use_mdiis_inner = true,   
        burn_in_inner   = 2,
        use_mdiis_outer = false,  
        burn_in_outer   = 2,      
        
        # --- NEW: Complete Monte Carlo Resolution Control ---
        n_configs         = 5000, # e.g., 5000 for production, 500 for testing
        save_step         = 400,  # steps to decorrelate chains
        sweep_mult_burnin = 1,    # Fast sampling during burn-in
        sweep_mult_prod   = 4,    # Deep sampling during production
        # --------------------------------------------------
        
        out_dir = out_dir,
        resume = false
    )
    
    C_k, W_solv, h_fixed, configs, W_err_list, C_err_history, δC_history = results
    
    println("\n==================================================")
    println("   FINAL EXPORT & SUMMARY")
    println("==================================================")
    
    export_xyz(joinpath(out_dir, "test_chains_final.xyz"), configs[1:10])
    save_convergence_history(joinpath(out_dir, "convergence_history.csv"), W_err_list, C_err_history, δC_history)
end

Base.invokelatest(main)