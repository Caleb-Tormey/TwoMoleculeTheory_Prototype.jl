# scripts/run_simulation.jl

include("../src/TwoMoleculeTheory.jl")
using StaticArrays
using Printf
using Dates

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

# --- UPDATED: Beautifully Categorized Parameter Logging ---
function save_run_log(filename::String, sys::SystemParameters, ch::ChainParameters, grid::RadialGrid, settings::NamedTuple)
    open(filename, "w") do io
        println(io, "==================================================")
        println(io, "   TWO-MOLECULE THEORY: SIMULATION RUN LOG")
        println(io, "==================================================")
        println(io, "Timestamp: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        
        println(io, "\n--- SYSTEM & GRID PARAMETERS ---")
        println(io, rpad("Temperature (K)", 30), ": ", sys.T_sys)
        println(io, rpad("Density (ρ)", 30), ": ", sys.ρ)
        println(io, rpad("Number of Sites", 30), ": ", sys.N_sites)
        println(io, rpad("Monomers per Chain", 30), ": ", sys.N_monomers)
        println(io, rpad("Grid Points (N)", 30), ": ", grid.N)
        println(io, rpad("Grid Spacing Δr (Å)", 30), ": ", grid.Δr)
        
        println(io, "\n--- LENNARD-JONES & CHAIN PHYSICS ---")
        println(io, rpad("Bond Length (Å)", 30), ": ", ch.l_bond)
        println(io, rpad("LJ Sigma", 30), ": ", ch.σ[1])
        println(io, rpad("LJ Epsilon", 30), ": ", ch.ϵ[1])
        println(io, rpad("Max LJ Cutoff (Å)", 30), ": ", ch.r_cut)
        println(io, rpad("Use Attractive LJ Ramp?", 30), ": ", settings.use_attractive_lj)
        if settings.use_attractive_lj
            println(io, rpad("  -> Ramp Iterations", 30), ": ", settings.lj_ramp_iters)
        end
        
        println(io, "\n--- MONTE CARLO RESOLUTION ---")
        println(io, rpad("Chains Generated (Ω_k)", 30), ": ", settings.n_configs)
        println(io, rpad("MC Save Step (Decorrelation)", 30), ": ", settings.save_step)
        println(io, rpad("Burn-in g(r) Sweeps Multiplier", 30), ": ", settings.sweep_mult_burnin, "x (", settings.sweep_mult_burnin * settings.n_configs, " sweeps)")
        println(io, rpad("Product g(r) Sweeps Multiplier", 30), ": ", settings.sweep_mult_prod, "x (", settings.sweep_mult_prod * settings.n_configs, " sweeps)")
        println(io, rpad("Transition to High Precision", 30), ": Outer Iteration ", settings.sweep_transition_iter)

        println(io, "\n--- CONVERGENCE & MDIIS LOGIC ---")
        println(io, rpad("Max Outer / Inner Iters", 30), ": ", settings.max_outer, " / ", settings.max_inner)
        println(io, rpad("Outer Solvation Tol (W_r)", 30), ": ", settings.outer_tol)
        println(io, rpad("Inner Correlation Tol (C_k)", 30), ": ", settings.inner_tol)
        
        println(io, "\n  [Inner Loop Solver]")
        println(io, rpad("  Method", 30), ": ", settings.use_mdiis_inner ? "MDIIS (with Picard Burn-in)" : "Pure Picard")
        println(io, rpad("  Inner Burn-in Steps", 30), ": ", settings.burn_in_inner)
        println(io, rpad("  Burn-in Mixing Ratio", 30), ": ", settings.mix_inner_burnin)
        println(io, rpad("  Production Mixing Ratio", 30), ": ", settings.mix_inner_prod)
        
        println(io, "\n[Outer Loop Solver]")
        println(io, rpad("  Method", 30), ": ", settings.use_mdiis_outer ? "MDIIS (with Picard Burn-in)" : "Pure Picard")
        println(io, rpad("  Outer Burn-in Steps", 30), ": ", settings.burn_in_outer)
        println(io, rpad("  Outer Mixing Ratio", 30), ": ", settings.mix_outer)
        
    end
    println("  -> Saved Run Parameters to: $filename")
end

function main()
    sys = SystemParameters(405.0, 0.001985875, 0.03123, 2, 24)

    ch_params = ChainParameters(
        1.54, 124.18, 114.0 * π / 180.0, 
        SVector(2.007, 4.012, 0.271, -6.290), 
        SVector(3.93, 3.93), SVector(0.07398, 0.07398), 
        10.0 * 3.93, 
        0.0,[i % 2 == 1 ? 1 : 2 for i in 1:24] 
    )

    grid = RadialGrid(2048, 0.1)

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    out_dir = joinpath("output", "run_$timestamp")
    println("\n[!] All output for this simulation will be saved to: $out_dir")
    mkpath(out_dir) 

    solver_settings = (
        max_outer = 20,       
        max_inner = 30,      
        
        mix_inner_burnin = 0.15, 
        mix_inner_prod   = 0.05, 
        mix_outer        = 0.20,
        
        use_mdiis_inner = false,   
        burn_in_inner   = 3,
        use_mdiis_outer = false,  
        burn_in_outer   = 100,      
        
        n_configs         = 20000, 
        save_step         = 400,
        sweep_mult_burnin = 1,    
        sweep_mult_prod   = 4,    
        sweep_transition_iter = 10,
        
        use_attractive_lj = false,
        lj_ramp_iters     = 5,
        
        inner_tol = 1e-5,
        outer_tol = 1e-4,
        
        resume = false
    )
    
    # Generate the pristine log file!
    save_run_log(joinpath(out_dir, "parameters_log.txt"), sys, ch_params, grid, solver_settings)

    results = solve_two_molecule_theory!(
        sys, ch_params, grid; 
        out_dir = out_dir, 
        solver_settings...
    )
    
    C_k, W_solv, h_fixed, configs, W_err_list, C_err_history, δC_history = results
    
    println("\n==================================================")
    println("   FINAL EXPORT & SUMMARY")
    println("==================================================")
    
    export_xyz(joinpath(out_dir, "test_chains_final.xyz"), configs[1:10])
    save_convergence_history(joinpath(out_dir, "convergence_history.csv"), W_err_list, C_err_history, δC_history)
end

Base.invokelatest(main)