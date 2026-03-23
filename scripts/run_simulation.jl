# scripts/run_simulation.jl

include("../src/TwoMoleculeTheory.jl")
using StaticArrays
using Printf

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

# --- NEW: Function to save the convergence tracking data ---
function save_convergence_history(filename::String, W_err_list, C_err_hist, dC_hist)
    open(filename, "w") do io
        # CSV Header
        println(io, "Outer_Iter,Inner_Iter,deltaC_Step,C_k_Error,W_r_Error")
        
        for out_it in 1:length(C_err_hist)
            num_inner = length(C_err_hist[out_it])
            for in_it in 1:num_inner
                dc_val = dC_hist[out_it][in_it]
                c_val = C_err_hist[out_it][in_it]
                
                # We only log the W(r) error at the end of the outer loop cycle
                if in_it == num_inner && out_it <= length(W_err_list)
                    w_val = W_err_list[out_it]
                    @printf(io, "%d,%d,%.6e,%.6e,%.6e\n", out_it, in_it, dc_val, c_val, w_val)
                else
                    # Use NaN so plotting software (Excel/Python) leaves a gap
                    @printf(io, "%d,%d,%.6e,%.6e,NaN\n", out_it, in_it, dc_val, c_val)
                end
            end
        end
    end
    println("  -> Saved Convergence History to: $filename")
end

function main()
    sys = SystemParameters(
        405.0, 
        0.001985875, 
        0.03123, 
        2, 
        24
    )

    ch_params = ChainParameters(
        1.54, 
        124.18, 
        114.0 * π / 180.0, 
        SVector(2.007, 4.012, 0.271, -6.290), 
        SVector(3.93, 3.93), 
        SVector(0.07398, 0.07398), 
        1.1225 * 3.93, 
        0.0,[i % 2 == 1 ? 1 : 2 for i in 1:24] 
    )

    grid = RadialGrid(2048, 0.1)

    results = solve_two_molecule_theory!(
        sys, ch_params, grid, 
        max_outer = 10,       
        max_inner = 15,      
        mix_inner = 0.10,    
        mix_outer = 0.30,
        burn_in_inner = 2,
        burn_in_outer = 2
    )
    
    C_k, W_solv, h_fixed, configs, W_err_list, C_err_history, δC_history = results
    
    println("\n==================================================")
    println("   FINAL EXPORT & SUMMARY")
    println("==================================================")
    
    export_xyz("test_chains_final.xyz", configs[1:10])
    
    # Export the consolidated history!
    save_convergence_history("convergence_history.csv", W_err_list, C_err_history, δC_history)
end

Base.invokelatest(main)