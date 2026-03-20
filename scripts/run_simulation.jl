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
    println("Saved data to $filename")
end

function main()
    sys = SystemParameters(
        405.0, 
        0.001985875, 
        0.03123, 
        2, 
        24
    )

    # Note the site_types mapping added at the end!
    ch_params = ChainParameters(
        1.54, 
        124.18, 
        114.0 * π / 180.0, 
        SVector(2.007, 4.012, 0.271, -6.290), 
        SVector(3.93, 3.93), 
        SVector(0.07398, 0.07398), 
        1.1225 * 3.93, 
        0.0,[i % 2 == 1 ? 1 : 2 for i in 1:24] # Alternating 1 and 2
    )

    grid = RadialGrid(2048, 0.1)

    # --- FIXED KEYWORD ARGUMENTS ---
    # We now explicitly tell it how many outer vs inner loops to run
    C_k, W_solv, h_fixed, configs = solve_two_molecule_theory!(
        sys, ch_params, grid, 
        max_outer=10,       # Generate chains 2 times
        max_inner=10,      # Converge C(r) 10 times per chain generation
        mix_param=0.05
    )
    
    # --- DIAGNOSTICS & SAVING ---
    println("\n==================================================")
    println("   EXPORTING RESULTS")
    println("==================================================")
    
    # 1. Save 10 random configs to an XYZ file to view in VMD/Ovito
    export_xyz("test_chains.xyz", configs[1:10])
    
    # 2. Save math results to CSV
    save_to_csv("C_k_output.csv", grid.k, C_k)
    save_to_csv("W_solv_output.csv", grid.r, W_solv)
    save_to_csv("h_r_fixed_output.csv", grid.r, h_fixed)
end

# Execute the function (safely handles Julia 1.12 world-age semantics)
Base.invokelatest(main)