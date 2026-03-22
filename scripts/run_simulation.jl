# scripts/run_simulation.jl

include("../src/TwoMoleculeTheory.jl")
using StaticArrays
using Printf

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

    # --- THOROUGH TEST ---
    results = solve_two_molecule_theory!(
        sys, ch_params, grid, 
        max_outer = 4,       
        max_inner = 10,      
        mix_inner = 0.1,    
        mix_outer = 0.30     
    )
    
    C_k, W_solv, h_fixed, configs, W_err_list, C_err_history, δC_history = results
    
    println("\n==================================================")
    println("   FINAL EXPORT & SUMMARY")
    println("==================================================")
    
    export_xyz("test_chains_final.xyz", configs[1:10])
    
    println("\n--- W(r) Outer Convergence History ---")
    for (i, err) in enumerate(W_err_list)
        @printf("Iteration %02d : ∫(ΔW_r)² dr = %.6e\n", i, err)
    end
end

Base.invokelatest(main)