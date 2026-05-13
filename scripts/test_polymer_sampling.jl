# scripts/test_polymer_sampling.jl
include("../src/TwoMoleculeTheory.jl")

using StaticArrays
using Printf
using Statistics
using DataFrames
using CSV

function main()
    println("--- INITIALIZING PHASE 1 POLYMER TEST ---")
    sys = SystemParameters(405.0, 0.001985875, 0.03123, 2, 24)
    site_types = [i % 2 == 1 ? 1 : 2 for i in 1:24]
    ch_params = ChainParameters(
        1.54, 124.18, 114.0 * π / 180.0, 
        SVector(2.007, 4.012, 0.271, -6.290), 
        SVector(3.93, 3.93), SVector(0.07398, 0.07398), 
        1.1225 * 3.93, 0.0, site_types 
    )

    grid = RadialGrid(2048, 0.1)
    
    # 0 Solvation Potential (Phase 1 isolated testing)
    W_solv = zeros(Float64, sys.N_sites, sys.N_sites, grid.N)

    println("\nGenerating configuration pool...")
    gen = PivotGenerator(1000, 400)
    configs = generate_configs!(gen, ch_params, sys, W_solv, grid)

    println("\n[1] Running Direct Sampling...")
    h_direct = zeros(Float64, sys.N_sites, sys.N_sites, grid.N)
    # Equivalent to 2.5 million MC total configs
    sample_direct!(h_direct, configs, 10_000, 33, 700, ch_params, sys, W_solv, grid) 

    println("\n[2] Running Window Sampling...")
    # win_width = 3.0, overlap = 1.5, n_steps = 300,000, sim_r_cut = 25.0
    win_sampler = WindowSampler(3.0, 1.5, 300_000, 70.0)
    h_window = zeros(Float64, sys.N_sites, sys.N_sites, grid.N)
    sample_window!(h_window, configs, win_sampler, ch_params, sys, W_solv, grid)

    # Exporting
    idx_max = 2048 # 25.0 Angstroms
    r_vals = grid.r[1:idx_max]
    
    # Reconstructing g(r) = h(r) + 1.0 (Note: Direct sample output is technically manipulated by 144 pairs, so we undo that conceptually if it's strictly h(r))
    g11_dir = h_direct[1, 1, 1:idx_max] .+ 1.0
    g12_dir = h_direct[1, 2, 1:idx_max] .+ 1.0
    g22_dir = h_direct[2, 2, 1:idx_max] .+ 1.0

    g11_win = h_window[1, 1, 1:idx_max] .+ 1.0
    g12_win = h_window[1, 2, 1:idx_max] .+ 1.0
    g22_win = h_window[2, 2, 1:idx_max] .+ 1.0

    df = DataFrame(
        r_Angstrom = r_vals,
        g11_Direct = g11_dir, g11_Window = g11_win,
        g12_Direct = g12_dir, g12_Window = g12_win,
        g22_Direct = g22_dir, g22_Window = g22_win
    )
    
    CSV.write("polymer_sampling_comparison.csv", df)
    println("\nSuccess! Raw comparison data saved to `polymer_sampling_comparison.csv`.")
    println("Plot this data to visually verify the window sampling aligns with the direct sampling structure.")
end

main()