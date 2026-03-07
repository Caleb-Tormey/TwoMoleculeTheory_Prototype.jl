using LinearAlgebra
using StaticArrays
using Random
using FFTW
using Printf
using Test

# ==============================================================================
# 1. ABSTRACT INTERFACES
# ==============================================================================

abstract type AbstractTheory end
abstract type AbstractSampler end
abstract type AbstractGenerator end

# ==============================================================================
# 2. SYSTEM PARAMETERS & GRID
# ==============================================================================

"""
    SystemParameters{T<:AbstractFloat}

Holds the thermodynamic and physical parameters for the melt.
"""
struct SystemParameters{T<:AbstractFloat}
    T_sys::T          # Temperature (K)
    k_B::T            # Boltzmann Constant (kcal/mol/K)
    ρ::T              # Monomer Density (Å⁻³)
    N_sites::Int      # Number of site types (e.g., 2 for A and B)
    N_monomers::Int   # Degree of polymerization (e.g., 24)
end

"""
    RadialGrid{T<:AbstractFloat}

Handles the 1D real and Fourier space grids for correlation functions.
"""
struct RadialGrid{T<:AbstractFloat}
    N::Int
    Δr::T
    Δk::T
    r::SVector{<:Any, T}
    k::SVector{<:Any, T}
end

function RadialGrid(N::Int, Δr::T) where {T<:AbstractFloat}
    Δk = T(π) / (Δr * N)
    r = SVector{N, T}([(i) * Δr for i in 1:N])
    k = SVector{N, T}([(i) * Δk for i in 1:N])
    return RadialGrid{T}(N, Δr, Δk, r, k)
end

# ==============================================================================
# 3. TRANSFORMS & RISM MATH
# ==============================================================================

"""
    fst!(f_k, f_r, grid)

Performs an in-place Forward Discrete Sine Transform (DST-I).
Converts real-space function `f_r` to Fourier-space `f_k`.
"""
function fst!(f_k::Array{T,3}, f_r::Array{T,3}, grid::RadialGrid{T}) where {T}
    N_types = size(f_r, 1)
    for i in 1:N_types, j in 1:N_types
        temp_r = @view(f_r[i, j, :]) .* grid.r
        # FFTW RODFT00 is DST-I
        f_k[i, j, :] .= (2 * T(π) * grid.Δr) .* FFTW.r2r(temp_r, FFTW.RODFT00) ./ grid.k
    end
end

"""
    ifst!(f_r, f_k, grid)

Performs an in-place Inverse Discrete Sine Transform.
"""
function ifst!(f_r::Array{T,3}, f_k::Array{T,3}, grid::RadialGrid{T}) where {T}
    N_types = size(f_k, 1)
    for i in 1:N_types, j in 1:N_types
        temp_k = @view(f_k[i, j, :]) .* grid.k
        scale = T(1.0) / (grid.r .* (2 * (grid.N + 1)) * 2 * T(π) * grid.Δr)
        f_r[i, j, :] .= FFTW.r2r(temp_k, FFTW.RODFT00) .* scale
    end
end

# ==============================================================================
# 4. THE DIVERGENCE CORRECTOR
# ==============================================================================

"""
    DivergenceCorrector{T}

Handles the pseudoinverse correction of the raw simulation data `h_raw(r)`
to enforce the k->0 sum rules (Eqs. A.8, A.11 in 2026 PDF).
"""
struct DivergenceCorrector{T<:AbstractFloat}
    q_matrix::Matrix{T}
    sum_rules::Vector{Matrix{T}}
    sum_orders::Vector{Int}
end

function DivergenceCorrector(grid::RadialGrid{T}, start_idx::Int, end_idx::Int) where {T}
    # For a 2-site symmetric/hetero molecule
    sum_rules = [[1.0 -1.0; -1.0 1.0], 
        [1.0  0.0; -1.0 0.0],[1.0 -1.0;  0.0 0.0],[1.0  0.0;  0.0 -1.0]
    ]
    sum_orders =[2, 0, 0, 0]
    
    # Construct the A matrix (q_matrix) for the pseudoinverse
    # This evaluates the moments r^(n+2) over the correction window
    # To keep it prototype-sized, we define the skeleton here.
    q_len = (end_idx - start_idx + 1) * 4 # 4 matrix elements
    q_matrix = zeros(T, length(sum_orders), q_len)
    
    # Logic to fill q_matrix based on trapezoidal integration of r^(n+2)
    # ... (Implementation of Eq A.26) ...
    
    return DivergenceCorrector{T}(q_matrix, sum_rules, sum_orders)
end

"""
    correct_h!(h_fixed, h_raw, corrector, grid)

Applies the minimal correcting vector to `h_raw` via Moore-Penrose pseudoinverse (Eq. A.28).
"""
function correct_h!(h_fixed::Array{T,3}, h_raw::Array{T,3}, corrector::DivergenceCorrector{T}, grid::RadialGrid{T}) where {T}
    # 1. Compute violation RHS
    # 2. Solve q = pinv(A) * RHS
    # 3. Apply q to the window in h_fixed
    h_fixed .= h_raw # Placeholder: In reality, we add q to the specific window
    return nothing
end

# ==============================================================================
# 5. GENERATOR & SAMPLER
# ==============================================================================

struct PivotGenerator <: AbstractGenerator
    N_configs::Int
    save_step::Int
end

struct DirectSampler <: AbstractSampler
    MC_steps::Int
end

"""
    sample_two_chains!(h_sim, sampler, configs, W_solv, params)

Performs the Direct Sampling Monte Carlo (Eq. II.11) using pre-generated configs.
Multi-threaded, lock-free accumulation into `h_sim`.
"""
function sample_two_chains!(h_sim::Array{T,3}, sampler::DirectSampler, configs::Vector{Vector{SVector{4, T}}}, W_solv::Array{T,3}, params::SystemParameters{T}, grid::RadialGrid{T}) where {T}
    # Thread-local workspaces to avoid race conditions and allocations
    h_threads = [zeros(T, size(h_sim)) for _ in 1:Threads.nthreads()]
    
    Threads.@threads for step in 1:sampler.MC_steps
        t_id = Threads.threadid()
        rng = TaskLocalRNG() # Fast, thread-safe RNG in modern Julia
        
        # 1. Pick two random conformations
        idx1 = rand(rng, 1:length(configs))
        idx2 = rand(rng, 1:length(configs))
        mol1 = configs[idx1]
        mol2 = configs[idx2]
        
        # 2. Random 3D Rotation & Translation along z-axis
        # ... (Rotation/Translation logic goes here) ...
        
        # 3. Compute Energy U^{FE} = U_{LJ} + W(r)
        # 4. Accumulate exp(-U / kBT) into h_threads[t_id]
    end
    
    # Reduce threads
    h_sim .= 0.0
    for t_id in 1:Threads.nthreads()
        h_sim .+= h_threads[t_id]
    end
    
    # Normalize
    # ... (Normalization logic) ...
end

# ==============================================================================
# 6. TWO-MOLECULE THEORY SOLVER
# ==============================================================================

struct TwoMoleculeTheory <: AbstractTheory
    params::SystemParameters
    grid::RadialGrid
    corrector::DivergenceCorrector
end

"""
    solve_inner_loop!(theory, C_k, Ω_k, h_sim)

Executes the inner loop updating C(k) via Eq. II.14.
"""
function solve_inner_loop!(theory::TwoMoleculeTheory, C_k::Array{T,3}, Ω_k::Array{T,3}, h_sim::Array{T,3}) where {T}
    grid = theory.grid
    N = grid.N
    N_sites = theory.params.N_sites
    ρ = theory.params.ρ
    
    Δ_Two = zeros(T, N_sites, N_sites, N)
    Δ_PRISM = zeros(T, N_sites, N_sites, N)
    
    # 1. Correct the simulation h(r)
    h_fixed = zeros(T, size(h_sim))
    correct_h!(h_fixed, h_sim, theory.corrector, grid)
    
    # 2. Transform to Fourier space
    h_k = zeros(T, size(h_sim))
    fst!(h_k, h_fixed, grid)
    
    # 3. Compute Δ_Two(k) = -Ω⁻¹(k) * H(k) * Ω⁻¹(k)
    for i in 1:N
        Ω_mat = SMatrix{N_sites, N_sites, T}(Ω_k[:, :, i])
        H_mat = SMatrix{N_sites, N_sites, T}(h_k[:, :, i])
        Ω_inv = inv(Ω_mat)
        Δ_Two[:, :, i] .= -1.0 .* (Ω_inv * H_mat * Ω_inv)
    end
    
    # 4. Picard Mix C_k
    mix_param = T(0.05)
    for i in 1:N
        # Δ_PRISM(k) = -C(k) * [I - ρ Ω(k) C(k)]⁻¹  (Eq. II.12)
        Ω_mat = SMatrix{N_sites, N_sites, T}(Ω_k[:, :, i])
        C_mat = SMatrix{N_sites, N_sites, T}(C_k[:, :, i])
        I_mat = SMatrix{N_sites, N_sites, T}(I)
        
        inv_term = inv(I_mat - ρ .* Ω_mat * C_mat)
        Δ_PRISM[:, :, i] .= -1.0 .* C_mat * inv_term
        
        # Update rule: Eq II.14 & II.15
        δC = Δ_PRISM[:, :, i] .- Δ_Two[:, :, i]
        C_k[:, :, i] .+= mix_param .* δC
    end
end

# ==============================================================================
# 7. REGRESSION TESTING & VERIFICATION
# ==============================================================================

@testset "TwoMolecule Architecture Tests" begin
    @testset "Grid Setup" begin
        grid = RadialGrid(2048, 0.1)
        @test grid.N == 2048
        @test isapprox(grid.Δk, π / 204.8)
    end

    @testset "Fourier Transforms" begin
        grid = RadialGrid(1024, 0.1)
        # Create a test Gaussian
        f_r = zeros(Float64, 2, 2, 1024)
        for i in 1:1024
            r = grid.r[i]
            val = exp(-r^2 / 2.0)
            f_r[1, 1, i] = val
        end
        
        f_k = similar(f_r)
        f_r_recovered = similar(f_r)
        
        fst!(f_k, f_r, grid)
        ifst!(f_r_recovered, f_k, grid)
        
        # Test if inverse transform recovers original function
        @test isapprox(f_r[1, 1, 10], f_r_recovered[1, 1, 10], atol=1e-5)
    end
end