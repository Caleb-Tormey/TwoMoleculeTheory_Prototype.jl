# src/Types.jl

# ---------------------------------------------------------
# ABSTRACT INTERFACES
# ---------------------------------------------------------
abstract type AbstractTheory end
abstract type AbstractSampler end
abstract type AbstractGenerator end

# ---------------------------------------------------------
# SYSTEM & GRID TYPES
# ---------------------------------------------------------
struct SystemParameters{T<:AbstractFloat}
    T_sys::T          
    k_B::T            
    ρ::T              
    N_sites::Int      
    N_monomers::Int   
end

struct RadialGrid{T<:AbstractFloat}
    N::Int
    Δr::T
    Δk::T
    r::Vector{T}   # Changed from SVector to standard Vector
    k::Vector{T}   # Changed from SVector to standard Vector
end

# ---------------------------------------------------------
# MONTE CARLO TYPES
# ---------------------------------------------------------
struct ChainParameters{T<:AbstractFloat}
    l_bond::T                 
    k_bend::T                 
    θ_b::T                    
    a_torsion::SVector{4, T}  
    σ::SVector{2, T}          
    ϵ::SVector{2, T}          
    r_cut::T                  
    LJ_shift::T     
    site_types::Vector{Int}   # NEW: Maps atom index to site type (1 or 2)
end

const Monomer{T} = SVector{4, T}
const Molecule{T} = Vector{Monomer{T}}

struct PivotGenerator <: AbstractGenerator
    N_configs::Int
    save_step::Int
end

struct DirectSampler <: AbstractSampler
    MC_steps::Int
end

struct ThreadWorkspace{T<:AbstractFloat}
    mol1_shifted::Molecule{T}
    mol2_shifted::Molecule{T}
    dist_indices::Matrix{Int}   
    g_r_accum::Array{T, 3}      
end

# ---------------------------------------------------------
# MATH / CORRECTOR TYPES
# ---------------------------------------------------------
struct DivergenceCorrector{T<:AbstractFloat}
    pinv_q::Matrix{T}            
    sum_rules::Vector{Matrix{T}} 
    sum_orders::Vector{Int}      
    begin_idx::Matrix{Int}       
    end_idx::Matrix{Int}         
end
# (Append this to the bottom of src/Types.jl)

# ---------------------------------------------------------
# MDIIS SOLVER TYPES
# ---------------------------------------------------------
mutable struct MDIIS_State{T<:AbstractFloat}
    max_m::Int
    curr_m::Int
    head::Int
    x_hist::Vector{Array{T,3}}
    R_hist::Vector{Array{T,3}}
end