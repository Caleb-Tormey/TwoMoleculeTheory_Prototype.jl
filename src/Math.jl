# src/Math.jl

function RadialGrid(N::Int, Δr::T) where {T<:AbstractFloat}
    Δk = T(π) / (Δr * N)
    r = Vector{T}([i * Δr for i in 1:N])
    k = Vector{T}([i * Δk for i in 1:N])
    return RadialGrid{T}(N, Δr, Δk, r, k)
end

function fst!(f_k::Array{T,3}, f_r::Array{T,3}, grid::RadialGrid{T}) where {T}
    N_sites = size(f_r, 1)
    for i in 1:N_sites, j in 1:N_sites
        temp_r = @view(f_r[i, j, :]) .* grid.r
        # Element-wise division with ./
        f_k[i, j, :] .= (2 * T(π) * grid.Δr) .* FFTW.r2r(temp_r, FFTW.RODFT00) ./ grid.k
    end
end

function ifst!(f_r::Array{T,3}, f_k::Array{T,3}, grid::RadialGrid{T}) where {T}
    N_sites = size(f_k, 1)
    for i in 1:N_sites, j in 1:N_sites
        temp_k = @view(f_k[i, j, :]) .* grid.k
        # Fixed the missing dot: T(1.0) ./ (...)
        scale = T(1.0) ./ (grid.r .* (2 * (grid.N + 1)) * 2 * T(π) * grid.Δr)
        f_r[i, j, :] .= FFTW.r2r(temp_k, FFTW.RODFT00) .* scale
    end
end

function trap_integrate(data::AbstractVector{T}, h::T) where {T}
    N = length(data)
    return (h / 2.0) * (data[1] + data[N] + 2.0 * sum(@view data[2:N-1]))
end

function h_moment(h_data::AbstractVector{T}, n::Int, grid::RadialGrid{T}) where {T}
    temp = h_data .* (grid.r .^ (n + 2))
    sign_val = n == 2 ? T(-1.0) : T(1.0)
    prefactor = sign_val * (T(4.0) * T(π) / (n + 1))
    return prefactor * trap_integrate(temp, grid.Δr)
end

"""
    save_to_csv(filename, grid_vals, data)

Saves a 2x2 correlation or potential matrix to a clean CSV file.
"""
function save_to_csv(filename::String, grid_vals::Vector{T}, data::Array{T, 3}) where {T}
    open(filename, "w") do io
        println(io, "x, 11, 12, 21, 22")
        for i in 1:length(grid_vals)
            @printf(io, "%.6f, %.6e, %.6e, %.6e, %.6e\n", 
                    grid_vals[i], data[1,1,i], data[1,2,i], data[2,1,i], data[2,2,i])
        end
    end
    println("  -> Saved: $filename")
end