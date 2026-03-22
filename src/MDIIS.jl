# src/MDIIS.jl

"""
    MDIIS_State(max_m, dims, T)

Initializes the MDIIS history buffers with zeros to avoid allocations during the loop.
"""
function MDIIS_State(max_m::Int, dims::NTuple{3,Int}, ::Type{T}) where {T}
    x_hist = [zeros(T, dims) for _ in 1:max_m]
    R_hist = [zeros(T, dims) for _ in 1:max_m]
    return MDIIS_State{T}(max_m, 0, 0, x_hist, R_hist)
end

"""
    reset!(state::MDIIS_State)

Wipes the MDIIS history. Crucial when the outer loop updates W(r), as the 
inner loop's previous C(k) history is no longer physically valid.
"""
function reset!(state::MDIIS_State)
    state.curr_m = 0
    state.head = 0
end

"""
    update_MDIIS!(x, R, state, α)

Performs the Modified Direct Inversion in the Iterative Subspace (MDIIS).
Overwrites `x` in-place with the optimal next guess.
"""
function update_MDIIS!(x::Array{T,3}, R::Array{T,3}, state::MDIIS_State{T}, α::T) where {T}
    # 1. Advance the circular buffer
    state.head = mod1(state.head + 1, state.max_m)
    if state.curr_m < state.max_m
        state.curr_m += 1
    end
    
    # 2. Store current guess and residual
    state.x_hist[state.head] .= x
    state.R_hist[state.head] .= R
    
    m = state.curr_m
    
    # 3. If history is just 1, fall back to standard Picard mixing
    if m == 1
        x .+= α .* R
        return
    end
    
    # 4. Build the B matrix (dot products of residuals)
    B = zeros(T, m + 1, m + 1)
    for i in 1:m
        idx_i = mod1(state.head - i + 1, state.max_m)
        for j in i:m
            idx_j = mod1(state.head - j + 1, state.max_m)
            
            # Euclidean dot product of the 3D arrays
            val = dot(state.R_hist[idx_i], state.R_hist[idx_j])
            
            B[i, j] = val
            B[j, i] = val
        end
    end
    
    # 5. Tikhonov Regularization (The magic that prevents singular crashes!)
    B_max = maximum(abs.(B[1:m, 1:m]))
    for i in 1:m
        B[i, i] += max(B_max * T(1e-6), T(1e-12))
    end
    
    # 6. Apply the constraint row and column (Sum of c_i = 1)
    for i in 1:m
        B[i, m + 1] = T(-1.0)
        B[m + 1, i] = T(-1.0)
    end
    B[m + 1, m + 1] = T(0.0)
    
    # 7. Build RHS vector
    rhs = zeros(T, m + 1)
    rhs[m + 1] = T(-1.0)
    
    # 8. Solve for the optimal coefficients `c`
    c = B \ rhs
    
    # 9. Extrapolate the new optimal guess `x`
    x .= T(0.0)
    for i in 1:m
        idx_i = mod1(state.head - i + 1, state.max_m)
        # x_new = Sum[ c_i * (x_i + α * R_i) ]
        x .+= c[i] .* (state.x_hist[idx_i] .+ α .* state.R_hist[idx_i])
    end
end