# src/MonteCarlo.jl

function init_chain!(molecule::Molecule{T}, params::ChainParameters{T}) where {T}
    x_step = sin(params.θ_b / 2) * params.l_bond
    y_step = cos(params.θ_b / 2) * params.l_bond
    for i in 1:length(molecule)
        x = (i - 1) * x_step
        y = ((i - 1) % 2) * y_step
        molecule[i] = SVector{4, T}(x, y, 0.0, 1.0)
    end
end

function calc_internal_energy(
    molecule::Molecule{T}, 
    chain_params::ChainParameters{T}, 
    sys_params::SystemParameters{T},
    W_solv::Array{T,3}, 
    grid::RadialGrid{T}
) where {T}
    E_total = T(0.0)
    N = length(molecule)
    
    for i in 1:(N - 2)
        v1 = molecule[i] - molecule[i+1]
        v2 = molecule[i+2] - molecule[i+1]
        dot_val = v1[1]*v2[1] + v1[2]*v2[2] + v1[3]*v2[3]
        norm_v1 = sqrt(v1[1]^2 + v1[2]^2 + v1[3]^2)
        norm_v2 = sqrt(v2[1]^2 + v2[2]^2 + v2[3]^2)
        cos_θ = clamp(dot_val / (norm_v1 * norm_v2), -1.0, 1.0)
        θ = acos(cos_θ)
        E_total += T(0.5) * chain_params.k_bend * (θ - chain_params.θ_b)^2
    end
    
    for i in 1:(N - 3)
        a, b, c, d = molecule[i], molecule[i+1], molecule[i+2], molecule[i+3]
        cb = SVector{3, T}(c[1]-b[1], c[2]-b[2], c[3]-b[3])
        ba = SVector{3, T}(b[1]-a[1], b[2]-a[2], b[3]-a[3])
        dc = SVector{3, T}(d[1]-c[1], d[2]-c[2], d[3]-c[3])
        bc = SVector{3, T}(b[1]-c[1], b[2]-c[2], b[3]-c[3])
        n1 = normalize(cross(cb, ba))
        n2 = normalize(cross(dc, bc))
        cos_ϕ = clamp(dot(n1, n2), -1.0, 1.0)
        a_t = chain_params.a_torsion
        E_total += a_t[1] + cos_ϕ * (a_t[2] + cos_ϕ * (a_t[3] + cos_ϕ * a_t[4]))
    end
    
    exclude = 4 
    r_max = grid.N * grid.Δr
    for i in 1:(N - exclude)
        for j in (i + exclude):N
            dx = molecule[i][1] - molecule[j][1]
            dy = molecule[i][2] - molecule[j][2]
            dz = molecule[i][3] - molecule[j][3]
            dist = sqrt(dx^2 + dy^2 + dz^2)
            
            if dist < chain_params.r_cut
                # Using symmetric LJ for now as requested
                σ_ij, ϵ_ij = chain_params.σ[1], chain_params.ϵ[1] 
                term = (σ_ij / dist)^6
                E_total += T(4.0) * ϵ_ij * (term^2 - term + T(0.25)) - chain_params.LJ_shift
            end
            
            if dist < r_max
                dist_idx = dist / grid.Δr
                idx_low = max(1, floor(Int, dist_idx))
                idx_high = min(grid.N, idx_low + 1)
                fraction = dist_idx - floor(dist_idx)
                
                # Correct Site Type Mapping
                s1 = chain_params.site_types[i]
                s2 = chain_params.site_types[j]
                
                w_low = W_solv[s1, s2, idx_low]
                w_high = W_solv[s1, s2, idx_high]
                E_total += w_low * (T(1.0) - fraction) + w_high * fraction
            end
        end
    end
    return E_total
end

@inline function rodrigues_rotate(v::SVector{3, T}, k::SVector{3, T}, cos_θ::T, sin_θ::T) where {T}
    return v * cos_θ + cross(k, v) * sin_θ + k * dot(k, v) * (1 - cos_θ)
end

function rotate_arm!(molecule::Molecule{T}, pivot_idx::Int, axis::SVector{3, T}, angle::T, start_idx::Int, stop_idx::Int) where {T}
    cos_θ = cos(angle)
    sin_θ = sin(angle)
    pivot_pos = SVector{3, T}(molecule[pivot_idx][1:3])
    for i in start_idx:stop_idx
        pos = SVector{3, T}(molecule[i][1:3])
        v_rot = rodrigues_rotate(pos - pivot_pos, axis, cos_θ, sin_θ)
        molecule[i] = SVector{4, T}(pivot_pos[1] + v_rot[1], pivot_pos[2] + v_rot[2], pivot_pos[3] + v_rot[3], 1.0)
    end
end

function MC_step!(
    molecule::Molecule{T}, temp_molecule::Molecule{T}, current_energy::T, rng::AbstractRNG,
    chain_params::ChainParameters{T}, sys_params::SystemParameters{T}, W_solv::Array{T,3}, grid::RadialGrid{T},
    angle_range::T, dihedral_range::T
) where {T}
    accept_bend, accept_twist = 0, 0
    N = length(molecule)
    β = 1.0 / (sys_params.k_B * sys_params.T_sys)
    
    # BEND
    bend_idx = rand(rng, 2:(N-1))
    Δθ = (rand(rng, T) - T(0.5)) * 2 * angle_range
    v1 = SVector{3, T}(temp_molecule[bend_idx-1][1:3]) - SVector{3, T}(temp_molecule[bend_idx][1:3])
    v2 = SVector{3, T}(temp_molecule[bend_idx+1][1:3]) - SVector{3, T}(temp_molecule[bend_idx][1:3])
    axis_bend = normalize(cross(v1, v2))
    if (bend_idx - 1) <= (N - bend_idx)
        rotate_arm!(temp_molecule, bend_idx, axis_bend, Δθ, 1, bend_idx - 1)
    else
        rotate_arm!(temp_molecule, bend_idx, axis_bend, Δθ, bend_idx + 1, N)
    end
    
    new_energy = calc_internal_energy(temp_molecule, chain_params, sys_params, W_solv, grid)
    ΔE = new_energy - current_energy
    if ΔE <= 0.0 || rand(rng, T) <= exp(-ΔE * β)
        molecule .= temp_molecule; current_energy = new_energy; accept_bend = 1
    else
        temp_molecule .= molecule
    end
    
    # TWIST
    twist_idx = rand(rng, 2:(N-2))
    Δϕ = (rand(rng, T) - T(0.5)) * 2 * dihedral_range
    axis_twist = normalize(SVector{3, T}(temp_molecule[twist_idx+1][1:3]) - SVector{3, T}(temp_molecule[twist_idx][1:3]))
    if (twist_idx - 1) <= (N - twist_idx)
        rotate_arm!(temp_molecule, twist_idx, axis_twist, Δϕ, 1, twist_idx - 1)
    else
        rotate_arm!(temp_molecule, twist_idx, axis_twist, Δϕ, twist_idx + 1, N)
    end
    
    new_energy = calc_internal_energy(temp_molecule, chain_params, sys_params, W_solv, grid)
    ΔE = new_energy - current_energy
    if ΔE <= 0.0 || rand(rng, T) <= exp(-ΔE * β)
        molecule .= temp_molecule; current_energy = new_energy; accept_twist = 1
    else
        temp_molecule .= molecule
    end
    
    return current_energy, accept_bend, accept_twist
end

function generate_configs!(
    generator::PivotGenerator, chain_params::ChainParameters{T}, sys_params::SystemParameters{T}, 
    W_solv::Array{T,3}, grid::RadialGrid{T}
) where {T}
    N = sys_params.N_monomers
    rng = TaskLocalRNG()
    molecule = Vector{Monomer{T}}(undef, N)
    init_chain!(molecule, chain_params)
    temp_molecule = copy(molecule)
    
    current_energy = calc_internal_energy(molecule, chain_params, sys_params, W_solv, grid)
    angle_range = T(20.0 * π / 180.0)
    dihedral_range = T(π / 2.0)
    
    saved_configs = Vector{Molecule{T}}(undef, generator.N_configs)
    
    println("Warming up Pivot Algorithm (10,000 steps)...")
    for _ in 1:10_000 # Warmup
        current_energy, _, _ = MC_step!(molecule, temp_molecule, current_energy, rng, chain_params, sys_params, W_solv, grid, angle_range, dihedral_range)
    end
    
    total_steps = generator.N_configs * generator.save_step
    save_idx = 1
    
    # Track Acceptance
    bends_accepted = 0
    twists_accepted = 0
    
    prog = Progress(total_steps, dt=0.1, desc="Generating Chains: ", showspeed=true)
    
    for step in 1:total_steps
        current_energy, b_acc, t_acc = MC_step!(molecule, temp_molecule, current_energy, rng, chain_params, sys_params, W_solv, grid, angle_range, dihedral_range)
        
        bends_accepted += b_acc
        twists_accepted += t_acc
        
        if step % generator.save_step == 0
            saved_configs[save_idx] = copy(molecule)
            save_idx += 1
        end
        ProgressMeter.next!(prog)
    end
    ProgressMeter.finish!(prog)
    
    bend_ratio = (bends_accepted / total_steps) * 100
    twist_ratio = (twists_accepted / total_steps) * 100
    @printf("  Acceptance Rates -> Bend: %.2f%% | Twist: %.2f%%\n", bend_ratio, twist_ratio)
    
    return saved_configs
end

function random_rotation_matrix(rng::AbstractRNG, ::Type{T}) where {T}
    u1, u2, u3 = rand(rng, T), rand(rng, T), rand(rng, T)
    sq1_u1, sq_u1 = sqrt(T(1.0) - u1), sqrt(u1)
    θ1, θ2 = T(2π) * u2, T(2π) * u3
    w, x, y, z = sq1_u1 * sin(θ1), sq1_u1 * cos(θ1), sq_u1 * sin(θ2), sq_u1 * cos(θ2)
    return SMatrix{3, 3, T}(
        1-2y^2-2z^2, 2x*y+2w*z,   2x*z-2w*y,
        2x*y-2w*z,   1-2x^2-2z^2, 2y*z+2w*x,
        2x*z+2w*y,   2y*z-2w*x,   1-2x^2-2y^2
    )
end

function ThreadWorkspace(N_monomers::Int, N_sites::Int, N_grid::Int, ::Type{T}) where {T}
    return ThreadWorkspace{T}(
        Vector{Monomer{T}}(undef, N_monomers), Vector{Monomer{T}}(undef, N_monomers),
        zeros(Int, N_monomers, N_monomers), zeros(T, N_sites, N_sites, N_grid)
    )
end

function evaluate_two_chain!(
    ws::ThreadWorkspace{T}, mol1::Molecule{T}, mol2::Molecule{T},
    s1_idx::Int, s2_idx::Int, z_shift::T, rng::AbstractRNG,
    chain_params::ChainParameters{T}, W_solv::Array{T,3}, grid::RadialGrid{T}
) where {T}
    N = length(mol1)
    s1_pos = SVector{3, T}(mol1[s1_idx][1:3])
    s2_pos = SVector{3, T}(mol2[s2_idx][1:3])
    rot_mat = random_rotation_matrix(rng, T)
    z_vec = SVector{3, T}(0.0, 0.0, z_shift)
    
    @inbounds for i in 1:N
        p1 = SVector{3, T}(mol1[i][1:3]) - s1_pos
        ws.mol1_shifted[i] = SVector{4, T}(p1[1], p1[2], p1[3], 1.0)
        p2 = SVector{3, T}(mol2[i][1:3]) - s2_pos
        p2_rot = rot_mat * p2 + z_vec
        ws.mol2_shifted[i] = SVector{4, T}(p2_rot[1], p2_rot[2], p2_rot[3], 1.0)
    end
    
    E_inter = T(0.0)
    r_max = grid.N * grid.Δr
    @inbounds for i in 1:N, j in 1:N
        dx = ws.mol1_shifted[i][1] - ws.mol2_shifted[j][1]
        dy = ws.mol1_shifted[i][2] - ws.mol2_shifted[j][2]
        dz = ws.mol1_shifted[i][3] - ws.mol2_shifted[j][3]
        dist = sqrt(dx^2 + dy^2 + dz^2)
        
        dist_idx_float = dist / grid.Δr
        idx = clamp(round(Int, dist_idx_float), 1, grid.N)
        ws.dist_indices[i, j] = idx
        
        if dist < chain_params.r_cut
            σ_ij, ϵ_ij = chain_params.σ[1], chain_params.ϵ[1] 
            term = (σ_ij / dist)^6
            E_inter += T(4.0) * ϵ_ij * (term^2 - term + T(0.25)) - chain_params.LJ_shift
        end
        if dist < r_max
            idx_low = max(1, floor(Int, dist_idx_float))
            idx_high = min(grid.N, idx_low + 1)
            fraction = dist_idx_float - floor(dist_idx_float)
            
            s1 = chain_params.site_types[i]
            s2 = chain_params.site_types[j]
            
            w_low, w_high = W_solv[s1, s2, idx_low], W_solv[s1, s2, idx_high]
            E_inter += w_low * (T(1.0) - fraction) + w_high * fraction
        end
    end
    return E_inter
end

function sample_direct!(
    h_sim::Array{T,3}, configs::Vector{Molecule{T}}, MC_steps::Int, 
    start_n::Int, stop_n::Int, chain_params::ChainParameters{T}, 
    sys_params::SystemParameters{T}, W_solv::Array{T,3}, grid::RadialGrid{T}
) where {T}
    N_configs = length(configs)
    N_monomers = sys_params.N_monomers
    β = 1.0 / (sys_params.k_B * sys_params.T_sys)
    
    n_workspaces = max(Threads.nthreads(), Threads.maxthreadid())
    workspaces =[ThreadWorkspace(N_monomers, sys_params.N_sites, grid.N, T) for _ in 1:n_workspaces]
    
    prog = Progress(MC_steps, dt=0.1, desc="Direct Sampling: ", showspeed=true)
    
    Threads.@threads for step in 1:MC_steps
        t_id = Threads.threadid()
        rng = TaskLocalRNG()
        ws = workspaces[t_id]
        
        for n in start_n:stop_n
            z_shift = n * grid.Δr
            mol1 = configs[rand(rng, 1:N_configs)]
            mol2 = configs[rand(rng, 1:N_configs)]
            s1_idx = rand(rng, 1:N_monomers)
            s2_idx = rand(rng, 1:N_monomers)
            
            E_inter = evaluate_two_chain!(ws, mol1, mol2, s1_idx, s2_idx, z_shift, rng, chain_params, W_solv, grid)
            weight = (n^2) * exp(-β * E_inter)
            
            @inbounds for i in 1:N_monomers, j in 1:N_monomers
                dist_idx = ws.dist_indices[i, j]
                s1 = chain_params.site_types[i]
                s2 = chain_params.site_types[j]
                
                ws.g_r_accum[s1, s2, dist_idx] += weight
            end
        end
        ProgressMeter.next!(prog)
    end
    ProgressMeter.finish!(prog)
    
    h_sim .= 0.0
    for ws in workspaces
        h_sim .+= ws.g_r_accum
    end
    
    norm_const = 144.0  
    for i in 1:sys_params.N_sites, j in 1:sys_params.N_sites, k in 1:grid.N
        if k <= stop_n
            g_val = h_sim[i, j, k] / (k^2 * MC_steps * norm_const)
            h_sim[i, j, k] = g_val - 1.0 
        else
            # BUG FIXED: Beyond the sampling cutoff, h(r) naturally decays to 0.0
            h_sim[i, j, k] = 0.0
        end
    end
end

"""
    export_xyz(filename, configs)

Saves an array of molecules to a standard .xyz file for visualization in VMD/Ovito.
"""
function export_xyz(filename::String, configs::Vector{Molecule{T}}) where {T}
    open(filename, "w") do io
        for (c_idx, mol) in enumerate(configs)
            println(io, length(mol))
            println(io, "PE24 - Config $c_idx")
            for atom in mol
                @printf(io, "C %10.5f %10.5f %10.5f\n", atom[1], atom[2], atom[3])
            end
        end
    end
    println("Saved $(length(configs)) configurations to $filename")
end