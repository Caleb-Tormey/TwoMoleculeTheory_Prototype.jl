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
    
    # At the bottom of sample_direct!
    norm_const = 144.0  
    
    # Safe limit: 10 * sigma
    splice_n = round(Int, 10.0 * chain_params.σ[1] / grid.Δr)

    for i in 1:sys_params.N_sites, j in 1:sys_params.N_sites, k in 1:grid.N
        if k <= splice_n
            g_val = h_sim[i, j, k] / (k^2 * MC_steps * norm_const)
            h_sim[i, j, k] = g_val - 1.0 
        else
            h_sim[i, j, k] = 0.0 # Will be overwritten by PRISM tail
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

# ---------------------------------------------------------
# Window Sampling Transformations and Energy Evaluations
# ---------------------------------------------------------
# --- Rigid Body Transformations ---
function translate_chain!(molecule::Molecule{T}, shift::SVector{3, T}) where {T}
    @inbounds for i in 1:length(molecule)
        molecule[i] = SVector{4, T}(molecule[i][1] + shift[1], molecule[i][2] + shift[2], molecule[i][3] + shift[3], 1.0)
    end
end

function rotate_chain!(molecule::Molecule{T}, pivot_idx::Int, rot_mat::SMatrix{3, 3, T}) where {T}
    pivot_pos = SVector{3, T}(molecule[pivot_idx][1:3])
    @inbounds for i in 1:length(molecule)
        p = SVector{3, T}(molecule[i][1:3]) - pivot_pos
        p_rot = rot_mat * p
        molecule[i] = SVector{4, T}(pivot_pos[1] + p_rot[1], pivot_pos[2] + p_rot[2], pivot_pos[3] + p_rot[3], 1.0)
    end
end

# --- Isolated Intermolecular Energy ---
function calc_intermolecular_energy(
    mol1::Molecule{T}, mol2::Molecule{T},
    chain_params::ChainParameters{T}, W_solv::Array{T,3}, grid::RadialGrid{T}
) where {T}
    E_inter = T(0.0)
    N = length(mol1)
    r_max = grid.N * grid.Δr
    @inbounds for i in 1:N, j in 1:N
        dx = mol1[i][1] - mol2[j][1]
        dy = mol1[i][2] - mol2[j][2]
        dz = mol1[i][3] - mol2[j][3]
        dist = sqrt(dx^2 + dy^2 + dz^2)
        
        if dist < chain_params.r_cut
            σ_ij, ϵ_ij = chain_params.σ[1], chain_params.ϵ[1] 
            term = (σ_ij / dist)^6
            E_inter += T(4.0) * ϵ_ij * (term^2 - term + T(0.25)) - chain_params.LJ_shift
        end
        if dist < r_max
            dist_idx_float = dist / grid.Δr
            idx_low = max(1, floor(Int, dist_idx_float))
            idx_high = min(grid.N, idx_low + 1)
            fraction = dist_idx_float - floor(dist_idx_float)
            
            s1 = chain_params.site_types[i]
            s2 = chain_params.site_types[j]
            
            w_low = W_solv[s1, s2, idx_low]
            w_high = W_solv[s1, s2, idx_high]
            E_inter += w_low * (T(1.0) - fraction) + w_high * fraction
        end
    end
    return E_inter
end

# --- Window Sampling MC Step ---
function MC_step_window!(
    mol1::Molecule{T}, mol2::Molecule{T}, 
    temp_mol1::Molecule{T}, temp_mol2::Molecule{T},
    curr_E_tot::T, rng::AbstractRNG,
    chain_params::ChainParameters{T}, sys_params::SystemParameters{T},
    W_solv::Array{T,3}, grid::RadialGrid{T},
    r_min::T, r_max::T,
    angle_range::T, dihedral_range::T, trans_range::T
) where {T}
    N = length(mol1)
    β = 1.0 / (sys_params.k_B * sys_params.T_sys)
    
    temp_mol1 .= mol1
    temp_mol2 .= mol2
    
    # 1. Pivot on Mol 1
    if rand(rng, 1:2) == 1
        bend_idx = rand(rng, 2:(N-1))
        Δθ = (rand(rng, T) - T(0.5)) * 2 * angle_range
        v1 = SVector{3, T}(temp_mol1[bend_idx-1][1:3]) - SVector{3, T}(temp_mol1[bend_idx][1:3])
        v2 = SVector{3, T}(temp_mol1[bend_idx+1][1:3]) - SVector{3, T}(temp_mol1[bend_idx][1:3])
        axis_bend = normalize(cross(v1, v2))
        if (bend_idx - 1) <= (N - bend_idx)
            rotate_arm!(temp_mol1, bend_idx, axis_bend, Δθ, 1, bend_idx - 1)
        else
            rotate_arm!(temp_mol1, bend_idx, axis_bend, Δθ, bend_idx + 1, N)
        end
    else
        twist_idx = rand(rng, 2:(N-2))
        Δϕ = (rand(rng, T) - T(0.5)) * 2 * dihedral_range
        axis_twist = normalize(SVector{3, T}(temp_mol1[twist_idx+1][1:3]) - SVector{3, T}(temp_mol1[twist_idx][1:3]))
        if (twist_idx - 1) <= (N - twist_idx)
            rotate_arm!(temp_mol1, twist_idx, axis_twist, Δϕ, 1, twist_idx - 1)
        else
            rotate_arm!(temp_mol1, twist_idx, axis_twist, Δϕ, twist_idx + 1, N)
        end
    end

    # 2. Pivot on Mol 2
    if rand(rng, 1:2) == 1
        bend_idx = rand(rng, 2:(N-1))
        Δθ = (rand(rng, T) - T(0.5)) * 2 * angle_range
        v1 = SVector{3, T}(temp_mol2[bend_idx-1][1:3]) - SVector{3, T}(temp_mol2[bend_idx][1:3])
        v2 = SVector{3, T}(temp_mol2[bend_idx+1][1:3]) - SVector{3, T}(temp_mol2[bend_idx][1:3])
        axis_bend = normalize(cross(v1, v2))
        if (bend_idx - 1) <= (N - bend_idx)
            rotate_arm!(temp_mol2, bend_idx, axis_bend, Δθ, 1, bend_idx - 1)
        else
            rotate_arm!(temp_mol2, bend_idx, axis_bend, Δθ, bend_idx + 1, N)
        end
    else
        twist_idx = rand(rng, 2:(N-2))
        Δϕ = (rand(rng, T) - T(0.5)) * 2 * dihedral_range
        axis_twist = normalize(SVector{3, T}(temp_mol2[twist_idx+1][1:3]) - SVector{3, T}(temp_mol2[twist_idx][1:3]))
        if (twist_idx - 1) <= (N - twist_idx)
            rotate_arm!(temp_mol2, twist_idx, axis_twist, Δϕ, 1, twist_idx - 1)
        else
            rotate_arm!(temp_mol2, twist_idx, axis_twist, Δϕ, twist_idx + 1, N)
        end
    end

    # 3. Rigid Translation & Rotation of Mol 2
    shift = (rand(rng, SVector{3, T}) .- T(0.5)) .* trans_range
    translate_chain!(temp_mol2, shift)
    rot_mat = random_rotation_matrix(rng, T)
    rotate_chain!(temp_mol2, rand(rng, 1:N), rot_mat)

    # 4. Strict Window Bounding Check (using Middle Sites)
    mid = N ÷ 2
    r_mid_dist = norm(SVector{3, T}(temp_mol1[mid][1:3]) - SVector{3, T}(temp_mol2[mid][1:3]))
    if r_mid_dist < r_min || r_mid_dist > r_max
        return curr_E_tot, 0
    end

    # 5. Energy Eval & Metropolis
    E_intra1 = calc_internal_energy(temp_mol1, chain_params, sys_params, W_solv, grid)
    E_intra2 = calc_internal_energy(temp_mol2, chain_params, sys_params, W_solv, grid)
    E_inter  = calc_intermolecular_energy(temp_mol1, temp_mol2, chain_params, W_solv, grid)
    new_E_tot = E_intra1 + E_intra2 + E_inter

    ΔE = new_E_tot - curr_E_tot
    if ΔE <= 0.0 || rand(rng, T) <= exp(-ΔE * β)
        mol1 .= temp_mol1
        mol2 .= temp_mol2
        return new_E_tot, 1
    else
        return curr_E_tot, 0
    end
end

# --- Core Window Runner ---
function run_window(
    mol_pool::Vector{Molecule{T}},
    sys_params::SystemParameters{T}, chain_params::ChainParameters{T}, W_solv::Array{T,3}, grid::RadialGrid{T},
    r_min::T, r_mid::T, r_max::T, dr::T, n_steps::Int, rng::AbstractRNG
) where {T}
    N_sites = sys_params.N_sites
    n_bins = grid.N
    
    hist_L = zeros(T, N_sites, N_sites, n_bins)
    hist_R = zeros(T, N_sites, N_sites, n_bins)
    
    mol1 = copy(mol_pool[rand(rng, 1:length(mol_pool))])
    mol2 = copy(mol_pool[rand(rng, 1:length(mol_pool))])
    
    mid = length(mol1) ÷ 2
    p1 = SVector{3,T}(mol1[mid][1:3])
    p2 = SVector{3,T}(mol2[mid][1:3])
    
    diff_vec = p2 - p1
    n_diff = norm(diff_vec)
    dir = n_diff == 0 ? SVector{3,T}(1.0, 0.0, 0.0) : diff_vec / n_diff
    translate_chain!(mol2, (p1 + dir * r_mid) - p2)
    
    temp_mol1 = copy(mol1)
    temp_mol2 = copy(mol2)
    
    curr_E_tot = calc_internal_energy(mol1, chain_params, sys_params, W_solv, grid) + 
                 calc_internal_energy(mol2, chain_params, sys_params, W_solv, grid) + 
                 calc_intermolecular_energy(mol1, mol2, chain_params, W_solv, grid)
    
    angle_range = T(20.0 * π / 180.0)
    dihedral_range = T(π / 2.0)
    trans_range = T(0.5)

    M_total = 0
    count_L = 0.0
    count_R = 0.0

    for step in 1:n_steps
        curr_E_tot, acc = MC_step_window!(
            mol1, mol2, temp_mol1, temp_mol2, curr_E_tot, rng,
            chain_params, sys_params, W_solv, grid,
            r_min, r_max, angle_range, dihedral_range, trans_range
        )
        
        if step > 10_000 && step % 50 == 0
            M_total += 1
            r_mid_curr = norm(SVector{3,T}(mol1[mid][1:3]) - SVector{3,T}(mol2[mid][1:3]))
            
            # Robust Partition Function Trackers
            if r_mid_curr <= r_mid
                count_L += 1.0
            else
                count_R += 1.0
            end
            
            for i in 1:length(mol1), j in 1:length(mol2)
                dist = norm(SVector{3,T}(mol1[i][1:3]) - SVector{3,T}(mol2[j][1:3]))
                dist_idx = ceil(Int, dist / dr)
                if 1 <= dist_idx <= n_bins
                    s1 = chain_params.site_types[i]
                    s2 = chain_params.site_types[j]
                    if r_mid_curr <= r_mid
                        hist_L[s1, s2, dist_idx] += 1.0
                    else
                        hist_R[s1, s2, dist_idx] += 1.0
                    end
                end
            end
        end
    end
    
    for idx in 1:n_bins
        r = (idx - 0.5) * dr
        if r > 0
            vol = 4 * T(π) * (r^2) * dr * M_total
            for s1 in 1:N_sites, s2 in 1:N_sites
                hist_L[s1, s2, idx] /= vol
                hist_R[s1, s2, idx] /= vol
            end
        end
    end
    
    # Return as fractions of total accepted states
    frac_L = M_total > 0 ? count_L / M_total : T(0.5)
    frac_R = M_total > 0 ? count_R / M_total : T(0.5)
    
    return hist_L, hist_R, frac_L, frac_R
end

# --- Orchestrator & Multi-Site Stitching ---
function sample_window!(
    h_sim::Array{T,3}, configs::Vector{Molecule{T}}, sampler::WindowSampler{T},
    chain_params::ChainParameters{T}, sys_params::SystemParameters{T},
    W_solv::Array{T,3}, grid::RadialGrid{T}
) where {T}
    dr = grid.Δr
    starts = range(2.0, step=sampler.win_width - sampler.overlap, stop=sampler.sim_r_cut - sampler.win_width)
    num_windows = length(starts)
    
    L_hists = Vector{Array{T,3}}(undef, num_windows)
    R_hists = Vector{Array{T,3}}(undef, num_windows)
    L_counts = zeros(T, num_windows)
    R_counts = zeros(T, num_windows)
    
    println("   -> Simulating $num_windows windows in parallel...")
    Threads.@threads for i in 1:num_windows
        rng = TaskLocalRNG()
        r_min = T(starts[i])
        r_max = r_min + sampler.win_width
        r_mid = r_min + sampler.overlap
        
        L_hists[i], R_hists[i], L_counts[i], R_counts[i] = run_window(configs, sys_params, chain_params, W_solv, grid, r_min, r_mid, r_max, dr, sampler.n_steps, rng)
    end
    
    # 1. Calculate pure thermodynamic scaling factors (alpha)
    alphas = zeros(T, num_windows)
    alphas[1] = 1.0
    for i in 2:num_windows
        alphas[i] = (R_counts[i-1] * alphas[i-1]) / max(L_counts[i], T(1e-8))
    end
    
    println("   -> Stitching overlapping matrices...")
    N_sites = sys_params.N_sites
    master_hist = zeros(T, N_sites, N_sites, grid.N)
    
    for s1 in 1:N_sites, s2 in 1:N_sites
        master_pair = zeros(T, grid.N)
        master_pair .+= L_hists[1][s1, s2, :]
        current_R = R_hists[1][s1, s2, :] .* alphas[1]
        
        # 2. Simple Arithmetic Mean for exact thermodynamic overlaps
        for i in 2:num_windows
            L_i_scaled = L_hists[i][s1, s2, :] .* alphas[i]
            R_i_scaled = R_hists[i][s1, s2, :] .* alphas[i]
            
            C_i = (current_R .+ L_i_scaled) .* T(0.5)
            master_pair .+= C_i
            current_R = R_i_scaled
        end
        master_pair .+= current_R
        master_hist[s1, s2, :] .= master_pair
    end
    
    # 3. Dynamic Truncation & Normalization
    max_chain_ext = sys_params.N_monomers * chain_params.l_bond
    valid_r_max = sampler.sim_r_cut - max_chain_ext
    
    idx_norm_end = min(grid.N, max(1, floor(Int, valid_r_max / dr)))
    idx_norm_start = max(1, floor(Int, (valid_r_max - 15.0) / dr))
    
    for s1 in 1:N_sites, s2 in 1:N_sites
        # Sample the flat physics EXACTLY before the drop-off begins
        tail_mean = mean(master_hist[s1, s2, idx_norm_start:idx_norm_end])
        if tail_mean > 0
            master_hist[s1, s2, :] ./= tail_mean
        end
        
        for k in 1:grid.N
            if k <= idx_norm_end
                h_sim[s1, s2, k] = master_hist[s1, s2, k] - 1.0
            else
                h_sim[s1, s2, k] = 0.0 # Force tail flat where physics are truncated
            end
        end
    end
end