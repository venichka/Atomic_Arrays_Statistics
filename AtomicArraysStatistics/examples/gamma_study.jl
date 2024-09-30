# Activate project

begin
    if pwd()[end-21:end] == "AtomicArraysStatistics"
        PATH_ENV = "."
    else
        PATH_ENV = "../"
    end
    using Pkg
    Pkg.activate(PATH_ENV)
end

begin
    using LinearAlgebra
    using QuantumOptics
    using PyPlot
    # using GLMakie
    using AtomicArrays
    using Revise

    using AtomicArraysStatistics
end

begin
    const EMField = field.EMField
    const em_inc_function = AtomicArrays.field.gauss
    # const em_inc_function = AtomicArrays.field.plane
    const NMAX = 100
    const NMAX_T = 5
    const DIRECTION = "L"

    const PATH_FIGS, PATH_DATA = AtomicArraysStatistics.path()
end

begin
    # System parameters
    const a = 0.0#0.18
    const γ = 0.1
    const e_dipole = [1., 0, 0]
    const T = [0:0.05:500;]
    const N = 2
    const Ncenter = 1

    const pos = geometry.chain_dir(a, N; dir="z", pos_0=[0, 0, -(N-1)*a / 2])
    const Delt = [(i < N) ? 0.0 : 0.0 for i = 1:N]
    const S = SpinCollection(pos, e_dipole; gammas=γ, deltas=Delt)

    # Define Spin 1/2 operators
    spinbasis = SpinBasis(1//2)
    sx = sigmax(spinbasis)
    sy = sigmay(spinbasis)
    sz = sigmaz(spinbasis)
    sp = sigmap(spinbasis)
    sm = sigmam(spinbasis)
    I_spin = identityoperator(spinbasis)

    # Incident field
    E_ampl = 0.66 + 0im
    E_kvec = 2π
    E_w_0 = 0.5
    if (DIRECTION == "R")
        E_pos0 = [0.0,0.0,-a/2]
        E_polar = [1.0, 0im, 0.0]
        E_angle = [0.0, 0.0]  # {θ, φ}
    elseif (DIRECTION == "L")
        E_pos0 = [0.0,0.0,a/2]
        E_polar = [-1.0, 0im, 0.0]
        E_angle = [π, 0.0]  # {θ, φ}
    else
        println("DIRECTION wasn't specified")
    end

    E_inc = EMField(E_ampl, E_kvec, E_angle, E_polar;
        position_0=E_pos0, waist_radius=E_w_0)
end

"""Impinging field"""

let
    x = range(-3.0, 3.0, NMAX)
    y = 0.0
    z = range(-5.0, 5.0, NMAX)
    e_field = Matrix{ComplexF64}(undef, length(x), length(z))
    for i in eachindex(x)
        for j in eachindex(z)
            e_field[i, j] = em_inc_function([x[i], y, z[j]], E_inc)[1]
        end
    end
    fig_0 = PyPlot.figure(figsize=(7, 4))
    PyPlot.contourf(x, z, real(e_field)', 30)
    for p in pos
        PyPlot.plot(p[1], p[3], "o", color="w", ms=2)
    end
    PyPlot.xlabel("x")
    PyPlot.ylabel("z")
    PyPlot.colorbar(label="Amplitude")
    display(fig_0)
end

"System Hamiltonian"

begin
    # Field-spin interaction
    E_vec = [em_inc_function(S.spins[k].position, E_inc) for k = 1:N]
    Om_R = field.rabi(E_vec, S.polarizations)


    Γ, J = AtomicArrays.quantum.JumpOperators(S)
    Jdagger = [dagger(j) for j = J]
    Ω = AtomicArrays.interaction.OmegaMatrix(S)
    H = AtomicArrays.quantum.Hamiltonian(S) - sum(Om_R[j] * J[j] +
                                                            conj(Om_R[j]) * Jdagger[j]
                                                            for j = 1:N)

    H.data

    H_nh = H - 0.5im * sum(Γ[j,k] * Jdagger[j] * J[k] for j = 1:N, k = 1:N)

    w, v = eigenstates(dense(H))

    # Jump operators description
    J_s = AtomicArraysStatistics.jump_op_source_mode(Γ, J)
end

begin 
    w_nh, v_nh = eigenstates(dense(H_nh))
end

eigen(Γ)
_, J_s0 = diagonaljumps(Γ, J)

phi = 0.0
theta = pi / 1.0
psi0 = AtomicArrays.quantum.blochstate(phi, theta, N)
psi1 = AtomicArrays.quantum.blochstate(phi, 0.0, N)

let 
    rho_G = (psi0) ⊗ dagger(psi0)
    rho_E = (psi1) ⊗ dagger(psi1)
    rho_D = (dagger(J_s0[1]) * psi0) ⊗ dagger(dagger(J_s0[1]) * psi0)
    rho_B = (dagger(J_s0[2]) * psi0) ⊗ dagger(dagger(J_s0[2]) * psi0)
    basis_0 = [rho_G, rho_D, rho_B, rho_E]
    basis_1 = [v_nh[i] ⊗ dagger(v_nh[i]) for i in 1:2^N]
    matrix_proj = [real(tr(i * j)) for i in basis_1, j in basis_0]
end

v_nh[sortperm(abs.(imag(w_nh)))]


transpose(real((dagger(J_s0[1]) * psi0).data))
transpose(real((J_s0[3] * psi1).data))
transpose(real(AtomicArraysStatistics.dicke_state(N, N ÷ 2, 0, J).data))
transpose(real(AtomicArraysStatistics.dicke_state(N, N / 2, 0.5, J).data))
norm(dagger(J_s0[3]) * psi0)

function dark_states(J_s)
    N = length(J_s)
    psi0 = Ket(basis(J_s[1]), [(i < 2^N) ? 0 : 1 for i in 1:2^N])
    if N > 2
        d1 = [dagger(j) * psi0 for j in J_s]
        d2 = [dagger(J_s[end]) * d for d in d1]
        return [d1, d2 ./ norm.(d2)]
    else
        d1 = [dagger(j) * psi0 for j in J_s]
        return d1
    end
end

J_z, J_z_sum, J_sum, J2 = AtomicArraysStatistics.collective_ops_all(J)
dagger(dark_states(J_s0)[2][1])*sum(J_z)*dark_states(J_s0)[2][1]
dagger(dark_states(J_s0)[1][2])*J2*dark_states(J_s0)[1][2]
transpose(real((((J_s0[2])) * psi1).data))
transpose(real((((sum(J))*(J_s0[2]) * psi1/ sqrt(2) ).data)))
Γ

psi0 = Ket(basis(J_s0[1]), [(i < 2^N) ? 0 : 1 for i in 1:2^N])
begin 
    d00_1 = 1/2*(dagger(J[3])*dagger(J[1]) - dagger(J[3])*dagger(J[2])-dagger(J[4])*dagger(J[1])+dagger(J[4])*dagger(J[2]))*psi0
    d00_2 = 1/2/sqrt(3)*(2*dagger(J[1])*dagger(J[2]) + 2*dagger(J[3])*dagger(J[4])
    - dagger(J[1])*dagger(J[4]) - dagger(J[2])*dagger(J[3])
    - dagger(J[2])*dagger(J[4]) - dagger(J[1])*dagger(J[3]))*psi0
    print("|0,0>_1 = ", transpose(real(d00_1.data)), "\n")
    print("|0,0>_2 = ", transpose(real(d00_2.data)), "\n")
    print("j*(j+1)_1 = ", real(dagger(d00_1)*J2*d00_1), 
          " m_1 = ", real(dagger(d00_1)*J_z_sum*d00_1), "\n")
    print("j*(j+1)_2 = ", real(dagger(d00_2)*J2*d00_2), 
          " m_2 = ", real(dagger(d00_2)*J_z_sum*d00_2), "\n")
end

J2_test(state) = dagger(state) * J2 * state
Jz_test(state) = dagger(state) * J_z_sum * state
J2_test.(ds[2])
begin
    ds = dark_states(J_s0)
    d00_1t = 1/sqrt(2)*(dagger(+ J[1] + J[2] + J[3] - J[4])*ds[1][2])
    d00_2t = 1/sqrt(2)*(dagger(+ J[1] + J[2] + J[3] - J[4])*ds[1][1])
    d00_3t = 1/sqrt(2)*(dagger(+ J[1] + J[2] + J[3] + J[4])*ds[1][3])
    J2_test.([d00_1t, d00_2t, d00_3t])
end
norm(d00_3t)
transpose(real(d00_3t.data))
dagger(d00_3t) * J2 * d00_3t
dagger(d00_3t) * J_z_sum * d00_3t
transpose(real(1/sqrt(2)*(dagger(J_sum)*ds[1][1]).data))
transpose(real(1/2*(dagger(J_sum)^2*ds[1][1]).data))
transpose(real((J_s0[1]*psi1).data))

tst = AtomicArraysStatistics.spherical_basis_jm_4(J)
J2_test.(tst[1])
[J2_test.(i) for i in tst[2]]
J2_test.(tst[3][1])
Jz_test.(tst[1])
[Jz_test.(i) for i in tst[2]]
Jz_test.(tst[3][1])

function find_state_index(states::Vector{Vector}, j::Int, m::Int; degeneracy::Union{Int, Nothing}=nothing)
    # Determine min and max j from the states input
    max_j = length(states) - 1  # max_j is determined by the length of states
    min_j = 0  # min_j is assumed to be 0 based on the structure
    
    # Create j_map dynamically based on max_j
    j_map = Dict(max_j => 1)
    for i in 1:max_j
        j_map[max_j - i] = i + 1
    end
    
    # Ensure j is within the valid range
    if j < min_j || j > max_j
        error("Invalid j value. Valid range is $min_j to $max_j.")
    end
    
    # Ensure m is within the valid range of -j to +j
    if m < -j || m > j
        error("Invalid m value. For j = $j, valid range is -$j to +$j.")
    end

    # Map j to the correct index in `states`
    j_idx = j_map[j]

    # Now map m to the correct sublist in the selected j vector
    m_idx = j + 1 + m  # for example, m = -j will be 1, m = -j+1 will be 2, and so on

    # Retrieve the specific |j,m> state(s)
    state_group = states[j_idx][m_idx]

    # Determine degeneracy based on the structure
    if isa(state_group, Vector)
        num_degenerate = length(state_group)
        if isnothing(degeneracy)
            return j_idx, m_idx  # Return the whole set of degenerate states
        elseif degeneracy > num_degenerate || degeneracy < 1
            error("Invalid degeneracy index. Available indices: 1 to $num_degenerate.")
        else
            return j_idx, m_idx, degeneracy
        end
    else
        # No degeneracy, return the single state
        return j_idx, m_idx
    end
end

function find_flattened_state_index(states::Vector{Vector}, j::Int, m::Int; degeneracy::Union{Int, Nothing}=nothing)
    # Determine min and max j from the states input
    max_j = length(states) - 1  # max_j is determined by the length of states
    min_j = 0  # min_j is assumed to be 0 based on the structure
    
    # Create j_map dynamically based on max_j
    j_map = Dict(max_j => 1)
    for i in 1:max_j
        j_map[max_j - i] = i + 1
    end
    
    # Ensure j is within the valid range
    if j < min_j || j > max_j
        error("Invalid j value. Valid range is $min_j to $max_j.")
    end
    
    # Ensure m is within the valid range of -j to +j
    if m < -j || m > j
        error("Invalid m value. For j = $j, valid range is -$j to +$j.")
    end
    
    # Map j to the correct index in `states`
    j_idx = j_map[j]

    # Now map m to the correct sublist in the selected j vector
    m_idx = j + 1 + m  # for example, m = -j will be 1, m = -j+1 will be 2, and so on

    # Retrieve the specific |j,m> state(s)
    state_group = states[j_idx][m_idx]

    # Flatten the states vector
    flattened_states = vcat(vcat(states...)...)

    # Determine degeneracy based on the structure and find the correct index
    if isa(state_group, Vector)
        num_degenerate = length(state_group)
        if isnothing(degeneracy)
            return findall(x -> x in state_group, flattened_states)  # Return all indices for the degenerate states
        elseif degeneracy > num_degenerate || degeneracy < 1
            error("Invalid degeneracy index. Available indices: 1 to $num_degenerate.")
        else
            specific_state = state_group[degeneracy]
            return findfirst(x -> x == specific_state, flattened_states)  # Return the index of the specific state
        end
    else
        # No degeneracy, return the index of the single state
        return findfirst(x -> x == state_group, flattened_states)
    end
end



test1 = map(x -> AtomicArraysStatistics.find_flattened_state_index(tst, x),[(2,-2), (2, 2)])
vcat(vcat(tst...)...)[test1]
find_state_index(tst, 2,2; degeneracy=nothing)
find_flattened_state_index(tst, 1,-1; degeneracy=nothing)
tst[3][1][1]
AtomicArraysStatistics.state_string.([(1,-1,1), ()])

ψ_D = 1.0/sqrt(2.0) * (Ket(basis(H), [0,1,0,0]) - 
Ket(basis(H), [0,0,1,0]))
ψ_B = 1.0/sqrt(2.0) * (Ket(basis(H), [0,1,0,0]) + 
Ket(basis(H), [0,0,1,0]))

