# 2 atoms: dark and bright state projections

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
    using AtomicArrays
    using Revise
    using ProgressMeter
    using HDF5, FileIO, Printf

    using AtomicArraysStatistics

    import EllipsisNotation: Ellipsis
    const .. = Ellipsis()
end

begin
    const EMField = field.EMField
    # const em_inc_function = AtomicArrays.field.gauss
    const em_inc_function = AtomicArrays.field.plane
    const NMAX = 50
    const NMAX_T = 5

    const PATH_FIGS, PATH_DATA = AtomicArraysStatistics.path()

    # System parameters
    const γ = 1.
    const e_dipole = [1., 0, 0]
    const T = [0:0.05:500;]
    const N = 2
    const Ncenter = 1

    # Define Spin 1/2 operators
    spinbasis = SpinBasis(1//2)
    sx = sigmax(spinbasis)
    sy = sigmay(spinbasis)
    sz = sigmaz(spinbasis)
    sp = sigmap(spinbasis)
    sm = sigmam(spinbasis)
    I_spin = identityoperator(spinbasis)
end


begin
    # Parameters to vary
    dir_list = ["R", "L"];
    Delt_list = range(-2.0, 2.0, NMAX);
    E_list = range(1e-2, 2.0e-0, NMAX);
    d_list = range(1.0e-1, 10e-1, NMAX);

    projections_D = zeros(2, NMAX, NMAX, NMAX);
    projections_B = zeros(2, NMAX, NMAX, NMAX);
    "done"
end

progress = Progress(2 * NMAX * NMAX * NMAX)
Threads.@threads for kkiijjmm in CartesianIndices((2, NMAX, NMAX, NMAX))
    (kk, ii, jj, mm) = (Tuple(kkiijjmm)[1], Tuple(kkiijjmm)[2], 
    Tuple(kkiijjmm)[3], Tuple(kkiijjmm)[4])

    DIRECTION = dir_list[kk]
    Delt_0 = Delt_list[ii]
    E_ampl = E_list[jj]
    a = d_list[mm]

    # Atoms

    pos = geometry.chain_dir(a, N; dir="z", pos_0=[0, 0, -a / 2])
    Delt = [(i < N) ? 0.0 : Delt_0 for i = 1:N]
    S = SpinCollection(pos, e_dipole; gammas=γ, deltas=Delt)

    # Incident field
    E_kvec = 2π
    if (DIRECTION == "R")
        E_pos0 = [0.0,0.0,0.0]
        E_polar = [1.0, 0im, 0.0]
        E_angle = [0.0, 0.0]  # {θ, φ}
    elseif (DIRECTION == "L")
        E_pos0 = [0.0,0.0,0.0*a]
        E_polar = [-1.0, 0im, 0.0]
        E_angle = [π, 0.0]  # {θ, φ}
    else
        println("DIRECTION wasn't specified")
    end

    # Atom-field interaction
    E_inc = EMField(E_ampl, E_kvec, E_angle, E_polar;
        position_0=E_pos0, waist_radius=0.1)
    E_vec = [em_inc_function(S.spins[k].position, E_inc) for k = 1:N]
    Om_R = field.rabi(E_vec, S.polarizations)

    # Hamiltonian
    Γ, J = AtomicArrays.quantum.JumpOperators(S)
    Jdagger = [dagger(j) for j = J]
    Ω = AtomicArrays.interaction.OmegaMatrix(S)
    H = AtomicArrays.quantum.Hamiltonian(S) - sum(Om_R[j] * J[j] +
                                                  conj(Om_R[j]) * Jdagger[j]
                                                  for j = 1:N)
    # Dark and Bright states
    ψ_D = 1.0/sqrt(2.0) * (Ket(basis(H), [0,1,0,0]) - 
                           sign(Γ[1,2])*Ket(basis(H), [0,0,1,0]))
    ρ_D = ψ_D ⊗ dagger(ψ_D)
    ψ_B = 1.0/sqrt(2.0) * (Ket(basis(H), [0,1,0,0]) + 
                           sign(Γ[1,2])*Ket(basis(H), [0,0,1,0]))
    ρ_B = ψ_B ⊗ dagger(ψ_B)

    # Steady-state
#   ρ_ss = QuantumOptics.steadystate.eigenvector(H, J; rates=Γ)

    phi = 0.
    theta = pi/1.
    Ψ₀ = AtomicArrays.quantum.blochstate(phi,theta,N)
    ρ₀ = Ψ₀⊗dagger(Ψ₀)
    _, ρ_t = QuantumOptics.timeevolution.master_h(T, ρ₀, H, J; rates=Γ)
    ρ_ss = ρ_t[end]

    # Projections
    projections_D[kk, ii, jj, mm] = real(tr(ρ_ss * ρ_D))
    projections_B[kk, ii, jj, mm] = real(tr(ρ_ss * ρ_B))

    next!(progress)
end

maximum(projections_D[1,..])

maximum(projections_B[1,..])

begin
    eff_D = AtomicArrays.field.objective(projections_B[1,..], projections_B[2,..]);
    (maximum(eff_D), argmax(eff_D), (Delt_list[argmax(eff_D)[1]], E_list[argmax(eff_D)[2]], d_list[argmax(eff_D)[3]]))
end

let
    # Position of the maximum
    perm = sortperm([maximum(projections_D[2, :, i, :]) for i = 1:NMAX],
                    rev=true)
    idx = 12
    (perm[idx], argmax(projections_D[2, :, perm[idx], :])[1], maximum(projections_D[2, :, perm[idx], :]), 
    (Delt_list[argmax(projections_D[2, :, perm[idx], :])[1]], E_list[perm[idx]], d_list[argmax(projections_D[2, :, perm[idx], :])[2]]))

end


let
    idx = 13
    x = Delt_list
    y = d_list
    c_D_1 = projections_B[1, :, idx, :]
    c_D_2 = projections_B[2, :, idx, :]
    cmap = "viridis"

    fig, ax = plt.subplots(1, 2, figsize=(9,3))
    im1 = ax[1].pcolormesh(x, y, c_D_1', cmap=cmap)
    ax[1].set_title(L"R")
    ax[1].set_xlabel(L"\Delta")
    ax[1].set_ylabel(L"a")
    fig.colorbar(im1, ax=ax[1])
    im2 = ax[2].pcolormesh(x, y, c_D_2', cmap=cmap)
    ax[2].set_title(L"L")
    ax[2].set_xlabel(L"\Delta")
    fig.colorbar(im2, ax=ax[2])
    fig
end

let
    idx = 13
    x = Delt_list
    y = d_list
    c_D_1 = eff_D[:, idx, :]
    cmap = "viridis"

    fig, ax = plt.subplots(1, 1, figsize=(4,3))
    im1 = ax.pcolormesh(x, y, c_D_1', cmap=cmap)
    ax.set_title("Contrast for B population")
    ax.set_xlabel(L"\Delta")
    ax.set_ylabel(L"a")
    fig.colorbar(im1, ax=ax)
    fig
end

projections_D[2, 17, 7, 2]
(Delt_list[17], E_list[7], d_list[2])

let

    DIRECTION = "R"
    Delt_0 = 0.5
    E_ampl = 0.2
    a = 0.18

    # Atoms

    pos = geometry.chain_dir(a, N; dir="z", pos_0=[0, 0, -a / 2])
    Delt = [(i < N) ? 0.0 : Delt_0 for i = 1:N]
    S = SpinCollection(pos, e_dipole; gammas=γ, deltas=Delt)

    # Incident field
    E_kvec = 2π
    if (DIRECTION == "R")
        E_pos0 = [0.0,0.0,0.0]
        E_polar = [1.0, 0im, 0.0]
        E_angle = [0.0, 0.0]  # {θ, φ}
    elseif (DIRECTION == "L")
        E_pos0 = [0.0,0.0,0.0*a]
        E_polar = [-1.0, 0im, 0.0]
        E_angle = [π, 0.0]  # {θ, φ}
    else
        println("DIRECTION wasn't specified")
    end

    # Atom-field interaction
    E_inc = EMField(E_ampl, E_kvec, E_angle, E_polar;
        position_0=E_pos0, waist_radius=0.1)
    E_vec = [em_inc_function(S.spins[k].position, E_inc) for k = 1:N]
    Om_R = field.rabi(E_vec, S.polarizations)

    # Hamiltonian
    Γ, J = AtomicArrays.quantum.JumpOperators(S)
    Jdagger = [dagger(j) for j = J]
    Ω = AtomicArrays.interaction.OmegaMatrix(S)
    H = AtomicArrays.quantum.Hamiltonian(S) - sum(Om_R[j] * J[j] +
                                                  conj(Om_R[j]) * Jdagger[j]
                                                  for j = 1:N)
    # Dark and Bright states
    ψ_D = 1.0/sqrt(2.0) * (Ket(basis(H), [0,1,0,0]) - 
                           sign(Γ[1,2])*Ket(basis(H), [0,0,1,0]))
    ρ_D = ψ_D ⊗ dagger(ψ_D)
    ψ_B = 1.0/sqrt(2.0) * (Ket(basis(H), [0,1,0,0]) + 
                           sign(Γ[1,2])*Ket(basis(H), [0,0,1,0]))
    ρ_B = ψ_B ⊗ dagger(ψ_B)

    # Steady-state
    ρ_ss = QuantumOptics.steadystate.eigenvector(H, J; rates=Γ)

    phi = 0.
    theta = pi/1.
    Ψ₀ = AtomicArrays.quantum.blochstate(phi,theta,N)
    ρ₀ = Ψ₀⊗dagger(Ψ₀)
    _, ρ_t = QuantumOptics.timeevolution.master_h(T, ρ₀, H, J; rates=Γ)
end