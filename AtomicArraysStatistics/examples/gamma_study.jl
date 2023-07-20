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

using LinearAlgebra
using QuantumOptics
using PyPlot
# using GLMakie
using AtomicArrays
using Revise

using AtomicArraysStatistics

begin
    const EMField = field.EMField
    const em_inc_function = AtomicArrays.field.gauss
    # const em_inc_function = AtomicArrays.field.plane
    const NMAX = 100
    const NMAX_T = 5
    const DIRECTION = "L"

    const PATH_FIGS, PATH_DATA = AtomicArraysStatistics.path()
end

function g2_jump_opers(rho_ss::Operator, J)
    N = length(J)
    num = real(sum([AtomicArraysStatistics.correlation_3op_1t(rho_ss, 
    dagger(J[i]), dagger(J[j])*J[j], J[i]) for i = 1:N, j = 1:N]))
    denom = real(sum([QuantumOptics.expect(dagger(J[i])*J[i], rho_ss) for i = 1:N]).^2)
    return num / denom
end


begin
    # System parameters
    const a = 0.137#0.18
    const γ = 1.
    const e_dipole = [1., 0, 0]
    const T = [0:0.05:500;]
    const N = 4
    const Ncenter = 1

    const pos = geometry.chain_dir(a, N; dir="z", pos_0=[0, 0, -(N-1)*a / 2])
    const Delt = [(i < N) ? 0.0 : 0.15 for i = 1:N]
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

    w, v = eigenstates(dense(H))

    # Jump operators description
    J_s = AtomicArraysStatistics.jump_op_source_mode(Γ, J)
end

eigen(Γ)