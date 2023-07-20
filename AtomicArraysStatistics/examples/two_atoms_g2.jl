# Two atoms: g2 study

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
using GLMakie
using AtomicArrays
using Revise

using AtomicArraysStatistics

begin
    const EMField = field.EMField
    # const em_inc_function = AtomicArrays.field.gauss
    const em_inc_function = AtomicArrays.field.plane
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
    const delt_0 = 1e-1
    const a = (pi - delt_0) / (2*pi)#0.137
    const γ = 1.
    const e_dipole = [1., 0, 0]
    const T = [0:0.05:500;]
    const N = 2
    const Ncenter = 1

    const pos = geometry.chain_dir(a, N; dir="z", pos_0=[0, 0, -a / 2])
    const Delt = [(i < N) ? 0.0 : -γ*delt_0 for i = 1:N]
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
    E_ampl = 0.4 + 0im
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

    # Dark and Bright states
    ψ_D = 1.0/sqrt(2.0) * (Ket(basis(H), [0,1,0,0]) - 
                        sign(Γ[1,2])*Ket(basis(H), [0,0,1,0]))
    ψ_B = 1.0/sqrt(2.0) * (Ket(basis(H), [0,1,0,0]) + 
                        sign(Γ[1,2])*Ket(basis(H), [0,0,1,0]))

    # eigen(dense(H).data)
    w, v = eigenstates(dense(H))

    # Jump operators description
    J_s = AtomicArraysStatistics.jump_op_source_mode(Γ, J)

    # Directed-detection jump operators
    D = AtomicArraysStatistics.jump_op_direct_detection([10, 10, 10], 0.02^2*pi^2, S, 2π, J)
    D = AtomicArraysStatistics.jump_op_direct_detection(acos(1 / sqrt(3)), π/4, 0.02^2*pi^2, S, 2π, J)
end

"Dynamics"

begin
    # Initial state (Bloch state)
    const phi = 0.
    const theta = pi/1.

    # Time evolution

    # Quantum: master equation
    sx_master = zeros(Float64, N, length(T))
    sy_master = zeros(Float64, N, length(T))
    sz_master = zeros(Float64, N, length(T))

    embed(op::Operator,i) = QuantumOptics.embed(AtomicArrays.quantum.basis(S), i, op)

    function fout(t, rho)
        j = findfirst(isequal(t), T)
        for i = 1:N
            sx_master[i, j] = real(expect(embed(sx, i), rho))
            sy_master[i, j] = real(expect(embed(sy, i), rho))
            sz_master[i, j] = real(expect(embed(sz, i), rho))
        end
        return nothing
    end

    Ψ₀ = AtomicArrays.quantum.blochstate(phi,theta,N)
    ρ₀ = Ψ₀⊗dagger(Ψ₀)
    QuantumOptics.timeevolution.master_h(T, ρ₀, H, J; fout=fout, rates=Γ)

    sx_copy = copy(sx_master)
    sy_copy = copy(sy_master)
    sz_copy = copy(sz_master)

    QuantumOptics.timeevolution.master_h(T, ρ₀, H, J_s; fout=fout)
end
# Plots
let
    fig = PyPlot.figure(figsize=(8, 12))
    PyPlot.subplot(311)
    for i = 1:N
        PyPlot.plot(T, sx_master[i, :], label="J: atom "*string(i))
        PyPlot.plot(T, sx_copy[i, :], label="atom "*string(i))
    end
    PyPlot.xlabel("Time")
    PyPlot.ylabel(L"\langle \sigma_x \rangle")
    PyPlot.legend()

    PyPlot.subplot(312)
    for i = 1:N
        PyPlot.plot(T, sy_master[i, :], label="J: atom "*string(i))
        PyPlot.plot(T, sy_copy[i, :], label="atom "*string(i))
    end
    PyPlot.xlabel("Time")
    PyPlot.ylabel(L"\langle \sigma_y \rangle")
    PyPlot.legend()

    PyPlot.subplot(313)
    for i = 1:N
        PyPlot.plot(T, sz_master[i, :], label="J: atom "*string(i))
        PyPlot.plot(T, sz_copy[i, :], label="atom "*string(i))
    end
    PyPlot.xlabel("Time")
    PyPlot.ylabel(L"\langle \sigma_z \rangle")
    PyPlot.legend()
    display(fig)
end

"Correlation function of the 1st order"

begin
    tau = [0:0.05:1000;]
    ρ_ss = QuantumOptics.steadystate.eigenvector(H, J; rates=Γ)  # finding the steady-state

    corr_1 = QuantumOptics.timecorrelations.correlation(tau, ρ_ss, H, J,
                                                        Jdagger[1], J[1]; rates=Γ)
    corr_2 = QuantumOptics.timecorrelations.correlation(tau, ρ_ss, H, J,
                                                        Jdagger[2], J[2]; rates=Γ)
    corr_12 = QuantumOptics.timecorrelations.correlation(tau, ρ_ss, H, J,
                                                        Jdagger[1] + Jdagger[2],
                                                        J[1] + J[2]; rates=Γ)

    corr_1 = corr_1 .- corr_1[end]
    corr_2 = corr_2 .- corr_2[end]
    corr_12 = corr_12 .- corr_12[end]

    omega, spec_1 = QuantumOptics.timecorrelations.correlation2spectrum(tau, corr_1)
    omega, spec_2 = QuantumOptics.timecorrelations.correlation2spectrum(tau, corr_2)
    omega, spec_12 = QuantumOptics.timecorrelations.correlation2spectrum(tau, corr_12)
end
# Plots
let
    fig_0 = PyPlot.figure(figsize=(6, 8))
    PyPlot.subplot(211)
    PyPlot.plot(tau, real(corr_1), label="atom 1")
    PyPlot.plot(tau, real(corr_2), label="atom 2")
    PyPlot.plot(tau, real(corr_12), label="atom 1-2")
    PyPlot.xlabel(L"\tau")
    PyPlot.ylabel(L"\langle \sigma^\dag_i \sigma_i \rangle")
    PyPlot.legend()

    PyPlot.subplot(212)
    PyPlot.plot(omega, spec_1, label="atom 1")
    PyPlot.plot(omega, spec_2, label="atom 2")
    PyPlot.plot(omega, spec_12, label="atom 1-2")
    PyPlot.xlabel(L"\omega")
    PyPlot.ylabel(L"\mathrm{Spectrum}")
    PyPlot.xlim(-3,3)
    PyPlot.legend()

    display(gcf())
end


"g2 depending on angle"

begin
    tau_0 = T

    phi_var = range(0, 2π, NMAX)
    theta_var = range(0, π, NMAX ÷ 2)

    g2_result = zeros(2, NMAX)
    n_result = zeros(2, NMAX)

    # Steady-state for both directions
    ρ_ss_1 = ρ_ss
    ρ_ss_2 = ρ_ss
    H_eff_1 = H
    H_eff_2 = H
    H_1 = H
    H_2 = H
    g2_j_1 = 0.0; g2_j_2 = 0.0;
    for kk = 1:2
        E_angle = (kk == 1) ? [0, 0.0] : [π, 0.0] 
        E_polar = (kk == 1) ? [1.0, 0im, 0.0] : [-1.0, 0im, 0.0]
        E_pos0 = (kk == 1) ? [0.0, 0.0, -a/2] : [0.0, 0.0, a/2]
        E_inc = EMField(E_ampl, E_kvec, E_angle, E_polar;
            position_0=E_pos0, waist_radius=E_w_0)
        E_vec = [em_inc_function(S.spins[k].position, E_inc) for k = 1:N]
        Om_R = field.rabi(E_vec, S.polarizations)
        print(Om_R)
        Γ, J = AtomicArrays.quantum.JumpOperators(S)
        J_s = AtomicArraysStatistics.jump_op_source_mode(Γ, J)
        Jdagger = [dagger(j) for j = J]
        Ω = AtomicArrays.interaction.OmegaMatrix(S)
        H = AtomicArrays.quantum.Hamiltonian(S) - sum(Om_R[j] * J[j] +
                                                    conj(Om_R[j]) * Jdagger[j]
                                                    for j = 1:N)

        # H =  (sum(0.5*S.spins[i].delta * (Jdagger[i] * J[i] -J[i] * Jdagger[i]) 
        #          for i=1:N) - sum(Om_R[j] * J[j] + conj(Om_R[j]) * Jdagger[j]
        #                           for j = 1:N) + 
        #     sum(Ω[i,j]*Jdagger[i]*J[j] for i=1:N, j=1:N))

        if kk == 1
            ρ_ss_1 = QuantumOptics.steadystate.eigenvector(H, J; rates=Γ)
            H_eff_1 = AtomicArrays.quantum.Hamiltonian_eff(S) - sum(Om_R[j] * J[j] +
                                                    conj(Om_R[j]) * Jdagger[j]
                                                    for j = 1:N)
            H_1 = H
            g2_j_1 = g2_jump_opers(ρ_ss_1, J_s)
        elseif kk == 2
            ρ_ss_2 = QuantumOptics.steadystate.eigenvector(H, J; rates=Γ)
            H_eff_2 = AtomicArrays.quantum.Hamiltonian_eff(S) - sum(Om_R[j] * J[j] +
                                                    conj(Om_R[j]) * Jdagger[j]
                                                    for j = 1:N)
            H_2 = H
            g2_j_2 = g2_jump_opers(ρ_ss_2, J_s)
        end
    end

    w_1, v_1 = eigenstates(dense(H_1))
    w_2, v_2 = eigenstates(dense(H_2))
    w_eff_1, v_eff_1 = eigenstates(dense(H_eff_1))
    w_eff_2, v_eff_2 = eigenstates(dense(H_eff_2))

    ρ_D_1 = v_1[2]⊗dagger(v_1[2])
    ρ_D_2 = v_2[2]⊗dagger(v_2[2])
    ρ_D = ψ_D ⊗ dagger(ψ_D)
    ρ_B = ψ_B ⊗ dagger(ψ_B)
end

g2_jump_opers(ρ_D_2, J_s) - g2_jump_opers(ρ_D_1, J_s) 
tr(ρ_ss_1 * ρ_D)
tr(ρ_ss_2 * ρ_D)
tr(ρ_ss_1 * ρ_B)
tr(ρ_ss_2 * ρ_B)

[-0.0139431, -0.251986 - 0.253089*im, 0.149641 + 
0.0687009 *im, 0.922668]
[-0.0230892, -0.396428 - 0.346487*im, 0.179138 + 
0.0822428*im, 0.841969]
[-0.00709915, -0.131276 - 
0.178245*im, 0.128485 + 0.058988 *im, 0.965504]
ψ_per = Ket(basis(H), [-0.0230892, 0.179138 - 0.0822428*im, -0.396428 + 0.346487*im, 0.841969])
ρ_per = ψ_per ⊗ dagger(ψ_per)
tr(ρ_per * ρ_ss_1)
g2_jump_opers(ρ_per, J_s) 

Threads.@threads for kkii in CartesianIndices((2, NMAX))
    (kk, ii) = Tuple(kkii)[1], Tuple(kkii)[2]
    ρ_ss = (kk == 1) ? ρ_ss_1 : ρ_ss_2
    ϕ = phi_var[ii]
    # θ = theta_var[ii]
    D = AtomicArraysStatistics.jump_op_direct_detection(ϕ, pi / 4, 0.02^2*pi^2, S, 2π, J)
    g2 = AtomicArraysStatistics.coherence_function_g2(tau_0, H, J, D; rates=Γ, rho0=ρ_ss)
    g2_result[kk, ii] = real(g2[1])
    n_result[kk, ii] = real(QuantumOptics.expect(dagger(D)*D, ρ_ss))
end

function g2_analyt(theta, d, Omega21, Delta, Omega)
    (4 * Omega21^4(Delta * Omega * conj(Omega) - Omega21 * conj(Omega)^2 + Omega^2 * (-Omega21)) * (Delta * abs(Omega)^2 - Omega21 * (conj(Omega)^2 + Omega^2))) / (Delta^2 * ((Omega * Omega21 - Delta * conj(Omega)) * (Omega21 * conj(Omega) + Omega * Omega21 * exp(2 * im * pi * d * cos(theta)) - Delta * Omega) + Omega21 * conj(Omega) * (Omega * Omega21 + exp(-2 * im * pi * d * cos(theta)) * (Omega21 * conj(Omega) - Delta * Omega)) + (2 * (-Delta * Omega * conj(Omega) + Omega21 * conj(Omega)^2 + Omega^2 * Omega21)^2) / Delta^2)^2)
end

g2_analyt_data = [g2_analyt(i, 0.08, 0.507318, 1.63, 0.09087996823560401 + 0.04172327136617653*im) for i in phi_var]

let
    fig_01, ax = plt.subplots(1, 1, subplot_kw=Dict("projection" => "polar"), figsize=(5,5))
    ax.plot(phi_var, g2_result[1, :], color="red", linewidth=2, label="f")
    # ax.plot(phi_var, g2_result[1, :], color="blue", linewidth=2, label="exact")
    ax.plot(phi_var, [g2_j_1 for i = 1:NMAX], color="black", label="f_jump")
    ax.plot(phi_var, g2_result[2, :], color="blue", linewidth=2, label="b")
    # ax.plot(phi_var, 0.0045*real(g2_analyt_data), color="red", linewidth=2, label="approx")
    ax.plot(phi_var, [g2_j_2 for i = 1:NMAX], "--", color="black", label="b_jump")
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(true)
    ax.set_title(L"g^{(2)}(0)", va="bottom")
    ax.legend()
    display(fig_01)

    # fig_01.savefig(PATH_FIGS * "g2_N" * string(N) * "_" * "phi_RL_bright" * ".pdf", dpi=300)
end

let
    fig_01, ax = plt.subplots(1, 1, subplot_kw=Dict("projection" => "polar"), figsize=(5,5))
    ax.plot(phi_var, n_result[1, :], color="red", linewidth=2, label="f")
    ax.plot(phi_var, n_result[2, :], color="blue", linewidth=2, label="b")
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(true)
    ax.set_title(L"N_{ph}", va="bottom")
    ax.legend()
    display(fig_01)

    # fig_01.savefig(PATH_FIGS * "numPh_N" * string(N) * "_" * "phi_RL_bright" * ".pdf", dpi=300)
end

"g2 depending on both angles"

g2_result_2D = zeros((NMAX ÷ 2, NMAX))

Threads.@threads for iijj in CartesianIndices((NMAX ÷ 2, NMAX))
    (ii, jj) = Tuple(iijj)[1], Tuple(iijj)[2]
    θ = theta_var[ii]
    ϕ = phi_var[jj]
    D = AtomicArraysStatistics.jump_op_direct_detection(θ, ϕ, 0.02^2*pi^2, S, 2π, J)
    g2 = AtomicArraysStatistics.coherence_function_g2(tau_0, H, J, D; rates=Γ, rho0=ρ_ss_2)
    g2_result_2D[ii, jj] = real(g2[1])
    print(ii, "\n")
end
# Plots
let
    lθ = length(theta_var);
    lϕ = length(phi_var);

    X = zeros(lθ,lϕ);
    Y = zeros(lθ,lϕ);
    Z = zeros(lθ,lϕ);

    for ii=1:lθ
        for jj=1:lϕ
            X[ii,jj]= g2_result_2D[ii,jj]*cos(phi_var[jj])*sin(theta_var[ii]);
            Y[ii,jj]= g2_result_2D[ii,jj]*sin(phi_var[jj])*sin(theta_var[ii]);
            Z[ii,jj]= g2_result_2D[ii,jj]*cos(theta_var[ii]);
        end
    end
    x_at = [pos[i][1] for i = 1:N]
    y_at = [pos[i][2] for i = 1:N]
    z_at = [pos[i][3] for i = 1:N]

    GLMakie.activate!()
    fig, ax, pltobj = surface(X, Y, Z; shading = true, ambient = Vec3f(0.65, 0.65, 0.65),
        backlight = 1.0f0, color = sqrt.(X .^ 2 .+ Y .^ 2 .+ Z .^ 2),
        colormap = :viridis, transparency = true,
        figure = (; resolution = (1200, 800), fontsize = 22),
        axis=(type=Axis3, aspect = :data, azimuth = 2*pi/12, elevation = pi/30))
    meshscatter!(ax, x_at, y_at, z_at; color = "black",
                markersize = 0.05)
    wireframe!(X, Y, Z; overdraw = true, transparency = true, color = (:black, 0.1))
    wireframe!(X, Y*0 .+ minimum(Y), Z; transparency = true, color = (:grey, 0.1))
    Colorbar(fig[1, 2], pltobj, height = Relative(0.5))
    colsize!(fig.layout, 1, Aspect(1, 1.0))
    fig

    save((PATH_FIGS * "g2_N" * string(N) * "_" * "theta_phi_dark_" * DIRECTION * ".png"), fig) # here, you save your figure.
end
