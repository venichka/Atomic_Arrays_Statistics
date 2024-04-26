# Monte-Carlo trajectories

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
using Statistics, StatsBase
using QuantumOptics
using PyCall
PyCall.pygui(:qt5)
using PyPlot
using GLMakie
using WGLMakie
using AtomicArrays
using Revise
using BenchmarkTools
using ProgressMeter, Suppressor
using HDF5, FileIO, Printf

using AtomicArraysStatistics

const EMField = field.EMField
# const em_inc_function = AtomicArrays.field.gauss
const em_inc_function = AtomicArrays.field.plane
const NMAX = 30
const N_traj = 100
const NMAX_T = 5
const N_BINS = 1000
const DIRECTION = "L"
tau_max = 5e5

const PATH_FIGS, PATH_DATA = AtomicArraysStatistics.path()

function idx_2D_to_1D(i, j, nmax)
    return (i - 1) * nmax + j
end

function idx_1D_to_2D(i, nmax)
    return CartesianIndex((i - 1) ÷ nmax + 1, (i - 1) % nmax + 1)
end

# System parameters
begin
    const a = 0.21
    const γ = 1.0
    const e_dipole = [1.0, 0, 0]
    const T = [0:0.05:500;]
    const N = 2
    const Ncenter = 1

    const pos = geometry.chain_dir(a, N; dir="z", pos_0=[0, 0, -a / 2])
    const Delt = [(i < N) ? -1.184/2 : 1.184/2 for i = 1:N]
    const S = SpinCollection(pos, e_dipole; gammas=γ, deltas=Delt)

    # Define Spin 1/2 operators
    spinbasis = SpinBasis(1 // 2)
    sx = sigmax(spinbasis)
    sy = sigmay(spinbasis)
    sz = sigmaz(spinbasis)
    sp = sigmap(spinbasis)
    sm = sigmam(spinbasis)
    I_spin = identityoperator(spinbasis)

    # Incident field
    E_ampl = 0.497 + 0im
    E_kvec = 2π
    if (DIRECTION == "R")
        E_pos0 = [0.0, 0.0, 0.0]
        E_polar = [1.0, 0im, 0.0]
        E_angle = [0.0, 0.0]  # {θ, φ}
    elseif (DIRECTION == "L")
        E_pos0 = [0.0, 0.0, 0.0 * a]
        E_polar = [-1.0, 0im, 0.0]
        E_angle = [π, 0.0]  # {θ, φ}
    else
        println("DIRECTION wasn't specified")
    end

    E_inc = EMField(E_ampl, E_kvec, E_angle, E_polar;
        position_0=E_pos0, waist_radius=0.1)

    # Field-spin interaction
    E_vec = [em_inc_function(S.spins[k].position, E_inc) for k = 1:N]
    Om_R = field.rabi(E_vec, S.polarizations)

end

"""
load_jump_data()

Loads a dict with data according to the parameters specified in the workspace.

* Output:
- (theta_var, phi_var, jump_t, jump_i)
"""
function load_jump_data()
    time_str = @sprintf "%.0E" tau_max*N_traj
    NAME_PART = string(N)*"atoms_tmax"*time_str*"_nmax"*string(NMAX)*"_"*DIRECTION*".h5"

    data_dict_loaded = load(PATH_DATA*"jumps_"*NAME_PART)
    return data_dict_loaded["theta"],  data_dict_loaded["phi"], data_dict_loaded["jump_times"], data_dict_loaded["jump_op_idx"]
end


"""Impinging field"""

begin
    x = range(-3.0, 3.0, NMAX)
    y = 0.0
    z = range(-5.0, 5.0, NMAX)
    e_field = Matrix{ComplexF64}(undef, length(x), length(z))
    for i in eachindex(x)
        for j in eachindex(z)
            e_field[i, j] = em_inc_function([x[i], y, z[j]], E_inc)[1]
        end
    end
    # Plot
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

Γ, J = AtomicArrays.quantum.JumpOperators(S)
Jdagger = dagger.(J)
Ω = AtomicArrays.interaction.OmegaMatrix(S)
H = AtomicArrays.quantum.Hamiltonian(S) - sum(Om_R[j] * J[j] +
                                              conj(Om_R[j]) * Jdagger[j]
                                              for j = 1:N)
H.data

# eigen(dense(H).data)
w, v = eigenstates(dense(H))

# Jump operators description
J_s = AtomicArraysStatistics.jump_op_source_mode(Γ, J)
# d, D_op = diagonaljumps(Γ, J)

"Dynamics"

# Initial state (Bloch state)
const phi = 0.0
const theta = pi / 1.0

embed(op::Operator,i) = QuantumOptics.embed(AtomicArrays.quantum.basis(S), i, op)

sx_av = zeros(Float64, N, length(T))
sy_av = zeros(Float64, N, length(T))
sz_av = zeros(Float64, N, length(T))
function fout(t::Float64, psi::Ket)
    j = findfirst(isequal(t), T)
    for i = 1:N
        sx_av[i, j] += real(expect(embed(sx, i), psi) / norm(psi)^2)
        sy_av[i, j] += real(expect(embed(sy, i), psi) / norm(psi)^2)
        sz_av[i, j] += real(expect(embed(sz, i), psi) / norm(psi)^2)
    end
    return nothing
end

# Time evolution
psi0 = AtomicArrays.quantum.blochstate(phi, theta, N)
@suppress_err Threads.@threads for i=1:N_traj
    timeevolution.mcwf(T, psi0, H, J_s; fout=fout)
end
sx_av ./= N_traj
sy_av ./= N_traj
sz_av ./= N_traj


"Waiting time distributions and g2 functions"

"WTD for J operators"

tout, psi_t = timeevolution.mcwf(T, psi0, H, J_s)
psi_ss = psi_t[end]
tout, psi_t, jump_t, jump_i = timeevolution.mcwf([0:500.0:5e6;], psi_ss, H, J_s; display_jumps=true, maxiters=1e15)

length(jump_t)
length(jump_i)

w_tau = [jump_t[j+1] - jump_t[j] for j in 1:(length(jump_t) - 1)]
begin  # compute WTD by the angle
    w_tau_n = []
    idx_no_stat = []
    for i = 1:N
        jumps = jump_t[jump_i .== i]
        jumps_dist = [jumps[j+1] - jumps[j] 
                    for j in 1:(length(jumps) - 1)]
        jumps_dist = jumps_dist[jumps_dist .>= 0]
        if isempty(jumps_dist)
            append!(idx_no_stat, i)
            print(i, " ")
        end
        push!(w_tau_n, jumps_dist)
    end
    print("w_tau_n computed.\n")
end
w_tau_1 = w_tau_n[1]
w_tau_2 = w_tau_n[2]
w_tau_av = mean(w_tau)
w_tau_1_av = mean(w_tau_1)
w_tau_2_av = mean(w_tau_2)

"WTD for S(θ, ϕ) operators"

d_angle = 2.0 / NMAX * pi 

phi_var = [(i-0.5)*d_angle for i = 1:NMAX]
theta_var = [(i-0.5)*d_angle for i = 1:NMAX ÷ 2]
# dΩ = sin(θ) dθ dϕ
dΩ = [d_angle * d_angle * sin(theta_var[i]) for i = 1:NMAX ÷ 2, j = 1:NMAX]

# Computing steady states
_, psi_t_S = timeevolution.mcwf(T, psi0, H, J_s)
psi_ss_S = psi_t_S[end]

D = [AtomicArraysStatistics.jump_op_direct_detection(phi_var[(i-1) % NMAX + 1], theta_var[(i-1) ÷ NMAX + 1], dΩ[(i-1) ÷ NMAX + 1, (i-1) % NMAX + 1], S, 2π, J) for i = 1:NMAX*(NMAX ÷ 2)]

begin
tout_S, psi_t, jump_t_S, jump_i_S = timeevolution.mcwf([0:5e4:5e6;], psi_ss_S, H, D; display_jumps=true, maxiters=1e15)
end

# Time evolution
begin
    tau_max = 5e5
    T_wtau = [0:tau_max/100:tau_max;]
    _, _, jump_t_S, jump_i_S = timeevolution.mcwf(T_wtau, psi_ss_S, H, D; 
                                    display_jumps=true, maxiters=1e15);
    lk = ReentrantLock()
    progress = Progress(N_traj)
    Threads.@threads for i=1:N_traj
        _, _, jump_t_S_0, jump_i_S_0 = timeevolution.mcwf(T_wtau, psi_ss_S, 
                                H, D; display_jumps=true, maxiters=1e15);
        lock(lk) do
            append!(jump_t_S, jump_t_S_0)
            append!(jump_i_S, jump_i_S_0)
        end
        next!(progress)
    end
    finish!(progress)
end;

length(jump_t_S)
length(jump_i_S)


w_tau_S = [jump_t_S[j+1] - jump_t_S[j] for j in 1:(length(jump_t_S) - 1)]
w_tau_S = w_tau_S[w_tau_S .>= 0]
begin  # compute WTD by the angle
    w_tau_S_n = []
    idx_no_stat = []
    for i = 1:NMAX*(NMAX ÷ 2)
        jumps = jump_t_S[jump_i_S .== i]
        jumps_dist = [jumps[j+1] - jumps[j] 
                    for j in 1:(length(jumps) - 1)]
        jumps_dist = jumps_dist[jumps_dist .>= 0]
        if isempty(jumps_dist)
            append!(idx_no_stat, i)
            print(i, " ")
        end
        push!(w_tau_S_n, jumps_dist)
    end
    print("w_tau_S_n computed.\n")
end

begin # angle distribution (note that N_BINS for StatsBase should be x2 
      # in comparison with Matplotlib)
    w_angle_0 = zeros(Float64, NMAX ÷ 2, NMAX)
    for i = 1:NMAX*(NMAX ÷ 2)
        if isempty(w_tau_S_n[i])
            w_angle_0[(i-1) ÷ NMAX + 1, (i-1) % NMAX + 1] = 0.0
        else
            h_0 = StatsBase.fit(Histogram, w_tau_S_n[i] ./ mean(w_tau_S_n[i]), 
                                nbins=N_BINS)
            h_0_norm = normalize(h_0, mode=:pdf)
            w_angle_0[(i-1) ÷ NMAX + 1, (i-1) % NMAX + 1] = h_0_norm.weights[1]
        end
        print(i, "\n")
    end
end


begin
    idx = 25
    h = fit(Histogram, w_tau_S_n[idx] ./ mean(w_tau_S_n[idx]), nbins=N_BINS)
    h_0 = normalize(h, mode=:pdf)
    
    fig_03, axs = plt.subplots(1, 1, sharey=true, tight_layout=true, figsize=(6,3))
    # axs.plot(test[1:1000])
    n, bins, patches = axs.hist(w_tau_S ./ mean(w_tau_S), bins=N_BINS, density=true, histtype="bar", label=L"w(\tau / \bar{\tau})")
    axs.hist(w_tau_S_n[idx] ./ mean(w_tau_S_n[idx]), bins=N_BINS, density=true, alpha=0.5, histtype="bar", label=L"w_n(\tau / \bar{\tau})")
    axs.plot(bins[1:end-1], γ*exp.(-γ*bins[1:end-1]), color="red", label=L"\gamma \exp(-\gamma \tau / \bar{\tau})")
    axs.plot(h_0.edges[1][1:end-1], h_0.weights, color="blue")
    axs.set_xlim((0, 5))
    # axs.set_ylim((0, 2))
    # axs.set_yscale("log")
    axs.set_xlabel(L"\tau / \bar{\tau}")
    axs.set_ylabel(L"w(\tau / \bar{\tau}), g^{2}(\tau / \bar{\tau})")
    axs.legend(loc="upper right")
    display(fig_03)
    # fig_03.savefig(PATH_FIGS * "wtau_g2_N" * string(N) * "_" * "jump_" * DIRECTION * ".pdf", dpi=300)
end

for i=1:NMAX*NMAX ÷2
    print(length(w_tau_S_n[i]), "\n")
end

begin
    fig_04, axs = plt.subplots(1, 1, sharey=true, tight_layout=true, figsize=(6,6))
    n, bins, patches = axs.hist(w_tau_S ./ mean(w_tau_S), bins=N_BINS, density=true, histtype="bar", label=L"w(\tau / \bar{\tau})")
    for i = 1:NMAX * (NMAX ÷ 2)
        axs.hist(w_tau_S_n[i] ./ mean(w_tau_S_n[i]), bins=N_BINS, density=true, alpha=0.1, histtype="bar")
    end
    axs.plot(bins[1:end-1], γ*exp.(-γ*bins[1:end-1]), color="red", label=L"\gamma \exp(-\gamma \tau / \bar{\tau})")
    axs.set_xlim((0, 1))
    # axs.set_ylim((0, 2))
    # axs.set_yscale("log")
    axs.set_xlabel(L"\tau / \bar{\tau}")
    axs.set_ylabel(L"w(\tau / \bar{\tau}), g^{2}(\tau / \bar{\tau})")
    axs.legend(loc="upper right")
    display(fig_04)
    # fig_03.savefig(PATH_FIGS * "wtau_g2_N" * string(N) * "_" * "jump_" * DIRECTION * ".pdf", dpi=300)
end

begin
    lθ = length(theta_var);
    lϕ = length(phi_var);

    X = zeros(lθ,lϕ);
    Y = zeros(lθ,lϕ);
    Z = zeros(lθ,lϕ);

    for ii=1:lθ
        for jj=1:lϕ
            X[ii,jj] = w_angle_0[ii,jj]*cos(phi_var[jj])*sin(theta_var[ii]);
            Y[ii,jj] = w_angle_0[ii,jj]*sin(phi_var[jj])*sin(theta_var[ii]);
            Z[ii,jj] = w_angle_0[ii,jj]*cos(theta_var[ii]);
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
        axis=(type=Axis3, aspect = :data, 
              azimuth = 2*pi/12, elevation = pi/30,
            #   limits=(-2,2,-2,2,-2,2)
              ))
    meshscatter!(ax, x_at, y_at, z_at; color = "black",
                markersize = 0.05)
    wireframe!(X, Y, Z; overdraw = true, transparency = true, color = (:black, 0.1))
    wireframe!(X, Y*0 .+ minimum(Y), Z; transparency = true, color = (:grey, 0.1))
    Colorbar(fig[1, 2], pltobj, height = Relative(0.5))
    colsize!(fig.layout, 1, Aspect(1, 1.0))
    fig
end


"""Writing DATA"""


# fig_1.savefig(PATH_FIGS*"obj4D_lattice_4x4_mf_nit10.png", dpi=300)

data_dict_jumps_S = Dict("theta" => collect(theta_var), "phi" => collect(phi_var), 
                     "jump_times" => jump_t_S, "jump_op_idx" => jump_i_S)

time_str = @sprintf "%.0E" tau_max*N_traj
NAME_PART = string(N)*"atoms_tmax"*time_str*"_nmax"*string(NMAX)*"_"*DIRECTION*".h5"
save(PATH_DATA*"jumps_"*NAME_PART, data_dict_jumps_S)

data_dict_loaded = load(PATH_DATA*"jumps_"*NAME_PART)
data_dict_loaded["jump_times"] == data_dict_jumps_S["jump_times"]



"g2 function"

tau_0 = [0:0.5:1000;]
ρ_ss = QuantumOptics.steadystate.eigenvector(H, J; rates=Γ)  # finding the steady-state
g2 = AtomicArraysStatistics.coherence_function_g2(tau_0, H, J_s, J_s[1] + J_s[2]; rho0=ρ_ss)
g2_1 = AtomicArraysStatistics.coherence_function_g2(tau_0, H, J_s, J_s[1]; rho0=ρ_ss)
g2_2 = AtomicArraysStatistics.coherence_function_g2(tau_0, H, J_s, J_s[2]; rho0=ρ_ss)

A = dagger(J_s[2])
B = dagger(J_s[1]) * J_s[1]
C = J_s[2]
n_ss_1 = QuantumOptics.expect(B, ρ_ss)
n_ss_2 = QuantumOptics.expect(A*C, ρ_ss)
G2 = AtomicArraysStatistics.correlation_3op_1t(tau_0, ρ_ss, H, J, A, B, C;
                                               rates=Γ)
g2 = G2 / (n_ss_1 * n_ss_2)

begin
    const N_BINS = 500
    fig_02, axs = plt.subplots(1, 1, sharey=true, tight_layout=true, figsize=(6,3))
    n, bins, patches = axs.hist(w_tau / w_tau_av, bins=N_BINS, density=true, histtype="bar", label=L"w(\tau / \bar{\tau})")
    axs.plot(bins[1:end-1], n)
    axs.hist(w_tau_2 / w_tau_2_av, bins=N_BINS, alpha=0.3, density=true, histtype="bar", label=L"w_1(\tau / \bar{\tau})")
    axs.hist(w_tau_1 / w_tau_1_av, bins=N_BINS, alpha=0.3, density=true, histtype="bar", label=L"w_2(\tau / \bar{\tau})")
    axs.plot(bins[1:end-1], γ*exp.(-γ*bins[1:end-1]), color="red", label=L"\gamma \exp(-\gamma \tau / \bar{\tau})")
    axs.plot(tau_0/w_tau_1_av, g2_1, color="blue", label=L"g^{(2)}_1(\tau / \bar{\tau})")
    axs.plot(tau_0/w_tau_2_av, g2_2, "--", color="blue", label=L"g^{(2)}_2(\tau / \bar{\tau})")
    axs.plot(tau_0/w_tau_av, g2, ":", color="blue", label=L"g^{(2)}(\tau / \bar{\tau})")
    axs.set_xlim((0, 5))
    axs.set_ylim((0, 1))
    axs.set_xlabel(L"\tau / \bar{\tau}")
    axs.set_ylabel(L"w(\tau / \bar{\tau}), g^{2}(\tau / \bar{\tau})")
    axs.legend(loc="upper right")
    display(fig_02)
    # fig_02.savefig(PATH_FIGS * "wtau_g2_N" * string(N) * "_" * "jump_" * DIRECTION * ".pdf", dpi=300)
end



fig_01, ax = plt.subplots(3, 1, figsize=(6,9), constrained_layout = true)
for i = 1:N
    ax[1].plot(T, sx_av[i, :], label="atom "*string(i))
end
ax[1].grid(true)
ax[1].set_ylabel(L"\sigma_x")
ax[1].legend()
for i = 1:N
    ax[2].plot(T, sy_av[i, :], label="atom "*string(i))
end
ax[2].grid(true)
ax[2].set_ylabel(L"\sigma_y")
ax[2].legend()
for i = 1:N
    ax[3].plot(T, sz_av[i, :], label="atom "*string(i))
end
ax[3].grid(true)
ax[3].set_ylabel(L"\sigma_z")
ax[3].legend()
ax[3].set_xlabel(L"t")
display(fig_01)
