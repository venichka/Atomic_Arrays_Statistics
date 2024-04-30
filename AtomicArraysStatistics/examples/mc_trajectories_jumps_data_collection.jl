# Monte-Carlo trajectories: data collection

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
    using Statistics, StatsBase
    using QuantumOptics
    using AtomicArrays
    using Revise
    using BenchmarkTools
    using ProgressMeter, Suppressor
    using HDF5, FileIO, Printf

    using AtomicArraysStatistics
end


function idx_2D_to_1D(i, j, nmax)
    return (i - 1) * nmax + j
end

function idx_1D_to_2D(i, nmax)
    return CartesianIndex((i - 1) ÷ nmax + 1, (i - 1) % nmax + 1)
end


# parameters for computing
begin
    const EMField = field.EMField
    # const em_inc_function = AtomicArrays.field.gauss
    const em_inc_function = AtomicArrays.field.plane
    NMAX = 10
    N_traj = 10
    NMAX_T = 5
    N_BINS = 1000
    DIRECTION = "R"
    tau_max = 5e5

    const PATH_FIGS, PATH_DATA = AtomicArraysStatistics.path()
end

# System parameters
begin
    const a = 0.21
    const γ = 1.0
    const e_dipole = [1.0, 0, 0]
    const T = [0:0.05:500;] # for computing steady-states
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


"System Hamiltonian"

begin
    Γ, J = AtomicArrays.quantum.JumpOperators(S)
    Jdagger = dagger.(J)
    Ω = AtomicArrays.interaction.OmegaMatrix(S)
    H = AtomicArrays.quantum.Hamiltonian(S) - sum(Om_R[j] * J[j] +
                                                conj(Om_R[j]) * Jdagger[j]
                                                for j = 1:N)
    # eigen(dense(H).data)
    w, v = eigenstates(dense(H))

    # Jump operators description
    J_s = AtomicArraysStatistics.jump_op_source_mode(Γ, J)
end

"Waiting time distributions and g2 functions"

"WTD for J operators"

begin
    # Initial state (Bloch state)
    phi = 0.0
    theta = pi / 1.0
    psi0 = AtomicArrays.quantum.blochstate(phi, theta, N)

    # steady-state
    tout, psi_t = timeevolution.mcwf(T, psi0, H, J_s)
    psi_ss = psi_t[end]

    # jumps computation
    T_wtau = [0:tau_max/100:tau_max;]
    jump_t = Float64[]
    jump_i = Int[]
    lk = ReentrantLock()

    function compute_jumps(T, psi_ss, H, J_s, jump_t, jump_i, progress)
        _, _, jump_t_0, jump_i_0 = timeevolution.mcwf(T, psi_ss, H, J_s;
                                                    display_jumps=true, maxiters=1e15)
        lock(lk) do
            append!(jump_t, jump_t_0)
            append!(jump_i, jump_i_0)
        end
        next!(progress)
    end

    progress = Progress(N_traj)
    Threads.@threads for _=1:N_traj
        compute_jumps(T_wtau, psi_ss, H, J_s, jump_t, jump_i, progress)
    end

    finish!(progress)
    println("J operators jumps computed")

end

"WTD for S(θ, ϕ) operators"

begin
    d_angle = 2.0 / NMAX * pi 

    phi_var = [(i-0.5)*d_angle for i = 1:NMAX]
    theta_var = [(i-0.5)*d_angle for i = 1:NMAX ÷ 2]
    # dΩ = sin(θ) dθ dϕ
    dΩ = [d_angle * d_angle * sin(theta_var[i]) for i = 1:NMAX ÷ 2, j = 1:NMAX]

    # Computing steady states
    _, psi_t_S = timeevolution.mcwf(T, psi0, H, J_s)
    psi_ss_S = psi_t_S[end]

    D = [AtomicArraysStatistics.jump_op_direct_detection(phi_var[(i-1) % NMAX + 1], theta_var[(i-1) ÷ NMAX + 1], dΩ[(i-1) ÷ NMAX + 1, (i-1) % NMAX + 1], S, 2π, J) for i = 1:NMAX*(NMAX ÷ 2)]
end

# Time evolution
begin
    T_wtau = [0:tau_max/100:tau_max;]
    jump_t_S = Float64[]
    jump_i_S = Int[]
    lk = ReentrantLock()

    progress = Progress(N_traj)
    Threads.@threads for _=1:N_traj
        compute_jumps(T_wtau, psi_ss_S, H, D, jump_t_S, jump_i_S, progress)
    end
    finish!(progress)
    println("S operators jumps computed")
end;


"""Writing DATA"""

begin
    data_dict_jumps = Dict("theta" => collect(theta_var),
                            "phi" => collect(phi_var), 
                            "jump_times_S" => jump_t_S,
                            "jump_op_idx_S" => jump_i_S,
                            "jump_times" => jump_t,
                            "jump_op_idx" => jump_i
                            )

    time_str = @sprintf "%.0E" tau_max*N_traj
    NAME_PART = string(N)*"atoms_tmax"*time_str*"_nmax"*string(NMAX)*"_"*DIRECTION*".h5"
    save(PATH_DATA*"jumps_"*NAME_PART, data_dict_jumps)

    data_dict_loaded = load(PATH_DATA*"jumps_"*NAME_PART)
    if data_dict_loaded["jump_times"] == data_dict_jumps["jump_times"]
        println("Data written successfully")
    else
        println("Couldn't write data")
    end
end