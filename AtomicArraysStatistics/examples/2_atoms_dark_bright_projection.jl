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
    using CSV, DataFrames

    using AtomicArraysStatistics

    import EllipsisNotation: Ellipsis
    const .. = Ellipsis()
end

begin
    const EMField = field.EMField
    # const em_inc_function = AtomicArrays.field.gauss
    const em_inc_function = AtomicArrays.field.plane
    NMAX = 50
    NMAX_T = 5

    const PATH_FIGS, PATH_DATA = AtomicArraysStatistics.path()

    # System parameters
    γ = 1.
    e_dipole = [1., 0, 0]
    T = [0:0.05:500;]
    N = 2
    Ncenter = 1
    DETUNING_SYMMETRY = true
    GEOMETRY = "chain"

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
    d_list = range(1.5e-1, 10e-1, NMAX);

    projections_D = zeros(2, NMAX, NMAX, NMAX);
    projections_B = zeros(2, NMAX, NMAX, NMAX);
    "done"
end

progress = Progress(2 * NMAX * NMAX * NMAX)
for kkiijjmm in CartesianIndices((2, NMAX, NMAX, NMAX))
    (kk, ii, jj, mm) = (Tuple(kkiijjmm)[1], Tuple(kkiijjmm)[2], 
    Tuple(kkiijjmm)[3], Tuple(kkiijjmm)[4])

    DIRECTION = dir_list[kk]
    Delt_0 = Delt_list[ii]
    E_ampl = E_list[jj]
    a = d_list[mm]

    # Atoms

    if GEOMETRY == "chain"
        pos = geometry.chain_dir(a, N; dir="z", pos_0=[0, 0, -a / 2])
    end
    if DETUNING_SYMMETRY
        Delt = [(i < N) ? - Delt_0 / 2 : Delt_0 / 2 for i = 1:N]
    else
        Delt = [(i < N) ? 0.0 : Delt_0 for i = 1:N]
    end
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
    # ψ_D = 1.0/sqrt(2.0) * (Ket(basis(H), [0,1,0,0]) - 
    #                        sign(Γ[1,2])*Ket(basis(H), [0,0,1,0]))
    ψ_D = 1.0/sqrt(2.0) * (Ket(basis(H), [0,1,0,0]) - 
                           Ket(basis(H), [0,0,1,0]))
    ρ_D = ψ_D ⊗ dagger(ψ_D)
    # ψ_B = 1.0/sqrt(2.0) * (Ket(basis(H), [0,1,0,0]) + 
    #                        sign(Γ[1,2])*Ket(basis(H), [0,0,1,0]))
    ψ_B = 1.0/sqrt(2.0) * (Ket(basis(H), [0,1,0,0]) + 
                           Ket(basis(H), [0,0,1,0]))
    ρ_B = ψ_B ⊗ dagger(ψ_B)

    # Steady-state
    ρ_ss = QuantumOptics.steadystate.eigenvector(H, J; rates=Γ)

    # phi = 0.
    # theta = pi/1.
    # Ψ₀ = AtomicArrays.quantum.blochstate(phi,theta,N)
    # ρ₀ = Ψ₀⊗dagger(Ψ₀)
    # _, ρ_t = QuantumOptics.timeevolution.master_h(T, ρ₀, H, J; rates=Γ)
    # ρ_ss = ρ_t[end]

    # Projections
    projections_D[kk, ii, jj, mm] = real(tr(ρ_ss * ρ_D))
    projections_B[kk, ii, jj, mm] = real(tr(ρ_ss * ρ_B))

    next!(progress)
end

maximum(projections_D[1,..])

maximum(projections_B[1,..])

begin
    eff_D = AtomicArrays.field.objective(projections_D[1,..], projections_D[2,..]);
    eff_B = AtomicArrays.field.objective(projections_B[1,..], projections_B[2,..]);
    (maximum(eff_D), argmax(eff_D), (Delt_list[argmax(eff_D)[1]], E_list[argmax(eff_D)[2]], d_list[argmax(eff_D)[3]]))
end

begin
    # Position of the maximum
    idx_RD = argmax(projections_D[1,..])
    idx_LD = argmax(projections_D[2,..])
    ((projections_D[1, idx_RD], projections_B[1, idx_RD]),
    (projections_D[2, idx_LD], projections_B[2, idx_LD]), 
    (Delt_list[idx_RD[1]], E_list[idx_RD[2]], d_list[idx_RD[3]]),
    (Delt_list[idx_LD[1]], E_list[idx_LD[2]], d_list[idx_LD[3]]))
end

begin 
    idx_RB = argmax(projections_B[1,..])
    idx_LB = argmax(projections_B[2,..])
    ((projections_B[1, idx_RB], projections_D[1, idx_RB]), 
    (projections_B[2, idx_LB], projections_D[2, idx_LB]),
    (Delt_list[idx_RB[1]], E_list[idx_RB[2]], d_list[idx_RB[3]]),
    (Delt_list[idx_LB[1]], E_list[idx_LB[2]], d_list[idx_LB[3]]))
end

begin
    # max contrast position
    idx_D = argmax(eff_D)
    idx_B = argmax(eff_B)
    ((projections_D[1,idx_D], projections_D[2,idx_D],projections_B[1,idx_D], projections_B[2,idx_D]),
    (Delt_list[idx_D[1]], E_list[idx_D[2]], d_list[idx_D[3]]),
    (projections_D[1,idx_B], projections_D[2,idx_B],projections_B[1,idx_B], projections_B[2,idx_B]),
    (Delt_list[idx_B[1]], E_list[idx_B[2]], d_list[idx_B[3]]))
end

let
    idxD = idx_D[2]
    idxB = idx_B[2]

    x = Delt_list
    y = d_list
    c_D_1 = projections_D[1, :, idxD, :]
    c_D_2 = projections_D[2, :, idxD, :]
    c_B_1 = projections_B[1, :, idxB, :]
    c_B_2 = projections_B[2, :, idxB, :]
    cmap = "viridis"

    id_max_D_R = argmax(projections_D[1, :, idxD, :])
    id_max_D_L = argmax(projections_D[2, :, idxD, :])
    id_max_B_R = argmax(projections_B[1, :, idxB, :])
    id_max_B_L = argmax(projections_B[2, :, idxB, :])

    fig, ax = plt.subplots(2, 2, figsize=(8,6), constrained_layout=true)
    im1 = ax[1,1].pcolormesh(x, y, c_D_1', cmap=cmap)
    ax[1,1].scatter([x[id_max_D_R[1]]], [y[id_max_D_R[2]]], c="r")
    ax[1,1].set_title(L"D,R")
    ax[1,1].set_xlabel(L"\Delta")
    ax[1,1].set_ylabel(L"a")
    # ax[1].set_ylim(0.42,0.46)
    fig.colorbar(im1, ax=ax[1,1])
    im2 = ax[1,2].pcolormesh(x, y, c_D_2', cmap=cmap)
    ax[1,2].scatter([x[id_max_D_L[1]]], [y[id_max_D_L[2]]], c="b")
    ax[1,2].set_title(L"D,L")
    ax[1,2].set_xlabel(L"\Delta")
    fig.colorbar(im2, ax=ax[1,2])

    im3 = ax[2,1].pcolormesh(x, y, c_B_1', cmap=cmap)
    ax[2,1].scatter([x[id_max_B_R[1]]], [y[id_max_B_R[2]]], c="r")
    ax[2,1].set_title(L"B,R")
    ax[2,1].set_xlabel(L"\Delta")
    ax[2,1].set_ylabel(L"a")
    # ax[1].set_ylim(0.42,0.46)
    fig.colorbar(im3, ax=ax[2,1])
    im4 = ax[2,2].pcolormesh(x, y, c_B_2', cmap=cmap)
    ax[2,2].scatter([x[id_max_B_L[1]]], [y[id_max_B_L[2]]], c="b")
    ax[2,2].set_title(L"B,L")
    ax[2,2].set_xlabel(L"\Delta")
    fig.colorbar(im4, ax=ax[2,2])
    fig
end

let
    idxD = idx_D[2]
    idxB = idx_B[2]
    x = Delt_list
    y = d_list
    c_D_1 = eff_D[:, idxD, :]
    c_D_2 = eff_B[:, idxB, :]
    cmap = "viridis"

    id_max_D = argmax(eff_D[:, idxD, :])
    id_max_B = argmax(eff_B[:, idxB, :])

    fig, ax = plt.subplots(1, 2, figsize=(9,3))
    im1 = ax[1].pcolormesh(x, y, c_D_1', cmap=cmap, 
                        # vmin=0, vmax=0.2
                        )
    ax[1].scatter([x[id_max_D[1]]], [y[id_max_D[2]]], c="r")
    ax[1].set_title("Contrast for D population")
    ax[1].set_xlabel(L"\Delta")
    ax[1].set_ylabel(L"a")
    fig.colorbar(im1, ax=ax[1])

    im2 = ax[2].pcolormesh(x, y, c_D_2', cmap=cmap,
                          # vmin=0, vmax=0.2
                          )
    ax[2].scatter([x[id_max_B[1]]], [y[id_max_B[2]]], c="r")
    ax[2].set_title("Contrast for B population")
    ax[2].set_xlabel(L"\Delta")
    fig.colorbar(im2, ax=ax[2])
    fig
end



# Define the file path
csv_file = PATH_DATA*"experiment_results_N"*string(N)*".csv"

# Function to check if a row with the same values exists in the DataFrame
function row_exists(df, params)
    for i in 1:nrow(df)
        if all(df[i, key] == value for (key, value) in params)
            return true
        end
    end
    return false
end

# Function to add or update experiment results to the CSV file
function add_experiment_results(params)
    df = CSV.read(csv_file, DataFrame)
    if !row_exists(df, params)
        append!(df, DataFrame(params))
        CSV.write(csv_file, df)
    else
        println("Row with the same values already exists.")
    end
end


directions = ["R", "L", "E"]
states = ["max_D", "max_B"]

for st in states
    if st == "max_D"
        arr_Dir = projections_D
        arr_eff = eff_D
    elseif st == "max_B"
        arr_Dir = projections_B
        arr_eff = eff_B
    end
    for dir in directions
        if dir == "R"
            idx = argmax(arr_Dir[1, ..])
        elseif dir == "L"
            idx = argmax(arr_Dir[2, ..])
        elseif dir == "E"
            idx = argmax(arr_eff)
        end

        params_to_write = Dict("State" => st, "N"=>N,
                       "a"=>round(d_list[idx[3]], digits=3), 
                       "E₀"=>round(E_list[idx[2]], digits=3),
                       "Direction"=>dir, 
                       "detuning_symmetry"=>DETUNING_SYMMETRY, 
                       "geometry"=>GEOMETRY,
                       "proj_B_L" => ((dir == "R") ? 0.0 : 
                            round(projections_B[2,idx], digits=3)),
                       "proj_B_R" => ((dir == "L") ? 0.0 : 
                            round(projections_B[1,idx], digits=3)), 
                       "proj_D_L" => ((dir == "R") ? 0.0 : 
                            round(projections_D[2,idx], digits=3)),
                       "proj_D_R" => ((dir == "L") ? 0.0 : 
                            round(projections_D[1,idx], digits=3)), 
                       )
        for i = 1:N
            if (DETUNING_SYMMETRY) 
                params_to_write["Δ_$i"] = round(0.5*(-1.0)^i*Delt_list[idx[1]],
                                                digits=3)
            else
                params_to_write["Δ_$i"] = round((Float64(i)-1.0)*Delt_list[idx[1]],
                                                digits=3)
            end
        end

        # Check if the CSV file exists, and create it with headers if it doesn't
        if !isfile(csv_file)
            headers = ["State", "N", "detuning_symmetry", "geometry", 
                    "Direction", "a", "E₀"]
            for i in 1:N
                push!(headers, "Δ_$i")
            end  
            append!(headers, ["proj_B_L", "proj_B_R", "proj_D_L", "proj_D_R"]) 

            df = DataFrame([[] for _ = headers] , headers)
            append!(df, DataFrame(params_to_write))
            CSV.write(csv_file, df)
        end
        # Example usage: add experiment parameters for a specific configuration
        add_experiment_results(params_to_write)
    end
end

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