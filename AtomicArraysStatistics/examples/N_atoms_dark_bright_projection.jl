# 2 atoms: dark and bright state projections
# TODO: rewrite the file for N atoms, write a function computing D and B states
# TODO: fix for N = 3 -- find dark/bright states and edit the file
# TODO: decrease γ and Δ and E_0 by an order of magnitude


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

function detuning_generator_sym(N::Int, Delt_0::Real)
    Ncenter = Int(ceil(N / 2))
    Delt = [(i <= Ncenter) ? - Delt_0 / 2 : Delt_0 / 2 for i = 1:N]
    Delt[Ncenter] = (mod(N, 2) != 0) ? 0.0 : Delt[Ncenter]
    return Delt
end

function detuning_generator_nonsym_1atom(N::Int, Delt_0::Real)
    return [(i < N) ? 0.0 : Delt_0 for i = 1:N]
end

begin
    const EMField = field.EMField
    # const em_inc_function = AtomicArrays.field.gauss
    const em_inc_function = AtomicArrays.field.plane
    NMAX = 50
    NMAX_T = 5

    const PATH_FIGS, PATH_DATA = AtomicArraysStatistics.path()

    # System parameters
    γ = 0.1
    e_dipole = [1., 0, 0]
    T = [0:0.05:500;]
    N = 2
    Ncenter = Int(ceil(N / 2))
    DETUNING_SYMMETRY = true
    GEOMETRY = "chain"
    NON_HERMITIAN = true

    if NON_HERMITIAN
        i_states = ["|B>", "|D>"]
    else
        # Choose which states to project onto (|j,m>)
        if N == 2
            i_states = [(1,0), (0,0)]
        elseif N == 4
            i_states = [(2,-1), (2,0),
                        (1,-1,1), (1,-1,2), (1,-1,3),
                        (0,0,1), (0,0,2)]
        end
    end
end


begin
    # Parameters to vary
    dir_list = ["R", "L"];
    Delt_list = γ .* range(-2.0, 2.0, NMAX);
    E_list = γ .* range(1e-2, 2.0e-0, NMAX);
    d_list = range(2e-1, 10e-1, NMAX);

    projections = zeros(2, length(i_states), NMAX, NMAX, NMAX);

    # debug
    w_list = []
    w_list_all = zeros(ComplexF64, 2, 2^N, NMAX, NMAX, NMAX)
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
        pos = geometry.chain_dir(a, N; dir="z", pos_0=[0, 0, - (N - 1) * a / 2])
    end
    if DETUNING_SYMMETRY
        Delt = detuning_generator_sym(N, Delt_0)
    else
        Delt = detuning_generator_nonsym_1atom(N, Delt_0)
    end
    gammas_array = [AtomicArraysStatistics.gamma_detuned(γ, delt) for delt in Delt]
    S = SpinCollection(pos, [e_dipole for i in 1:N], gammas_array; deltas=Delt)

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
    # Steady-state
    λ, _ = eigen(Γ)
    ρ_ss = nothing
    try
        # Attempt to calculate the steady state using the eigenvector method
        ρ_ss = QuantumOptics.steadystate.eigenvector(H, J; rates=Γ, nev=1)
    catch e
        # If an error occurs, proceed with the alternative approach
        @warn "Failed to calculate steady state with eigenvector method. Proceeding with time evolution. Error: $e"
        
        # Alternative approach
        phi = 0.0
        theta = pi / 1.0
        Ψ₀ = AtomicArrays.quantum.blochstate(phi, theta, N)
        ρ₀ = Ψ₀ ⊗ dagger(Ψ₀)
        T = [0:0.5:10.0 / abs(λ[1]);]
        _, ρ_t = QuantumOptics.timeevolution.master_h(T, ρ₀, H, J; rates=Γ)
        ρ_ss = ρ_t[end]
    end

    # Dark and Bright states from non-Hermitian Hamiltonian
    if NON_HERMITIAN
        H_nh = H - 0.5im * sum(Γ[j,k] * Jdagger[j] * J[k] for j = 1:N, k = 1:N)
        w_nh, v_nh = eigenstates(dense(H_nh); warning=false)
        sorting_idx = sortperm(abs.(imag(w_nh)))
        v_nh_sorted = v_nh[sorting_idx]
        w_list_all[kk, :, ii, jj, mm] = w_nh[sorting_idx]
        append!(w_list, w_nh[sorting_idx][1])
        # Choose only bright and dark states
        rhos = [v_nh_sorted[end] ⊗ dagger(v_nh_sorted[end]),
                v_nh_sorted[1] ⊗ dagger(v_nh_sorted[1])]
        # rhos = [v_nh_sorted[end] ⊗ dagger(v_nh_sorted[end]),
        #         v_nh_sorted[1] ⊗ dagger(v_nh_sorted[1])]
        # Projections
        projections[kk, :, ii, jj, mm] = [real(tr(ρ_ss * ρ)) for ρ in rhos]
    else
        # Dark and Bright states from |j,m>
        if N == 4
            states_all = AtomicArraysStatistics.spherical_basis_jm_4(J; γ=γ)
            states_flattened = vcat(vcat(states_all...)...)
            idx_fl = map(x -> AtomicArraysStatistics.find_flattened_state_index(
                states_all, x), i_states)
            states = states_flattened[idx_fl]
            rhos = [ψ ⊗ dagger(ψ) for ψ in states]
            # Projections
            projections[kk, :, ii, jj, mm] = [real(tr(ρ_ss * ρ)) for ρ in rhos]
        elseif N == 2
            ψ_D = 1.0/sqrt(2.0) * (Ket(basis(H), [0,1,0,0]) - 
                                Ket(basis(H), [0,0,1,0]))
            ψ_B = 1.0/sqrt(2.0) * (Ket(basis(H), [0,1,0,0]) + 
                                Ket(basis(H), [0,0,1,0]))
            ρ_D = ψ_D ⊗ dagger(ψ_D)
            ρ_B = ψ_B ⊗ dagger(ψ_B)
            # Projections
            projections[kk, 1, ii, jj, mm] = real(tr(ρ_ss * ρ_B))
            projections[kk, 2, ii, jj, mm] = real(tr(ρ_ss * ρ_D))
        end
    end
    next!(progress)
end

maximum(projections[1, 2, ..])

maximum(projections[1, 1, ..])

begin
    eff = AtomicArrays.field.objective(projections[1,..], projections[2,..]);
    (maximum(eff[2,..]), argmax(eff[2,..]), (Delt_list[argmax(eff[2,..])[1]], E_list[argmax(eff[2,..])[2]], d_list[argmax(eff[2,..])[3]]))
end

begin
    # Position of the maximum
    i_st = 2  # index of the state
    idx_RD = argmax(projections[1,i_st,..])
    idx_LD = argmax(projections[2,i_st,..])
    ((projections[1,i_st, idx_RD], projections[1,1, idx_RD]),
    (projections[2,i_st, idx_LD], projections[2,1, idx_LD]), 
    (Delt_list[idx_RD[1]], E_list[idx_RD[2]], d_list[idx_RD[3]]),
    (Delt_list[idx_LD[1]], E_list[idx_LD[2]], d_list[idx_LD[3]]))
end

begin 
    i_st = 1  # index of the state
    idx_RB = argmax(projections[1,i_st,..])
    idx_LB = argmax(projections[2,i_st,..])
    ((projections[1,i_st, idx_RB], projections[1,2, idx_RB]), 
    (projections[2,i_st, idx_LB], projections[2,2, idx_LB]),
    (Delt_list[idx_RB[1]], E_list[idx_RB[2]], d_list[idx_RB[3]]),
    (Delt_list[idx_LB[1]], E_list[idx_LB[2]], d_list[idx_LB[3]]))
end

begin
    # state idices
    i_B = 1
    i_D = 2
    # max contrast position
    idx_D = argmax(eff[i_D,..])
    idx_B = argmax(eff[i_B,..])
    ((projections[1,i_D,idx_D], projections[2,i_D,idx_D],projections[1,i_B,idx_D], projections[2,i_B,idx_D]),
    (Delt_list[idx_D[1]], E_list[idx_D[2]], d_list[idx_D[3]]),
    (projections[1,i_D,idx_B], projections[2,i_D,idx_B],projections[1,i_B,idx_B], projections[2,i_B,idx_B]),
    (Delt_list[idx_B[1]], E_list[idx_B[2]], d_list[idx_B[3]]))
end

begin
    # select the states to compare
    i_B = 1
    i_D = 2
    idx_D = argmax(eff[i_D,..])
    idx_B = argmax(eff[i_B,..])
end
let
    idxD = idx_D[2]
    idxB = idx_B[2]

    x = Delt_list
    y = d_list
    c_D_1 = projections[1,i_D, :, idxD, :]
    c_D_2 = projections[2,i_D, :, idxD, :]
    c_B_1 = projections[1,i_B, :, idxB, :]
    c_B_2 = projections[2,i_B, :, idxB, :]
    cmap = "viridis"

    id_max_D_R = argmax(projections[1,i_D, :, idxD, :])
    id_max_D_L = argmax(projections[2,i_D, :, idxD, :])
    id_max_B_R = argmax(projections[1,i_B, :, idxB, :])
    id_max_B_L = argmax(projections[2,i_B, :, idxB, :])

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
    c_D_1 = eff[i_D, :, idxD, :]
    c_D_2 = eff[i_B, :, idxB, :]
    cmap = "viridis"

    id_max_D = argmax(eff[i_D, :, idxD, :])
    id_max_B = argmax(eff[i_B, :, idxB, :])

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

minimum(abs.(imag(w_list))/γ*2)
w_list_all[1,:,idx_B]

let 
    minimum(abs.(imag(w_list))/γ*2)
    fig, ax = plt.subplots(1, 1) 
    ax.plot(real(w_list)/γ*2, imag(w_list)/γ*2, "o", ms=0.05)
    # ax.set_xlim(-0.2,0.2)
    # ax.set_ylim(-0.3,0.0)
    fig
end


# Define the file path
if NON_HERMITIAN
    csv_file = PATH_DATA*"max_projection_params_N"*string(N)*"_nh_new.csv"
else
    csv_file = PATH_DATA*"max_projection_params_N"*string(N)*".csv"
end
# csv_file = PATH_DATA*"experiment_results.csv"

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
if NON_HERMITIAN
    states = i_states
else
    states = AtomicArraysStatistics.state_string.(i_states)
end
# TODO: this should be generalized

for (i, st) in enumerate(states)
    arr_Dir = projections[:,i,..]
    arr_eff = eff[i,..]
    for (j, dir) in enumerate(directions)
        if dir == "R"
            idx = argmax(arr_Dir[1, ..])
        elseif dir == "L"
            idx = argmax(arr_Dir[2, ..])
        elseif dir == "E"
            idx = argmax(arr_eff)
        end

        params_to_write = Dict("State_proj_max" => st, "N"=>N,
                       "a"=>round(d_list[idx[3]], digits=3), 
                       "E₀"=>round(E_list[idx[2]], digits=3),
                       "Direction"=>dir, 
                       "detuning_symmetry"=>DETUNING_SYMMETRY, 
                       "geometry"=>GEOMETRY,
                       )
        # Iteratively add "proj_"*state to the dictionary
        projection_keys = String[]
        for (id, state) in enumerate(states)
            key = "proj_" * state
            push!(projection_keys, key * "_L")
            push!(projection_keys, key * "_R")
            params_to_write[key*"_L"] = round(projections[2, id, idx], digits=3)
            params_to_write[key*"_R"] = round(projections[1, id, idx], digits=3)
        end
        if (DETUNING_SYMMETRY) 
            delts = round.(detuning_generator_sym(N, Delt_list[idx[1]]), 
                           digits=3)
        else
            delts = round.(detuning_generator_nonsym_1atom(N,
                                                           Delt_list[idx[1]]),
                           digits=3)
        end
        for i = 1:N
            params_to_write["Δ_$i"] = delts[i]
        end

        # Check if the CSV file exists, and create it with headers if it doesn't
        if !isfile(csv_file)
            headers = ["State_proj_max", "N", "detuning_symmetry", "geometry", 
                    "Direction", "a", "E₀"]
            for i in 1:N
                push!(headers, "Δ_$i")
            end  
            append!(headers, projection_keys) 

            df = DataFrame([[] for _ = headers] , headers)
            append!(df, DataFrame(params_to_write))
            CSV.write(csv_file, df)
        end
        # Example usage: add experiment parameters for a specific configuration
        add_experiment_results(params_to_write)
    end
end