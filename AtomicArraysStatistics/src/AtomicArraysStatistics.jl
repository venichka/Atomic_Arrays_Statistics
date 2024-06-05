module AtomicArraysStatistics

using LinearAlgebra
using QuantumOptics
using AtomicArrays
using DataFrames, CSV

"""
    AtomicArraysStatistics.correlation_3op_1t([tspan, ]rho0, H, J, A, B, C; <keyword arguments>)
Calculate one time correlation values ``⟨A(0)B(\\tau)C(0)⟩``.
The calculation is done by multiplying the initial density operator
with ``C \\rho A`` performing a time evolution according to a master equation
and then calculating the expectation value ``\\mathrm{Tr} \\{B ρ\\}``
Without the `tspan` argument the points in time are chosen automatically from
the ode solver and the final time is determined by the steady state termination
criterion specified in [`steadystate.master`](@ref).
# Arguments
* `tspan`: Points of time at which the correlation should be calculated.
* `rho0`: Initial density operator.
* `H`: Operator specifying the Hamiltonian.
* `J`: Vector of jump operators.
* `A`: Operator at time `t=0`.
* `B`: Operator at time `t=\\tau`.
* `C`: Operator at time `t=0`.
* `rates=ones(N)`: Vector or matrix specifying the coefficients (decay rates)
        for the jump operators.
* `Jdagger=dagger.(J)`: Vector containing the hermitian conjugates of the jump
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function correlation_3op_1t(tspan, rho0::Operator, H::AbstractOperator, J,
                     A, B, C;
                     rates=nothing,
                     Jdagger=dagger.(J),
                     kwargs...)
    function fout(t, rho)
        expect(B, rho)
    end
    t,u = timeevolution.master(tspan, C*rho0*A, H, J; rates=rates, Jdagger=Jdagger,
                        fout=fout, kwargs...)
    return u
end

"""
    AtomicArraysStatistics.correlation_3op_1t(rho0, H, J, A, B, C; <keyword arguments>)
Calculate steady-state correlation values ``⟨A(0)B(0)C(0)⟩``.
The calculation is done by multiplying the initial density operator
with ``C \\rho A`` and then calculating the expectation value ``\\mathrm{Tr} \\{B C \\rho A\\}``
# Arguments
* `rho0`: Initial density operator.
* `A`: Operator at time `t=0`.
* `B`: Operator at time `t=\\tau`.
* `C`: Operator at time `t=0`.
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function correlation_3op_1t(rho0::Operator, A, B, C; kwargs...)
    return expect(B, C*rho0*A)
end


"""
    AtomicArraysStatistics.coherence_function_g2([tspan, ]rho0, H, J, A_op; <keyword arguments>)
Calculate one time correlation values

``g^{(2)}(\\tau) =
        \\frac{\\langle A^\\dagger(0)A^\\dagger(\\tau)A(\\tau)A(0)\\rangle}
        {\\langle A^\\dagger(\\tau)A(\\tau)\\rangle
         \\langle A^\\dagger(0)A(0)\\rangle}``.

# Arguments
* `tspan`: Points of time at which the correlation should be calculated.
* `H`: Operator specifying the Hamiltonian.
* `J`: Vector of jump operators.
* `A_op`: Operator at time `t=0`.
* `rates=ones(N)`: Vector or matrix specifying the coefficients (decay rates)
        for the jump operators.
* `rho0=nothing`: Initial density operator, if nothing `rho0 = rho_ss`.
* `Jdagger=dagger.(J)`: Vector containing the hermitian conjugates of the jump
* `kwargs...`: Further arguments are passed on to the ode solver.
"""
function coherence_function_g2(tspan, H::AbstractOperator, J, A_op;
                     rates=nothing,
                     rho0=nothing,
                     Jdagger=dagger.(J),
                     kwargs...)
    function fout(t, rho)
        expect(dagger(A_op) * A_op, rho)
    end
    if isnothing(rho0)
        rho0 = QuantumOptics.steadystate.eigenvector(H, J; rates=rates)
        n_ss = [expect(dagger(A_op) * A_op, rho0)]
    else
        _, n_ss = timeevolution.master(tspan, rho0, H, J; rates=rates, Jdagger=Jdagger,
                        fout=fout, kwargs...)
    end
    t,u = timeevolution.master(tspan, A_op*rho0*dagger(A_op), H, J; rates=rates, Jdagger=Jdagger,
                        fout=fout, kwargs...)
    return u ./ (n_ss[1] * n_ss)
end


"""
    AtomicArraysStatistics.jump_op_source_mode(Γ, J)
Calculate the source-mode jump operators

``\\hat{J}_l = \\sqrt{\\lambda_l} \\mathbf{b}_l^T \\hat{\\mathbf{\\Sigma}}``.

# Arguments
* `Γ`: matrix of decay rates
* `J`: Vector of common jump operators.

# Return
* `J_s`: vector of the source-mode jump operators
"""
function jump_op_source_mode(Γ, J)
    N = rank(Γ)
    λ_g, β_g = eigen(Γ)
    J_s = [sqrt.(λ_g[l]) * β_g[l,:]' * J for l = 1:N]
    return J_s
end


"""
    AtomicArraysStatistics.D_angle(θ::Number, ϕ::Number, S::SpinCollection)
Calculate the source-mode jump operators

``D(\\theta, \\phi) = \\frac{3}{8 \\pi} \\left( 1 - (\\mu \\cdot \\hat{r}(\\theta, \\phi)^2) \\right)``.

# Arguments
* `θ`: angle between z axis and radius-vector
* `ϕ`: angle in xy plane starting from x axis
* `S`: spin collection

# Return
* `D_θϕ`: angular distribution
"""
function D_angle(θ::Number, ϕ::Number, S::SpinCollection)
    μ = S.polarizations[1]
    r_n = [sin.(θ)*cos.(ϕ), sin.(θ)*sin.(ϕ), cos(θ)]
    return 3.0 / (8.0*pi) * (1 - (μ' * r_n)^2)
end


"""
    AtomicArraysStatistics.jump_op_direct_detection(r::Vector, dΩ::Number, S::SpinCollection, J)
Calculate the direct-detection jump operators

``\\hat{S}(\\theta, \\phi) = \\sqrt{\\gamma D(\\theta, \\phi) d\\Omega} \\sum_{j=1}^N e^{-i k_0 \\hat{r}(\\theta, \\phi) \\cdot \\mathbf{r}_j} \\hat{\\sigma}_j``

``D(\\theta, \\phi) = \\frac{3}{8\\pi} \\left( 1 - \\left[ \\hat{\\mu} \\cdot \\hat{r}(\\theta, \\phi) \\right]^2\\right)``

# Arguments
* `r`: radius-vector (depends on ``\\theta, \\phi``)
* `dΩ`: element of solid angle in direction ``r(\\theta, \\phi)``
* `S`: spin collection
* `k_0`: wave number, ``k_0 = \\omega_0 / c``, where ``\\omega_0`` is a transition frequency of a atom
* `J`: Vector of common jump operators.

TODO: take into account different atomic frequencies

# Return
* `S_op`: vector of the source-mode jump operators
"""
function jump_op_direct_detection(r::Vector, dΩ::Number, S::SpinCollection, k_0::Number, J)
    N = length(S.gammas)
    μ = S.polarizations[1]
    γ = S.gammas[1]
    r_n = r ./ norm(r)
    D_θϕ = 3.0 / (8.0*pi) * (1 - (μ' * r_n)[1]^2)
    S_op = sqrt.(γ * D_θϕ * dΩ) * sum([exp(-im*k_0*(r_n' * S.spins[j].position)[1])*J[j]
                                       for j = 1:N])
    return S_op
end


function jump_op_direct_detection(θ::Real, ϕ::Real, dΩ::Number, S::SpinCollection, k_0::Number, J)
    N = length(S.gammas)
    μ = S.polarizations[1]
    γ = S.gammas[1]
    r_n = [sin.(θ)*cos.(ϕ), sin.(θ)*sin.(ϕ), cos(θ)]
    D_θϕ = 3.0 / (8.0*pi) * (1 - (μ' * r_n)[1]^2)
    S_op = sqrt.(γ * D_θϕ * dΩ) * sum([exp(-im*k_0*(r_n' * S.spins[j].position)[1])*J[j]
                                       for j = 1:N])
    return S_op
end


function compute_w_tau(jump_t)
    n = length(jump_t)
    w_tau = Vector{Float64}(undef, n-1)
    @inbounds for j in 1:(n-1)
        w_tau[j] = jump_t[j+1] - jump_t[j]
    end
    return filter!(x -> x >= 0, w_tau)
end

function compute_w_tau_n(w_tau_n, idx_no_stat, jump_t, jump_i, i)
    jumps = jump_t[jump_i .== i]
    jumps_dist = diff(jumps)
    jumps_dist = jumps_dist[jumps_dist .>= 0]
    if isempty(jumps_dist)
        push!(idx_no_stat, i)
        print(i, " ")
    end
    push!(w_tau_n, jumps_dist)
end

function g2_0_jump_opers(rho_ss::Operator, J_s)
    N = length(J_s)
    num = real(sum([AtomicArraysStatistics.correlation_3op_1t(rho_ss, 
    dagger(J_s[i]), dagger(J_s[j])*J_s[j], J_s[i]) for i = 1:N, j = 1:N]))
    denom = real(sum([QuantumOptics.expect(dagger(J_s[i])*J_s[i], rho_ss) for i = 1:N]).^2)
    return num / denom
end

function g2_tau_jump_opers(rho_ss::Operator, J_s, H, tspan)
    N = length(J_s)
    num = real(sum([AtomicArraysStatistics.correlation_3op_1t(tspan, rho_ss, H, J_s, 
    dagger(J_s[i]), dagger(J_s[j])*J_s[j], J_s[i]) for i = 1:N, j = 1:N]))
    denom = real(sum([QuantumOptics.expect(dagger(J_s[i])*J_s[i], rho_ss) for i = 1:N]).^2)
    return num ./ denom
end

# Function to retrieve parameters from the CSV file based on specified fields
function get_parameters_csv(csv_file, state, N, geometry, detuning_symmetry, direction)
    # Read the CSV file into a DataFrame
    df = CSV.read(csv_file, DataFrame)

    # Filter the DataFrame based on the specified fields
    filtered_df = filter(row -> row.State == state && row.N == N && row.geometry == geometry &&
                                row.detuning_symmetry == detuning_symmetry && row.Direction == direction, df)

    # Check if any rows match the criteria
    if nrow(filtered_df) == 0
        println("No matching parameters found.")
        return nothing
    end

    # Extract the desired parameters
    a = filtered_df.a[1]
    E₀ = filtered_df.E₀[1]
    Δ_params = zeros(Float64, N)
    for i in 1:N
        Δ_params[i] = filtered_df[!, Symbol("Δ_$i")][1]
    end

    return Dict("a" => a, "E_0" => E₀, "Δ_vec" => Δ_params)
end


"""
    AtomicArraysStatistics.path()

# Output: 
* PATH_FIGS
* PATH_DATA
"""
function path()
    home = homedir()
    if home == "C:\\Users\\nnefedkin"
        PATH_FIGS = "D:/nnefedkin/Google_Drive/Work/In process/Projects/Collective_effects_QMS/Figures/two_arrays/forward_scattering/"
        PATH_DATA = "D:/nnefedkin/Google_Drive/Work/In process/Projects/Collective_effects_QMS/Data/data_2arrays_mpc_mf/"
    elseif home == "/home/nikita"
        PATH_FIGS = "/home/nikita/Documents/Work/Projects/two_arrays/statistics/Figs/"
        PATH_DATA = "/home/nikita/Documents/Work/Projects/two_arrays/statistics/Data/"
    elseif home == "/Users/jimi"
        PATH_FIGS = "/Users/jimi/Google Drive/Work/In process/Projects/Statistics_QMS/Figs/"
        PATH_DATA = "/Users/jimi/Google Drive/Work/In process/Projects/Statistics_QMS/Data/"
    elseif home == "/Users/nikita"
        PATH_FIGS = "/Users/nikita/Google Drive/Work/In process/Projects/Statistics_QMS/Figs/"
        PATH_DATA = "/Users/nikita/Google Drive/Work/In process/Projects/Statistics_QMS/Data/"
    end
    return PATH_FIGS, PATH_DATA
end

end # module AtomicArraysStatistics
