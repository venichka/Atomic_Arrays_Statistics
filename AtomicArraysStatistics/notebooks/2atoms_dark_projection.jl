### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 394c87e0-b883-11ed-22fa-07d53915e1d6
begin
    if pwd()[end-21:end] == "AtomicArraysStatistics"
        PATH_ENV = "."
    else
        PATH_ENV = "../"
    end
    using Pkg
    Pkg.activate(PATH_ENV)
end


# ╔═╡ 02abe02b-2e01-4d2f-abd3-8aaf7abb9233
begin
	using LinearAlgebra
	using QuantumOptics
	using PyPlot
	using AtomicArrays
	using Revise
	using InteractiveUtils
	using PlutoUI

	using AtomicArraysStatistics

	import EllipsisNotation: Ellipsis
    const .. = Ellipsis()
end

# ╔═╡ e1fef183-0ead-4cbc-b0bf-6515b8cad27b
begin
	const EMField = field.EMField
	# const em_inc_function = AtomicArrays.field.gauss
	const em_inc_function = AtomicArrays.field.plane
	const NMAX = 20
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
	

# ╔═╡ ec6304bc-eadf-4447-bee8-7c10eb98b75c
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

# ╔═╡ 87548a12-5ec3-4b55-b552-4c6eca5b4b83
# ╠═╡ show_logs = false
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
#	ρ_ss = QuantumOptics.steadystate.eigenvector(H, J; rates=Γ)
	
	phi = 0.
	theta = pi/1.
	Ψ₀ = AtomicArrays.quantum.blochstate(phi,theta,N)
	ρ₀ = Ψ₀⊗dagger(Ψ₀)
	_, ρ_t = QuantumOptics.timeevolution.master_h(T, ρ₀, H, J; rates=Γ)
	ρ_ss = ρ_t[end]

	# Projections
	projections_D[kk, ii, jj, mm] = real(tr(ρ_ss * ρ_D))
	projections_B[kk, ii, jj, mm] = real(tr(ρ_ss * ρ_B))
end

# ╔═╡ e02fd079-1a7e-46b1-89ce-6b409b30d7aa
maximum(projections_D[2,..])

# ╔═╡ f2342c62-a164-4f75-a7ac-4d1e1ae5add9
maximum(projections_B[2,..])

# ╔═╡ 60943572-93d4-4369-af53-f28c120580bc
@bind idx PlutoUI.Slider(1:NMAX, default=1)

# ╔═╡ 16c847e6-6e79-4359-8e36-6c95e687973f
print(E_list[idx])

# ╔═╡ 2f9a9aa6-9f7b-42df-a2cc-f96e4fae49b8
let
	x = Delt_list
	y = d_list
	c_D_1 = projections_D[1, :, idx, :]
	c_D_2 = projections_D[2, :, idx, :]
	cmap = "viridis"
	
	fig, ax = plt.subplots(1, 2, figsize=(9,3))
	im1 = ax[1].pcolormesh(x, y, c_D_1, cmap=cmap)
	ax[1].set_title(L"R")
	ax[1].set_xlabel(L"\Delta")
    ax[1].set_ylabel(L"a")
	fig.colorbar(im1, ax=ax[1])
	im2 = ax[2].pcolormesh(x, y, c_D_2, cmap=cmap)
	ax[2].set_title(L"L")
	ax[2].set_xlabel(L"\Delta")
	fig.colorbar(im2, ax=ax[2])
	fig
end

# ╔═╡ c9c0b8a2-9759-4736-a800-a3e18eaa4aa9
projections_D[2, 7, 3, 2]

# ╔═╡ 2bd91f19-a6e9-41c1-83b5-7ad944349621
# ╠═╡ disabled = true
#=╠═╡
begin
	
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
  ╠═╡ =#

# ╔═╡ c88897c5-dd04-4c2b-a0be-f8306bd21d21
#=╠═╡
result[2][end] - ρ_ss
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═394c87e0-b883-11ed-22fa-07d53915e1d6
# ╠═02abe02b-2e01-4d2f-abd3-8aaf7abb9233
# ╠═e1fef183-0ead-4cbc-b0bf-6515b8cad27b
# ╠═ec6304bc-eadf-4447-bee8-7c10eb98b75c
# ╠═87548a12-5ec3-4b55-b552-4c6eca5b4b83
# ╠═e02fd079-1a7e-46b1-89ce-6b409b30d7aa
# ╠═f2342c62-a164-4f75-a7ac-4d1e1ae5add9
# ╠═60943572-93d4-4369-af53-f28c120580bc
# ╠═16c847e6-6e79-4359-8e36-6c95e687973f
# ╠═2f9a9aa6-9f7b-42df-a2cc-f96e4fae49b8
# ╠═c9c0b8a2-9759-4736-a800-a3e18eaa4aa9
# ╠═2bd91f19-a6e9-41c1-83b5-7ad944349621
# ╠═c88897c5-dd04-4c2b-a0be-f8306bd21d21
