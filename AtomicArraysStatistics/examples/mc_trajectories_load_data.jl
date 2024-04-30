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

begin
    using LinearAlgebra
    using Statistics, StatsBase
    using QuantumOptics
    using PyPlot
    using WGLMakie
    using AtomicArrays
    using Revise
    using BenchmarkTools
    using ProgressMeter
    using HDF5, FileIO, Printf

    using AtomicArraysStatistics
end

#function
begin
    function idx_2D_to_1D(i, j, nmax)
        return (i - 1) * nmax + j
    end

    function idx_1D_to_2D(i, nmax)
        return CartesianIndex((i - 1) ÷ nmax + 1, (i - 1) % nmax + 1)
    end

    """
    load_jump_data()

    Loads a dict with data according to the parameters specified in the workspace.

    * Output:
    - (theta_var, phi_var, jump_t, jump_i)
    """
    function load_data(name::String, direction::String; 
                            tau_max::Real = 5e5, N_traj::Int = 10,
                            N_angle_points::Int = 10, N_atoms::Int = 2)
        time_str = @sprintf "%.0E" tau_max*N_traj
        NAME_PART = string(N_atoms)*"atoms_tmax"*time_str*"_nmax"*string(N_angle_points)*"_"*direction*".h5"
        return load(PATH_DATA*name*"_"*NAME_PART)
    end
end

# parameters
begin
    const PATH_FIGS, PATH_DATA = AtomicArraysStatistics.path()
    NAME = "jumps"
    NMAX = 30
    DIRECTION = "R"
    N_traj = 1000
end


"WTD for S(θ, ϕ) operators"

# loading data
begin
    data_dict = load_data(NAME, DIRECTION; N_traj=N_traj, N_angle_points=NMAX)

    theta_var = data_dict["theta"]
    phi_var = data_dict["phi"]
    if NAME == "w_angle"
        w_angle_0 = data_dict["w_angle_0"]
    elseif NAME == "jumps"
        jump_t = data_dict["jump_times"]
        jump_i = data_dict["jump_op_idx"]
        jump_t_S = data_dict["jump_times_S"]
        jump_i_S = data_dict["jump_op_idx_S"]
    end
end

if NAME == "jumps"

    N_BINS = 500

    d_angle = 2.0 / NMAX * pi 

    # dΩ = sin(θ) dθ dϕ
    dΩ = [d_angle * d_angle * sin(theta_var[i]) for i = 1:NMAX ÷ 2, j = 1:NMAX]

    length(jump_t_S)
    length(jump_i_S)


    begin  # compute WTD by the angle
        w_tau_S = AtomicArraysStatistics.compute_w_tau(jump_S_t)

        # Initialize arrays outside the loop for efficiency
        w_tau_S_n = Vector{Vector{Float64}}()
        idx_no_stat_S = Int[]


        # Iterate through indices and jumps
        for i in eachindex(D)
            AtomicArraysStatistics.compute_w_tau_n(w_tau_S_n, idx_no_stat_S,
                                                   jump_t_S, jump_i_S, i)
        end
        println("w_tau_S_n computed.")
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

# plots
    let
        idx = NMAX*NMAX ÷ 2 ÷ 2
        h = fit(Histogram, w_tau_S_n[idx] ./ mean(w_tau_S_n[idx]), nbins=N_BINS)
        h_0 = normalize(h, mode=:pdf)
        
        fig_03, axs = plt.subplots(1, 1, sharey=true, tight_layout=true, figsize=(6,3))
        # axs.plot(test[1:1000])
        n, bins, patches = axs.hist(w_tau_S ./ mean(w_tau_S), bins=N_BINS, density=true, histtype="bar", label=L"w(\tau / \bar{\tau})")
        axs.hist(w_tau_S_n[idx] ./ mean(w_tau_S_n[idx]), bins=N_BINS, density=true, alpha=0.5, histtype="bar", label=L"w_n(\tau / \bar{\tau})")
        axs.plot(bins[1:end-1], γ*exp.(-γ*bins[1:end-1]), color="red", label=L"\gamma \exp(-\gamma \tau / \bar{\tau})")
        axs.plot(h_0.edges[1][1:end-1], h_0.weights, color="blue")
        axs.set_xlim((0, 1))
        # axs.set_ylim((0, 2))
        # axs.set_yscale("log")
        axs.set_xlabel(L"\tau / \bar{\tau}")
        axs.set_ylabel(L"w(\tau / \bar{\tau}), g^{2}(\tau / \bar{\tau})")
        axs.legend(loc="upper right")
        display(fig_03)
        # fig_03.savefig(PATH_FIGS * "wtau_g2_N" * string(N) * "_" * "jump_" * DIRECTION * ".pdf", dpi=300)
    end

    # let
    #     fig_04, axs = plt.subplots(1, 1, sharey=true, tight_layout=true, figsize=(6,6))
    #     n, bins, patches = axs.hist(w_tau_S ./ mean(w_tau_S), bins=N_BINS, density=true, histtype="bar", label=L"w(\tau / \bar{\tau})")
    #     for i = 1:NMAX * (NMAX ÷ 2)
    #         axs.hist(w_tau_S_n[i] ./ mean(w_tau_S_n[i]), bins=N_BINS, density=true, alpha=0.1, histtype="bar")
    #     end
    #     axs.plot(bins[1:end-1], γ*exp.(-γ*bins[1:end-1]), color="red", label=L"\gamma \exp(-\gamma \tau / \bar{\tau})")
    #     axs.set_xlim((0, 1))
    #     # axs.set_ylim((0, 2))
    #     # axs.set_yscale("log")
    #     axs.set_xlabel(L"\tau / \bar{\tau}")
    #     axs.set_ylabel(L"w(\tau / \bar{\tau}), g^{2}(\tau / \bar{\tau})")
    #     axs.legend(loc="upper right")
    #     display(fig_04)
    #     # fig_03.savefig(PATH_FIGS * "wtau_g2_N" * string(N) * "_" * "jump_" * DIRECTION * ".pdf", dpi=300)
    # end
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

    WGLMakie.activate!()
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