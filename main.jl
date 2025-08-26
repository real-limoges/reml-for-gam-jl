using DataFrames
using BSplines
using MixedModels
using StatsModels
using Plots
using LinearAlgebra
using Random


function simulate_gam_data(n::Int; noise_level::Float64=0.15, seed::Int=42)
    Random.seed!(seed)
    x = rand(n)

    true_f(val) = val^11 * (10 * (1 - val))^6 + (10 * val)^3 * (1 - val)^10
    y_true = true_f.(x)
    noise = randn(n) .* (std(y_true) * noise_level)
    y = y_true + noise

    println("Data simulated successfully.")
    return DataFrame(x = x, y = y, true_f = y_true)
end

function create_bspline_basis_and_penalty(x::Vector, k::Int, order::Int)
    knots = range(minimum(x), maximum(x), length = k - order + 2)
    basis = BSplineBasis(order, knots)
    X_basis = basismatrix(basis, x)

    Ik = Matrix{Float64}(I, k, k)
    D1 = diff(Ik, dims = 1)
    D = diff(D1, dims=1)

    # Second-order difference penalty matrix (penalizes wiggliness)
    S_penalty = D' * D

    println("Basis matrix dimensions: ", size(X_basis))
    println("Penalty matrix dimensions: ", size(S_penalty))
    return basis, X_basis, S_penalty
end

function reparameterize_basis(X_basis::Matrix, S_penalty::Matrix; tol::Float64=1e-8)
    F = eigen(S_penalty)
    eigenvalues = F.values
    eigenvectors = F.vectors

    null_space_indices = findall(abs.(eigenvalues) .< tol)
    penalized_indices = findall(abs.(eigenvalues) .>= tol)

    X_fixed = X_basis * eigenvectors[:, null_space_indices]
    inv_sqrt_eigenvalues = diagm(1 ./ sqrt.(eigenvalues[penalized_indices]))
    X_random = X_basis * eigenvectors[:, penalized_indices] * inv_sqrt_eigenvalues

    println("\nReparameterization complete.")
    println(size(X_fixed, 2), " basis functions treated as fixed effects.")
    println(size(X_random, 2), " basis functions treated as random effects.")

    return (
        X_fixed = X_fixed,
        X_random = X_random,
        eigenvectors = eigenvectors,
        inv_sqrt_eigenvalues = inv_sqrt_eigenvalues,
        null_indices = null_space_indices,
        penalized_indices = penalized_indices
    )
end

function fit_gam_mixed_model(df::DataFrame, X_fixed::Matrix, X_random::Matrix)
    fit_data = hcat(df, DataFrame(X_fixed, :auto), DataFrame(X_random, :auto), makeunique=:true)
    fit_data.dummy_group = ones(nrow(df))

    fixed_terms = term.(Symbol.(names(fit_data, r"^x\d+$")))
    random_terms = term.(Symbol.(names(fit_data, r"^x\d+_\d+$")))

    lhs = term(:y)
    fixed_rhs = ConstantTerm(1) + reduce(+, term.(fixed_terms))
    random_rhs = reduce(+, term.(random_terms)) | term(:dummy_group)
    rhs = fixed_rhs + random_rhs

    lmm_formula = lhs ~ rhs

    println("\nFitting mixed model with formula:\n", lmm_formula)
    return fit(MixedModel, lmm_formula, fit_data)
end

# function reconstruct_smooth(model::LinearMixedModel, X_basis::Matrix, reparam::NamedTuple)
#     fixed_coeffs = fixef(model)
#     random_coeffs = ranef(model)[1]
#
#     # Transform coefficients back to the original basis space
#     original_basis_coeffs = reparam.eigenvectors[:, reparam.null_indices] * fixed_coeffs[2:end] .+
#                              reparam.eigenvectors[:, reparam.penalized_indices] * reparam.inv_sqrt_eigenvalues * random_coeffs
#
#     # Calculate the final smooth curve
#     return X_basis * original_basis_coeffs .+ fixed_coeffs[1]
# end

# function plot_gam_fit(df::DataFrame, basis::BSplineBasis, fitted_coeffs::Vector, intercept::Float64)
#     sort!(df, :x) # Sort for clean plotting lines
#
#     # Recalculate smooth on the sorted x-values for a clean line
#     sorted_X_basis = basismatrix(basis, df.x)
#     fitted_gam_sorted = sorted_X_basis * fitted_coeffs .+ intercept
#
#     p = plot(df.x, df.y, seriestype=:scatter, label="Data", color=:gray, alpha=0.5, markershape=:circle, markerstrokewidth=0)
#     plot!(p, df.x, df.true_f, label="True Function", linewidth=2.5, color=:black)
#     plot!(p, df.x, fitted_gam_sorted, label="Mixed Model GAM Fit", linewidth=2.5, linestyle=:dash, color=:dodgerblue)
#     title!("GAM as a Mixed Model in Julia")
#     xlabel!("Predictor (x)")
#     ylabel!("Response (y)")
#     plot!(p, legend=:bottomright)
#     return p
# end


function main()
    n_points = 10000
    n_basis_funcs = 12
    spline_order = 4 # Cubic

    sim_data = simulate_gam_data(n_points)

    basis, X_basis, S_penalty = create_bspline_basis_and_penalty(sim_data.x, n_basis_funcs, spline_order)

    reparam = reparameterize_basis(X_basis, S_penalty)
    model_fit = fit_gam_mixed_model(sim_data, reparam.X_fixed, reparam.X_random)
    println(model_fit)

    # We need the coefficients in the original basis space for plotting
    # TODO: Fix this
#     fixed_c = fixef(model_fit)
#     random_c = ranef(model_fit)[1]
#
#     num_fixed = length(fixed_c)
#     correct_indices = (num_fixed-1):num_fixed
#
#     original_coeffs = reparam.eigenvectors[:, reparam.null_indices] * fixed_c[correct_indices] .+
#                       reparam.eigenvectors[:, reparam.penalized_indices] * (reparam.inv_sqrt_eigenvalues .* random_c)
#
#     final_plot = plot_gam_fit(sim_data, basis, original_coeffs, fixed_c[1])
#     display(final_plot)
end

main()