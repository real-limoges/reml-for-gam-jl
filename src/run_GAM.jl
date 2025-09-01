using DataFrames, Random, GLM, Plots, Printf, Statistics

include("FastGAM.jl")
using .FastGAM

Random.seed!(42)

# Data Creation
println("--- Generating synthetic data ---")
n = 1000

column_data = Dict{Symbol, Vector{Float64}}()

column_data[:x0] = collect(range(0, 1, length=n))
column_data[:x1] = rand(n)
column_data[:x2] = rand(n) * 2
for i in 3:9
    column_data[Symbol("x", i)] = randn(n)
end

# Define the true underlying functions
true_f0 = sin.(3 * pi * column_data[:x0]) .* exp.(-column_data[:x0])
true_f1 = cos.(2 * pi * column_data[:x1])
linear_f2 = 2 * column_data[:x2]

# Combine into the response variable with some noise
column_data[:y] = true_f0 .+ true_f1 .+ linear_f2 .+ randn(n) * 0.3

data = DataFrame(column_data)


# Define and Fit the GAM
# I'm gonna just throw some stuff at it.

# Define the linear part of the model.
# x2 has a true linear effect. x3 is a nuisance variable.
linear_formula = @formula(y ~ 1 + x2 + x3)

# Define the smooth part of the model.
# Just taking non-linear of first two columns and giving them different splines
smooth_terms = [
    Smooth(:x0, spline_type=:b_spline, n_knots=20),
    Smooth(:x1, spline_type=:pc_spline, n_knots=15)
]

println("\n--- Fitting GAM with multiple smooths ---")
gam_multi = GAM(linear_formula, data, smooth_terms)
@time FastGAM.fit!(gam_multi)


# Summary
println("\n--- Model Summary ---")
@printf "REML score:         %.4f\n" gam_multi.reml_score
@printf "Scale estimate:     %.4f\n" gam_multi.scale
@printf "EDF:                %.4f\n" gam_multi.edf
@printf "Adj. R²:            %.4f\n" gam_multi.adj_r_squared

println("\n--- Smoothing Parameters (Lambdas) ---")
for (i, s) in enumerate(gam_multi.smooths)
    @printf "  - Smooth for '%-4s': %.4f\n" s.term gam_multi.lambdas[i]
end

# Plot Partial Effects
println("\n--- Generating partial effects plots ---")

function get_partial_fit(model::GAM, term::Symbol)
    # Extract the coefficients and basis matrix corresponding to the smooth term
    beta_smooth = model.beta[model.param_indices[term]]
    basis_smooth = model.X[:, model.param_indices[term]]
    
    # Calculate the fitted smooth function
    fitted_smooth = basis_smooth * beta_smooth
    
    # Center the fitted smooth for identifiability (makes it comparable to the true function)
    return fitted_smooth .- mean(fitted_smooth)
end

# Plot for x0
sort_idx_0 = sortperm(data.x0)
fitted_f0 = get_partial_fit(gam_multi, :x0)

p0 = plot(data.x0[sort_idx_0], true_f0[sort_idx_0], 
          label="True f(x0)", color=:black, linewidth=2, legend=:topright)
plot!(p0, data.x0[sort_idx_0], fitted_f0[sort_idx_0], 
      label="Fitted s(x0)", color=:red, linestyle=:dash, linewidth=2)
title!(p0, "Partial Effect for x0")
xlabel!(p0, "x0")
ylabel!(p0, "f(x0)")
savefig(p0, "partial_effect_x0.png")
println("Saved plot to partial_effect_x0.png")


# Plot for x1
sort_idx_1 = sortperm(data.x1)
fitted_f1 = get_partial_fit(gam_multi, :x1)

p1 = plot(data.x1[sort_idx_1], true_f1[sort_idx_1], 
          label="True f(x1)", color=:black, linewidth=2, legend=:bottomleft)
plot!(p1, data.x1[sort_idx_1], fitted_f1[sort_idx_1], 
      label="Fitted s(x1)", color=:blue, linestyle=:dash, linewidth=2)
title!(p1, "Partial Effect for x1")
xlabel!(p1, "x1")
ylabel!(p1, "f(x1)")
savefig(p1, "partial_effect_x1.png")
println("Saved plot to partial_effect_x1.png")

println("\nDone.")
