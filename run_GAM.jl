using Pkg

# Activate the local project environment
Pkg.activate(".")

using DataFrames, Random, GLM, Plots, Statistics, ToeplitzMatrices
include("src/FastGAM.jl")
using .FastGAM

# Set a seed for reproducibility
Random.seed!(42)

# Synthetic Data for a Logistic GAM
println("--- Generating Synthetic Binary Data ---")
n = 1000
x0 = collect(range(-3, 3, length=n))
x1 = rand(n) * 4 - 2
x2 = randn(n)

true_f0 = -sin.(x0)
true_f1 = x1.^2 .- 3
linear_f2 = 0.5 * x2

eta = true_f0 .+ true_f1 .+ linear_f2
prob = 1 ./ (1 .+ exp.(-eta))
y = [rand() < p for p in prob]
data = DataFrame(y=y, x0=x0, x1=x1, x2=x2)

# GAM Meat and Potatoes
println("\n--- Fitting Logistic GAM (Binomial family) ---")
linear_formula = @formula(y ~ 1 + x2)
smooth_terms = [Smooth(:x0, n_knots=20), Smooth(:x1, n_knots=15, spline_type=:cubic_spline)]
gam_model = GAM(linear_formula, data, smooth_terms, family=Binomial())
@time FastGAM.fit!(gam_model)

# Summary Statistics
println("\n--- Model Summary ---")
println("GCV Score: ", round(gam_model.gcv_score, digits=4))
println("Effective DoF: ", round(gam_model.edf, digits=2))
println("\n--- Smoothing Parameters (Lambdas) ---")
for (i, s) in enumerate(gam_model.smooths)
    println("  s($(s.term)): ", round(gam_model.lambdas[i], digits=4))
end

# Partial Effects Plots
function plot_partial_effects(model::GAM, true_funcs::Dict)
    smooth_vars = [s.term for s in model.smooths]
    p = plot(layout=(length(smooth_vars), 1), size=(800, 400 * length(smooth_vars)))
    link_name = string(canonicallink(model.family))
    
    for (i, var) in enumerate(smooth_vars)
        smooth_indices = model.param_indices[var]
        eta_partial = model.X[:, smooth_indices] * model.beta[smooth_indices]
        eta_partial .-= mean(eta_partial)
        
        x_data = model.data[!, var]
        sort_order = sortperm(x_data)
        
        plot!(p[i], x_data[sort_order], eta_partial[sort_order], label="Fitted s($(var))", lw=2)
        
        if haskey(true_funcs, var)
            true_y = true_funcs[var]
            true_y .-= mean(true_y)
            plot!(p[i], x_data[sort_order], true_y[sort_order], label="True f($(var))", ls=:dash, color=:black)
        end
        
        scatter!(p[i], x_data, fill(minimum(eta_partial), length(x_data)), marker=:vline, markersize=5, markeralpha=0.2, label="", color=:gray)
        title!(p[i], "Partial Effect for s($(var))")
        xlabel!(p[i], string(var))
        ylabel!(p[i], "f($(var)) on $(link_name) scale")
    end
    return p
end

true_functions = Dict(:x0 => true_f0, :x1 => true_f1)
plots = plot_partial_effects(gam_model, true_functions)
savefig(plots, "charts/logistic_gam_partial_effects.png")
println("\n--- Plots saved to charts/logistic_gam_partial_effects.png ---")
