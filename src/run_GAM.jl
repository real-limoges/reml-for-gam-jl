using DataFrames, Random, GLM
include("FastGAM.jl")
using .FastGAM

Random.seed!(42)

# Generate Data
n = 1000
x = collect(range(0, 1, length=n))
true_f = sin.(3 * pi * x) .* exp.(-x)
y = true_f + randn(n) * 0.2

data = DataFrame(
    x = x,
    y = y,
    col_3 = cos.(2 * pi * x),
    col_4 = x.^2,
    col_5 = log.(x .+ 1),
    col_6 = randn(n) * 0.5,
    col_7 = sin.(5 * pi * x),
    col_8 = exp.(x),
    col_9 = y * 2 + randn(n) * 0.1,
    col_10 = x * 10
)

# Cubic has linear build into basis (so just x); 1 + x otherwise
b_formula = @formula(y ~ 1 + x)
cubic_formula = @formula(y ~ 1)
pc_formula = @formula(y ~ 1 + x)

# B-Spline
println("--- Fitting GAM with B-Splines ---")
gam_b_spline = GAM(b_formula, data, spline_type=:b_spline, n_knots=15)
@time fit!(gam_b_spline)
println("Optimal lambda: ", round(gam_b_spline.lambda, digits=4))
println("REML score: ", round(gam_b_spline.reml_score, digits=4))

# Cubic
println("--- Fitting GAM with Cubic Splines ---")
gam_cubic_spline = GAM(cubic_formula, data, spline_type=:cubic_spline, n_knots=15)
@time fit!(gam_cubic_spline)
println("Optimal lambda: ", round(gam_cubic_spline.lambda, digits=4))
println("REML score: ", round(gam_cubic_spline.reml_score, digits=4))


# PC
println("--- Fitting GAM with PC Splines ---")
gam_pc_spline = GAM(pc_formula, data, spline_type=:pc_spline, n_knots=15)
@time fit!(gam_pc_spline)
println("Optimal lambda: ", round(gam_pc_spline.lambda, digits=4))
println("REML score: ", round(gam_pc_spline.reml_score, digits=4))
