using DataFrames, Random, GLM
include("FastGAM.jl")
using .FastGAM

Random.seed!(42)

# Generate Data
n = 1000

column_data = Dict{Symbol, Vector{Float64}}()

column_data[:x0] = collect(range(0, 1, length=n))
column_data[:x1] = rand(n)
column_data[:x2] = rand(n) * 2
for i in 3:9
    column_data[Symbol("x", i)] = randn(n)
end

true_f0 = sin.(3 * pi * column_data[:x0]) .* exp.(-column_data[:x0])
true_f1 = cos.(2 * pi * column_data[:x1])
linear_f2 = 2 * column_data[:x2]

column_data[:y] = true_f0 .+ true_f1 .+ linear_f2 .+ randn(n) * 0.3

data = DataFrame(column_data)


formula = @formula(y ~ x0 + x1 + x2 + x3)

# B-Spline
println("--- Fitting GAM with B-Splines ---")
gam_b_spline = GAM(formula, data, spline_type=:b_spline, n_knots=15)
@time FastGAM.fit!(gam_b_spline)
println("Optimal lambda: ", round(gam_b_spline.lambda, digits=4))
println("REML score: ", round(gam_b_spline.reml_score, digits=4))

# Cubic
println("--- Fitting GAM with Cubic Splines ---")
gam_cubic_spline = GAM(formula, data, spline_type=:cubic_spline, n_knots=15)
@time FastGAM.fit!(gam_cubic_spline)
println("Optimal lambda: ", round(gam_cubic_spline.lambda, digits=4))
println("REML score: ", round(gam_cubic_spline.reml_score, digits=4))


# PC
println("--- Fitting GAM with PC Splines ---")
gam_pc_spline = GAM(formula, data, spline_type=:pc_spline, n_knots=15)
@time FastGAM.fit!(gam_pc_spline)
println("Optimal lambda: ", round(gam_pc_spline.lambda, digits=4))
println("REML score: ", round(gam_pc_spline.reml_score, digits=4))
