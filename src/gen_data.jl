using CSV
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

function main()
    n_pts = 10_000
    data_file = "cached_data.csv"
    data = simulate_gam_data(n_pts)
    println(data)
    CSV.write(data_file, data)
end

main()