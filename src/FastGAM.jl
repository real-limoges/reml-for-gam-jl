module FastGAM

using DataFrames
using Distributions
using FFTW
using GLM
using LinearAlgebra
using Optim
using Statistics
using StatsModels: FormulaTerm, Term, TupleTerm, modelmatrix, response

include("types.jl")
include("families.jl")
include("basis_functions.jl")
include("constructor.jl")
include("solvers.jl")
include("predict.jl")

export GAM, Smooth, fit!, predict


# --- Model Constructor and Fitting ---


function fit!(model::GAM; initial_lambda_logs = nothing)
    num_smooths = length(model.smooths)
    if isnothing(initial_lambda_logs)
        initial_lambda_logs = zeros(num_smooths)
    end
    
    objective = l -> reml_score_fn(l, model)
    result = optimize(objective, initial_lambda_logs, LBFGS(), Optim.Options(g_tol=1e-6))
    
    # Finalize model parameters
    model.lambdas = exp.(Optim.minimizer(result))
    model.reml_score = -Optim.minimum(result)
    
    S_pen = zeros(size(model.S))
    for (i, s) in enumerate(model.smooths)
        indices = model.param_indices[s.term]
        S_pen[indices, indices] = model.lambdas[i] * model.S[indices, indices]
    end
    
    C = model.X' * model.X + S_pen
    C_chol = cholesky(C)
    model.beta = C_chol \ (model.X' * model.y)

    # Compute and store summary statistics
    n = length(model.y)
    model.fitted_values = model.X * model.beta
    model.residuals = model.y - model.fitted_values
    C_inv = inv(C_chol)
    model.edf = tr(C_inv * (model.X' * model.X))
    model.scale = sum(model.residuals.^2) / (n - model.edf)
    model.vcov = C_inv * model.scale
    
    rss = sum(model.residuals.^2)
    tss = sum((model.y .- mean(model.y)).^2)
    r_squared = 1 - rss / tss
    model.deviance_explained = r_squared
    model.adj_r_squared = 1 - ( (1 - r_squared) * (n - 1) / (n - model.edf - 1) )

    return model
end
