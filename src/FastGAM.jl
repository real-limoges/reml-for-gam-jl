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
include("basis_functions.jl")

export GAM, Smooth, fit!, predict


# --- Model Constructor and Fitting ---


function GAM(formula::FormulaTerm, data::DataFrame, smooths::Vector{Smooth})
    """
    Writing this down because I'll inevitably forget.

    # Arguments
    FormulaTerm: Formula for the *linear* part of the model (e.g., @formula(y ~ 1 + x1)).
    - data::DataFrame: The input data.
    - smooths::Vector{Smooth}: A vector of Smooth objects defining the non-linear model components.
    """

    # Linear Part
    mf = ModelFrame(formula, data)
    y = response(mf)
    X_linear = modelmatrix(mf)
    
    basis_matrices = []
    penalty_matrices = []
    spline_info = Dict{Symbol, Dict{Symbol, Any}}()
    
    # Create the Smooth Parts
    for s in smooths
        x_smooth = data[!, s.term]
        knots = quantile(x_smooth, range(0, 1, length=s.n_knots))
        
        B, S_smooth, transform_matrix = Matrix{Float64}(undef,0,0), Matrix{Float64}(undef,0,0), nothing

        if s.spline_type == :b_spline
            B = b_spline_basis(x_smooth, knots, s.degree)
            p_smooth = size(B, 2)
            D = diff(diff(Matrix{Float64}(I, p_smooth, p_smooth), dims=1), dims=1)
            S_smooth = D' * D
        elseif s.spline_type == :cubic_spline
            B = cubic_spline_basis(x_smooth, knots)
            S_smooth = cubic_spline_penalty(knots)
        elseif s.spline_type == :pc_spline
            X_cubic_basis = cubic_spline_basis(x_smooth, knots)
            S_cubic_penalty = cubic_spline_penalty(knots)
            eig = eigen(S_cubic_penalty)
            transform_matrix = eig.vectors
            B = X_cubic_basis * transform_matrix
            S_smooth = Diagonal(max.(0, eig.values))
        else
            error("Unknown spline_type: $(s.spline_type) for term $(s.term)")
        end
        
        push!(basis_matrices, B)
        push!(penalty_matrices, S_smooth)
        spline_info[s.term] = Dict(:knots => knots, :degree => s.degree, :transform_matrix => transform_matrix)
    end
    
    # Assemble Full Model Matrix and Penalty Matrix
    X = hcat(X_linear, basis_matrices...)
    p_total = size(X, 2)
    S = zeros(p_total, p_total)
    
    param_indices = Dict{Symbol, UnitRange{Int}}()
    current_idx = 1
    
    # Indices for linear part
    p_linear = size(X_linear, 2)
    param_indices[:linear] = current_idx:(current_idx + p_linear - 1)
    current_idx += p_linear
    
    # Indices and penalty blocks for smooth parts
    for (i, s) in enumerate(smooths)
        p_smooth = size(basis_matrices[i], 2)
        idx_range = current_idx:(current_idx + p_smooth - 1)
        param_indices[s.term] = idx_range
        S[idx_range, idx_range] = penalty_matrices[i]
        current_idx += p_smooth
    end

    # Initialize GAM Object
    empty_vec = zeros(0); empty_mat = zeros(0,0)
    return GAM(formula, smooths, data, y, X, S, param_indices,
               zeros(p_total), zeros(length(smooths)),
               empty_vec, empty_vec, empty_mat, 0.0, 0.0, 0.0, 0.0, 0.0,
               spline_info)
end


function reml_score_fn(lambda_logs, model::GAM)
    lambdas = exp.(lambda_logs)
    
    # Construct the full penalty matrix from lambdas and block components
    S_pen = zeros(size(model.S))
    for (i, s) in enumerate(model.smooths)
        indices = model.param_indices[s.term]
        # model.S already contains the block-diagonal structure of individual penalties
        S_pen[indices, indices] = lambdas[i] * model.S[indices, indices]
    end
    
    C = model.X' * model.X + S_pen
    try
        C_chol = cholesky(C)
        beta = C_chol \ (model.X' * model.y)
        y_hat = model.X * beta
        n = length(model.y)
        
        # Calculate effective df for sigma2
        edf = tr(inv(C_chol) * (model.X' * model.X)) 
        sigma2 = sum(abs2, model.y - y_hat) / (n - edf)
        
        # Approximation to REML. Maybe switch that logdet to a QR Decomp
        log_det_C = logdet(C_chol)
        reml = -( (n - 1) * log(sigma2) + log_det_C + sum(abs2, model.y - y_hat) / sigma2 ) / 2
        return -reml
    catch e; return Inf; end
end

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

function predict(model::GAM, newdata::DataFrame)
    # Reconstruct Linear Part
    X_linear_new = modelmatrix(model.formula, newdata)
    
    basis_matrices = []
    # Reconstruct Each Smooth Part
    for s in model.smooths
        x_smooth = newdata[!, s.term]
        info = model.spline_info[s.term]
        
        B = Matrix{Float64}(undef, 0, 0)
        if s.spline_type == :b_spline
            B = b_spline_basis(x_smooth, info[:knots], info[:degree])
        elseif s.spline_type == :cubic_spline
            B = cubic_spline_basis(x_smooth, info[:knots])
        elseif s.spline_type == :pc_spline
            X_cubic_basis = cubic_spline_basis(x_smooth, info[:knots])
            B = X_cubic_basis * info[:transform_matrix]
        end
        push!(basis_matrices, B)
    end
    
    X_new = hcat(X_linear_new, basis_matrices...)
    return X_new * model.beta
end

end