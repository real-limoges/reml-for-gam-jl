module FastGAM

using LinearAlgebra, Distributions, Optim, GLM, DataFrames

export GAM, fit!, predict

# mutable so it can be faster
mutable struct GAM
    formula::FormulaTerm
    data::DataFrame
    y::Vector{Float64}
    X::Matrix{Float64}
    S::Matrix{Float64} # Penalty matrix
    beta::Vector{Float64}
    lambda::Float64 # Smoothing parameter
    reml_score::Float64

    # Information for prediction
    spline_type::Symbol
    knots::Vector{Float64}
    degree::Int
    
    # Stores eigenvector matrix for PC splines
    pc_transform::Union{Matrix{Float64}, Nothing}
end

# B Splines
function b_spline_basis(x, knots, degree)
    n = length(x)
    k = length(knots)
    m = k + degree - 1

    # Create augmented knot vector
    aug_knots = [fill(knots[1], degree); knots; fill(knots[end], degree)]

    B = zeros(n, m)
    for i in 1:n
        for j in 1:m
            B[i, j] = b_spline_basis_element(x[i], j, degree, aug_knots)
        end
    end
    return B
end

function b_spline_basis_element(x, j, p, t)
    if p == 0
        return t[j] <= x < t[j+1] || (x == t[end] && j == length(t) - 1) ? 1.0 : 0.0
    end
    w1 = 0.0
    if t[j+p] - t[j] > 1e-9
        w1 = (x - t[j]) / (t[j+p] - t[j]) * b_spline_basis_element(x, j, p - 1, t)
    end
    w2 = 0.0
    if t[j+p+1] - t[j+1] > 1e-9
        w2 = (t[j+p+1] - x) / (t[j+p+1] - t[j+1]) * b_spline_basis_element(x, j + 1, p - 1, t)
    end
    return w1 + w2
end

# Cubics
function cubic_spline_basis(x, knots)
    n = length(x)
    k = length(knots)
    X = zeros(n, k + 2)
    X[:, 1] = x
    X[:, 2] = x.^2
    for j in 1:k
        X[:, j+2] = max.(0, x .- knots[j]).^3
    end
    return X
end


function cubic_spline_penalty(knots)
    k = length(knots)
    S = zeros(k + 2, k + 2)
    
    # The penalty applies only to the non-linear part
    S_k = zeros(k, k)
    for i in 1:k
        for j in 1:k
             S_k[i, j] = min(knots[i], knots[j])
        end
    end
    S[3:end, 3:end] = S_k
    return S
end


function GAM(formula::FormulaTerm, data::DataFrame;
             spline_type::Symbol=:b_spline, n_knots::Int=20, degree::Int=3)

    mf = ModelFrame(formula, data)
    y = response(mf)
    X_terms = modelmatrix(mf)

    smooth_term_symbol = Symbol(formula.rhs[end] |> string)
    x_smooth = data[!, smooth_term_symbol]

    knots = quantile(x_smooth, range(0, 1, length=n_knots))

    B = Matrix{Float64}(undef, 0, 0)
    S_smooth = Matrix{Float64}(undef, 0, 0)
    pc_transform = nothing

    if spline_type == :b_spline
        B = b_spline_basis(x_smooth, knots, degree)
        k = size(B, 2)

        D = diff(diff(Diagonal(ones(k)), dims=1), dims=1)        
        S_smooth = D' * D
    elseif spline_type == :cubic_spline
        if size(X_terms, 2) > 1
            X_terms = X_terms[:, 1:1] # Keep only intercept
        end
        B = cubic_spline_basis(x_smooth, knots)
        S_smooth = cubic_spline_penalty(knots)
    elseif spline_type == :pc_spline
        # Start with a cubic spline basis
        X_cubic = cubic_spline_basis(x_smooth, knots)
        S_cubic = cubic_spline_penalty(knots)

        # Eigen-decompose the penalty matrix
        eig = eigen(S_cubic)
        evals = eig.values
        evecs = eig.vectors

        # transformation matrix = eigenvectors
        pc_transform = evecs
        B = X_cubic * pc_transform

        # The penalty matrix is now diagonal with the eigenvalues
        # Use max to avoid small negative eigenvalues
        S_smooth = Diagonal(max.(0, evals)) 
    else
        error("Unknown spline_type: $spline_type")
    end

    # Combine fixed and smooth effects
    X = [X_terms B]

    # Pad penalty matrix
    n_fixed = size(X_terms, 2)
    S = zeros(size(X, 2), size(X, 2))
    S[n_fixed+1:end, n_fixed+1:end] = S_smooth

    return GAM(formula, data, y, X, S, zeros(size(X, 2)), 1.0, 0.0,
               spline_type, knots, degree, pc_transform)
end


function reml_score_fn(lambda_log, model::GAM)
    lambda = exp(lambda_log[1])
    C = model.X' * model.X + lambda * model.S

    try
        C_chol = cholesky(C)
        beta = C_chol \ (model.X' * model.y)
        y_hat = model.X * beta
        residuals = model.y - y_hat
        n = length(model.y)
        p = size(model.X, 2)
        sigma2 = sum(abs2, residuals) / (n - p)
        log_det_C = logdet(C_chol)

        # Probably should go from logdet(X'X) -> QR Decomposition
        log_det_XTX = logdet(model.X' * model.X + 1e-9I)
        reml = -((n - p) * log(2 * pi * sigma2) + sum(abs2, residuals) / sigma2 + log_det_C - log_det_XTX) / 2
        return -reml
    catch e
        isa(e, PosDefException) ? Inf : rethrow(e)
    end
end


function fit!(model::GAM; initial_lambda_log = 0.0)
    objective = lambda -> reml_score_fn(lambda, model)
    result = optimize(objective, [initial_lambda_log], LBFGS(), Optim.Options(g_tol=1e-6))
    lambda_log_opt = Optim.minimizer(result)
    model.lambda = exp(lambda_log_opt[1])
    model.reml_score = -Optim.minimum(result)
    C = model.X' * model.X + model.lambda * model.S
    C_chol = cholesky(C)
    model.beta = C_chol \ (model.X' * model.y)
    return model
end


function predict(model::GAM, newdata::DataFrame)
    mf = ModelFrame(model.formula, newdata)
    X_terms = modelmatrix(mf)

    smooth_term_symbol = Symbol(model.formula.rhs.terms[end] |> string)
    x_smooth = newdata[!, smooth_term_symbol]

    B = Matrix{Float64}(undef, 0, 0)

    if model.spline_type == :b_spline
        B = b_spline_basis(x_smooth, model.knots, model.degree)
    elseif model.spline_type == :cubic_spline
        if size(X_terms, 2) > 1
             X_terms = X_terms[:, 1:1]
        end
        B = cubic_spline_basis(x_smooth, model.knots)
    elseif model.spline_type == :pc_spline
        X_cubic = cubic_spline_basis(x_smooth, model.knots)
        B = X_cubic * model.pc_transform
    end

    X_new = [X_terms B]
    return X_new * model.beta
end

end