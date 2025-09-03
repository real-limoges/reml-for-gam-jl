# Solve the penalized weighted least squares system using Cholesky factorization.
# (X'WX + S_pen)beta = X'Wz

function update_coefficients!(beta::AbstractVector, ws::Workspace, model::GAM, ::Val{:gcv})
    try
        # Assemble matrices
        ws.W.diag .= model.w
        mul!(ws.X_W_X, model.X', ws.W * model.X)
        ws.C .= ws.X_W_X .+ ws.S_pen
        mul!(ws.X_W_z, model.X', ws.W * model.z)

        # Solve using Cholesky factorization
        # Symmetric() used so the 15th decimal place doesn't mess things up
        F = cholesky!(Symmetric(ws.C))
        beta .= F \ ws.X_W_z
        return true
    catch e
        # If factorization fails (e.g., not positive definite), return failure
        if isa(e, PosDefException); return false; end
        rethrow(e)
    end
end


# Solve the penalized weighted least squares system using FFT
function update_coefficients!(beta::AbstractVector, ws::Workspace, model::GAM, ::Val{:fft})
    # If we're able to run the fft solver, then X'WX becomes a banded toeplitz-like matrix

    # To do:
    # Constructing the required vectors for convolution based on model.w, model.z, and model.S
    # Performing FFTs, element-wise products/divisions, and an inverse FFT.
    # Then update beta and return bool
    throw(ErrorException("FFT solver is not yet implemented."))
end


# Performs one update step of the PIRLS algorithm.
# It updates weights w, pseudo-data z, solves for beta, and updates eta and mu.
function update_pirls!(model::GAM, link, lambdas, ws::Workspace, solver_val::Val)
    # Update weights and pseudo-data based on current mu and eta
    mu_eta_val = mueta.(link, model.eta)
    var_val = mu_variance.(model.family, model.mu)
    model.w .= mu_eta_val.^2 ./ var_val
    model.z .= model.eta .+ (model.y .- model.mu) ./ mu_eta_val
    
    # Assemble the combined penalty matrix S_pen
    ws.S_pen .= 0.0
    for (i, s) in enumerate(model.smooths)
        indices = model.param_indices[s.term]
        view(ws.S_pen, indices, indices) .+= lambdas[i] .* view(model.S, indices, indices)
    end
    
    successful_solve = update_coefficients!(model.beta, ws, model, solver_val)
    if !successful_solve
        return false
    end

    model.eta .= model.X * model.beta
    model.mu .= linkinv.(link, model.eta)
    
    return true
end


# The objective function for Optim.jl: calculates the GCV score for a given set of lambdas.
function gcv_score_fn(lambda_logs, model::GAM, link, ws::Workspace, solver_val::Val)
    lambdas = exp.(lambda_logs)
    max_iter, tol = 25, 1e-8
    
    # Initialize mu and eta for the first iteration
    model.mu .= (model.y .+ 0.5) ./ 2
    model.eta .= linkfun.(link, model.mu)
    
    local beta_old
    for _ in 1:max_iter
        beta_old = copy(model.beta)
        
        successful_update = update_pirls!(model, link, lambdas, ws, solver_val)
        if !successful_update
            return Inf # Return a high GCV score if the solver fails
        end
        
        if norm(model.beta - beta_old) < tol * (norm(model.beta) + tol)
            break
        end
    end
    
    # Calculate GCV score
    F = cholesky(Symmetric(ws.C)) # ws.C was updated inside update_coefficients! for the :gcv solver
    edf = tr(F \ ws.X_W_X)
    n = length(model.y)
    rss = sum(model.w .* (model.z .- model.eta).^2)
    gcv = (n * rss) / ((n - edf)^2)
    
    return gcv
end



# Main Fit Function
function fit!(model::GAM; 
              solver::Symbol = :gcv,
              initial_lambda_logs = nothing, 
              pirls_tol=1e-8, 
              max_pirls_iter=25)
              
    # If compatible then call it. B-splines equally spaced
    if solver === :fft
        # Check to make sure that it can support FFT (b splines + equally spaced grid) 
        # difference penalty S
    end
    
    num_smooths = length(model.smooths)
    if isnothing(initial_lambda_logs); initial_lambda_logs = zeros(num_smooths); end
    
    link = canonicallink(model.family)
    ws = model.workspace
    solver_val = Val(solver)

    # Optimize Smoothing Params using GCV
    objective = l -> gcv_score_fn(l, model, link, ws, solver_val)
    result = optimize(objective, initial_lambda_logs, LBFGS(), Optim.Options(g_tol = 1e-6))
    
    model.lambdas = exp.(Optim.minimizer(result))
    model.gcv_score = Optim.minimum(result)

    # Final PIRLS run with optimal lambdas
    
    # Re-initialize mu and eta
    model.mu .= (model.y .+ 0.5) ./ 2 
    model.eta .= linkfun.(link, model.mu)
    for _ in 1:max_pirls_iter
        beta_old = copy(model.beta)
        update_pirls!(model, link, model.lambdas, ws, solver_val)
        if norm(model.beta - beta_old) < pirls_tol * (norm(model.beta) + pirls_tol)
            break
        end
    end

    # Final summary statistics
    model.fitted_values .= model.mu
    model.residuals .= model.y .- model.mu
    
    # Re-compute C and C_unpen for edf and vcov
    # This part is specific to the matrix-based approach
    S_pen = ws.S_pen # S_pen was calculated in the last update_pirls! call
    W = Diagonal(model.w)
    C_unpen = model.X' * W * model.X
    C_pen = C_unpen + S_pen
    F_pen = cholesky(Symmetric(C_pen))

    model.edf = tr(F_pen \ C_unpen)
    model.vcov = inv(F_pen)
    
    return model
end