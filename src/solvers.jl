# src/solvers.jl

# PIRLS / GCV Solver
function gcv_score_fn(lambda_logs, model::GAM, link, ws::Workspace)
    lambdas = exp.(lambda_logs)
    max_iter, tol = 25, 1e-8
    
    model.mu .= (model.y .+ 0.5) ./ 2
    model.eta .= linkfun.(link, model.mu)
    
    local beta
    for _ in 1:max_iter
        beta_old = copy(model.beta)
        
        mu_eta_val = mueta.(link, model.eta)
        var_val = mu_variance.(model.family, model.mu)
        model.w .= mu_eta_val.^2 ./ var_val
        model.z .= model.eta .+ (model.y .- model.mu) ./ mu_eta_val
        
        ws.S_pen .= 0.0
        for (i, s) in enumerate(model.smooths)
            indices = model.param_indices[s.term]
            view(ws.S_pen, indices, indices) .= lambdas[i] .* view(model.S, indices, indices)
        end
        
        ws.W.diag .= model.w
        mul!(ws.X_W_X, model.X', ws.W * model.X)
        ws.C .= ws.X_W_X .+ ws.S_pen
        mul!(ws.X_W_z, model.X', ws.W * model.z)

        try
            beta = cholesky!(ws.C) \ ws.X_W_z
        catch e; return Inf; end

        model.eta .= model.X * beta
        model.mu .= linkinv.(link, model.eta)
        
        if norm(beta - beta_old) < tol * (norm(beta) + tol)
            model.beta .= beta
            break
        end
        model.beta .= beta
    end
    
    F = cholesky(ws.C)
    edf = tr(F \ ws.X_W_X)
    n = length(model.y)
    rss = sum(model.w .* (model.z .- model.eta).^2)
    gcv = (n * rss) / ((n - edf)^2)
    
    return gcv
end

function fit!(model::GAM; initial_lambda_logs = nothing, pirls_tol=1e-8, max_pirls_iter=25)
    num_smooths = length(model.smooths)
    if isnothing(initial_lambda_logs); initial_lambda_logs = zeros(num_smooths); end
    
    link = canonicallink(model.family)
    objective = l -> gcv_score_fn(l, model, link, model.workspace)
    result = optimize(objective, initial_lambda_logs, LBFGS(), Optim.Options(g_tol = 1e-6))
    
    model.lambdas = exp.(Optim.minimizer(result))
    model.gcv_score = Optim.minimum(result)

    # Final PIRLS run
    model.mu .= (model.y .+ 0.5) ./ 2
    model.eta .= linkfun.(link, model.mu)
    for _ in 1:max_pirls_iter
        beta_old = copy(model.beta)
        mu_eta_val = mueta.(link, model.eta)
        var_val = mu_variance.(model.family, model.mu)
        model.w .= mu_eta_val.^2 ./ var_val
        model.z .= model.eta .+ (model.y .- model.mu) ./ mu_eta_val
        
        S_pen = zeros(size(model.S))
        for (i, s) in enumerate(model.smooths)
            indices = model.param_indices[s.term]
            S_pen[indices, indices] = model.lambdas[i] * model.S[indices, indices]
        end
        
        W = Diagonal(model.w)
        C = model.X' * W * model.X + S_pen
        F = cholesky(C)
        model.beta = F \ (model.X' * W * model.z)
        model.eta = model.X * model.beta
        model.mu = linkinv.(link, model.eta)
        
        if norm(model.beta - beta_old) < pirls_tol * (norm(model.beta) + pirls_tol); break; end
    end

    # Final summary statistics
    model.fitted_values .= model.mu
    model.residuals .= model.y .- model.mu
    
    S_pen = zeros(size(model.S))
    for (i,s) in enumerate(model.smooths)
        indices = model.param_indices[s.term]
        S_pen[indices, indices] = model.lambdas[i] * model.S[indices, indices]
    end
    
    W = Diagonal(model.w)
    C_unpen = model.X' * W * model.X
    F_pen = cholesky(C_unpen + S_pen)
    model.edf = tr(F_pen \ C_unpen)
    model.vcov = inv(F_pen)
    
    return model
end
