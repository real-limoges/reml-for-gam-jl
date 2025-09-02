# src/predict.jl

function predict(model::GAM, newdata::DataFrame; type="response")
    linear_formula = model.formula
    X_linear_new = modelmatrix(linear_formula, newdata)
    
    basis_matrices_new = []
    for s in model.smooths
        info = model.spline_info[s.term]
        x_smooth_new = newdata[!, s.term]
        B_new = Matrix{Float64}(undef, 0, 0)

        if s.spline_type == :b_spline
            B_new = b_spline_basis(x_smooth_new, info[:knots], info[:degree])
        elseif s.spline_type == :cubic_spline
            B_new = cubic_spline_basis(x_smooth_new, info[:knots])
        elseif s.spline_type == :pc_spline
            X_cubic_basis = cubic_spline_basis(x_smooth_new, info[:knots])
            B_new = X_cubic_basis * info[:transform_matrix]
        end
        push!(basis_matrices_new, B_new)
    end
    
    X_new = hcat(X_linear_new, basis_matrices_new...)
    eta_pred = X_new * model.beta
    
    if type == "link"
        return eta_pred
    elseif type == "response"
        link = canonicallink(model.family)
        return linkinv.(link, eta_pred)
    else
        error("type must be 'link' or 'response'")
    end
end
