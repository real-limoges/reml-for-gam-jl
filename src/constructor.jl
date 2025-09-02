# src/constructor.jl
# Contains the main constructor for the GAM object.

"""
GAM(formula::FormulaTerm, data::DataFrame, smooths::Vector{Smooth}; family::Distribution=Normal()): Constructor for the GAM object.
"""
function GAM(formula::FormulaTerm, data::DataFrame, smooths::Vector{Smooth}; family::Distribution=Normal())
    # Linear Part
    mf = ModelFrame(formula, data)
    y = response(mf)
    X_linear = modelmatrix(mf)
    
    basis_matrices = []
    penalty_matrices = []
    spline_info = Dict{Symbol, Dict{Symbol, Any}}()
    
    # Setup Each Smooth Part
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
    
    # Assemble Matrices
    X = hcat(X_linear, basis_matrices...)
    p_total = size(X, 2)
    S = zeros(p_total, p_total)
    
    param_indices, current_idx = Dict{Symbol, UnitRange{Int}}(), 1
    p_linear = size(X_linear, 2)
    param_indices[:linear] = current_idx:(current_idx + p_linear - 1)
    current_idx += p_linear
    
    for (i, s) in enumerate(smooths)
        p_smooth = size(basis_matrices[i], 2)
        idx_range = current_idx:(current_idx + p_smooth - 1)
        param_indices[s.term] = idx_range
        S[idx_range, idx_range] = penalty_matrices[i]
        current_idx += p_smooth
    end

    # 4. Initialize GAM Struct
    n = length(y)
    workspace = Workspace(p_total, n)
    empty_vec = zeros(0); empty_mat = zeros(0,0)
    
    return GAM(formula, smooths, family, data, y, X, S, param_indices,
               zeros(n), zeros(n), zeros(n), zeros(n), # eta, mu, w, z
               workspace,
               zeros(p_total), zeros(length(smooths)), # beta, lambdas
               empty_vec, empty_vec, empty_mat, 0.0, 0.0, # fitted, resid, vcov, edf, gcv
               spline_info)
end
