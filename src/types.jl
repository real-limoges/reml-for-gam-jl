# src/types.jl

"""
Smooth(term::Symbol; spline_type=:b_spline, n_knots=20, degree=3): Defines a smooth term
"""
struct Smooth
    term::Symbol
    spline_type::Symbol
    n_knots::Int
    degree::Int
end
Smooth(term::Symbol; spline_type::Symbol=:b_spline, n_knots::Int=20, degree::Int=3) = Smooth(term, spline_type, n_knots, degree)


"""
Workspace: A mutable struct to hold pre-allocated arrays, allowing performance-critical loops.
"""
mutable struct Workspace
    S_pen::Matrix{Float64}      # Full penalized S
    W::Diagonal{Float64, Vector{Float64}} # Diagonal weight
    C::Matrix{Float64}          # (X'WX + S_pen)
    X_W_z::Vector{Float64}      # Stores X' * W * z
    X_W_X::Matrix{Float64}      # Stores X' * W * X
end
Workspace(p::Int, n::Int) = Workspace(
    zeros(p, p),
    Diagonal(zeros(n)),
    zeros(p, p),
    zeros(p),
    zeros(p, p)
)

"""
GAM: The main mutable struct that holds all information about GAM
"""
mutable struct GAM
    # Inputs
    formula::FormulaTerm
    smooths::Vector{Smooth}
    family::Distribution
    data::DataFrame
    y::Vector{Float64}
    X::Matrix{Float64}
    S::Matrix{Float64}
    param_indices::Dict{Symbol, UnitRange{Int}}
    
    # PIRLS working variables
    eta::Vector{Float64}
    mu::Vector{Float64}
    w::Vector{Float64}
    z::Vector{Float64}

    # Workspace for performance
    workspace::Workspace
    
    # Fitted parameters
    beta::Vector{Float64}
    lambdas::Vector{Float64}

    # Summary statistics
    fitted_values::Vector{Float64}
    residuals::Vector{Float64}
    vcov::Matrix{Float64}
    edf::Float64
    gcv_score::Float64
    
    # Information for prediction
    spline_info::Dict{Symbol, Dict{Symbol, Any}}
end
