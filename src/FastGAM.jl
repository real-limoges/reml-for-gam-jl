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

end
