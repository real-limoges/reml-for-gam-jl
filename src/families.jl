# src/families.jl

# Use GLM's robust link functions
const Link = GLM.Link
const canonicallink = GLM.canonicallink
const linkfun = GLM.linkfun
const linkinv = GLM.linkinv
const mueta = GLM.mueta
const inverselink = GLM.inverselink

# Variance functions for different families
mu_variance(::Normal, mu) = 1.0
mu_variance(::Binomial, mu) = mu * (1.0 - mu)
mu_variance(::Poisson, mu) = mu
