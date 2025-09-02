# src/basis_functions.jl

function b_spline_basis(x, knots, degree)
    n = length(x)
    aug_knots = [fill(knots[1], degree); knots; fill(knots[end], degree)]
    m = length(aug_knots) - degree - 1
    B = zeros(n, m)
    for i in 1:n, j in 1:m
        B[i, j] = b_spline_basis_element(x[i], j, degree, aug_knots)
    end
    return B
end

function b_spline_basis_element(x, j, p, t)
    if p == 0
        return t[j] <= x < t[j+1] || (x == t[end] && j == length(t) - p - 1) ? 1.0 : 0.0
    end
    w1 = 0.0
    w2 = 0.0
    if t[j+p] - t[j] > 1e-9
        w1 = (x - t[j]) / (t[j+p] - t[j]) * b_spline_basis_element(x, j, p - 1, t)
    end
    if t[j+p+1] - t[j+1] > 1e-9
        w2 = (t[j+p+1] - x) / (t[j+p+1] - t[j+1]) * b_spline_basis_element(x, j + 1, p - 1, t)
    end
    return w1 + w2
end

function cubic_spline_basis(x, knots)
    return hcat([max(0, val - k)^3 for val in x, k in knots])
end

function cubic_spline_penalty(knots)
    return [min(ki, kj) for ki in knots, kj in knots]
end
