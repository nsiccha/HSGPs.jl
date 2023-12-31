module HSGPs

export HSGP, n_functions, logpdf_and_y

using Distributions, LogExpFunctions

abstract type AbstractHSGP{T} <: ContinuousMultivariateDistribution end

struct HSGP{P,T1,T2} <: AbstractHSGP{T1}
    hyperprior::P
    pre_eig::Vector{T1}
    X::Matrix{T1}
    centeredness::Vector{T2}
    mean_shift::Vector{T2}
end
n_functions(hsgp::HSGP) = length(hsgp.pre_eig)
Base.length(hsgp::HSGP) = 3 + n_functions(hsgp)

# https://github.com/avehtari/casestudies/blob/967cdb3a6432e8985886b96fda306645fe156a29/Motorcycle/gpbasisfun_functions.stan#L12-L14
HSGP(hyperprior::AbstractVector, x::AbstractVector, n_functions::Integer=32, boundary_factor::Real=1.5, centeredness=zeros(n_functions), mean_shift=zeros(n_functions)) = begin 
    idxs = 1:n_functions
    pre_eig = (-.25 * (pi/2/boundary_factor)^2) .* idxs .^ 2
    # sin(diag_post_multiply(rep_matrix(pi()/(2*L) * (x+L), M), linspaced_vector(M, 1, M)))/sqrt(L);
    X = sin.((x .+ boundary_factor) .* (pi/(2*boundary_factor)) .* idxs') ./ sqrt(boundary_factor)
    HSGP(hyperprior, pre_eig, X, centeredness, mean_shift)
end

log_sds(hsgp::HSGP, parameters::AbstractVector) = log_sds(hsgp, parameters[2], parameters[3])
function log_sds(hsgp::HSGP, log_sigma, log_lengthscale, lengthscale=exp(log_lengthscale))
    # alpha * sqrt(sqrt(2*pi()) * rho) * exp(-0.25*(rho*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2);
    (log_sigma + .25 * log(2*pi) + .5 * log_lengthscale) .+ lengthscale^2 .* hsgp.pre_eig
end

@views logpdf_and_y(hsgp::HSGP, parameters::AbstractVector) = begin
    xic = parameters[4:end]
    lsds = log_sds(hsgp, parameters)
    w = xic .* exp.(lsds .* (1 .- hsgp.centeredness))
    intercept = parameters[1] - sum(w .* hsgp.mean_shift)
    lpdf = logpdf(hsgp.hyperprior[1], intercept) + sum(logpdf.(hsgp.hyperprior[2:3], parameters[2:3])) + sum(logpdf.(Normal.(0., exp.(lsds .* hsgp.centeredness)), xic))
    lpdf, intercept .+ hsgp.X * w
end

struct DummyHSGP{P} <: AbstractHSGP{eltype(P)}
    hyperprior::P
end
n_functions(::DummyHSGP) = 0
Base.length(::DummyHSGP) = 1
logpdf_and_y(dhsgp::DummyHSGP, parameters::AbstractVector) = logpdf(dhsgp.hyperprior, parameters[1]), parameters[1]
adapted(dhsgp::DummyHSGP, args...; kwargs...) = dhsgp

end