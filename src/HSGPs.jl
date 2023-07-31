module HSGPs

export AbstractHSGP, HSGP, AHSGP, n_functions, y_and_logpdf, adapted

abstract type AbstractHSGP{T} end

struct HSGP{P,T} <: AbstractHSGP{T}
    hyperprior::P
    pre_eig::Vector{T}
    X::Matrix{T}
end
n_functions(hsgp::HSGP) = length(hsgp.pre_eig)


struct AHSGP{P,T} <: AbstractHSGP{T}
    hsgp::HSGP{P,T}
    centeredness::Vector{T}
    mean_shift::Vector{T}
end

# https://github.com/avehtari/casestudies/blob/967cdb3a6432e8985886b96fda306645fe156a29/Motorcycle/gpbasisfun_functions.stan#L12-L14
HSGP(hyperprior, boundary_factor, n_functions, x) = begin 
    idxs = 1:n_functions
    pre_eig = (-.25 * (pi/2/boundary_factor)^2) .* idxs .^ 2
    # sin(diag_post_multiply(rep_matrix(pi()/(2*L) * (x+L), M), linspaced_vector(M, 1, M)))/sqrt(L);
    X = sin.((x .+ boundary_factor) .* (pi/(2*boundary_factor)) .* idxs') ./ sqrt(boundary_factor)
    HSGP(hyperprior, pre_eig, X)
end

log_sds(hsgp::HSGP, parameters::AbstractVector) = log_sds(hsgp, parameters[2], parameters[3])
function log_sds(hsgp::HSGP, log_sigma, log_lengthscale, lengthscale=exp(log_lengthscale))
    # alpha * sqrt(sqrt(2*pi()) * rho) * exp(-0.25*(rho*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2);
    (log_sigma + .25 * log(2*pi) + .5 * log_lengthscale) .+ lengthscale^2 .* hsgp.pre_eig
end
@views compute_w(hsgp::HSGP, parameters::AbstractVector) = parameters[4:end] .* log_sds(hsgp, parameters)
@views y_and_logpdf(hsgp::HSGP, parameters::AbstractVector) = begin 
    xi = parameters[4:end]
    w = xi .* log_sds(hsgp, parameters)
    lpdf = sum(logpdf.(hsgp.hyperprior, parameters[1:3])) + sum(logpdf.(Normal(), xi))
    parameters[1] .+ hsgp.X * w, lpdf
end

@views y_and_logpdf(ahsgp::AHSGP, parameters::AbstractVector) = begin
    hsgp = ahsgp.hsgp 
    xic = parameters[4:end]
    lsds = log_sds(hsgp, parameters)
    w = xic .* exp.(lsds .* (1 .- ahsgp.centeredness))
    intercept = parameters[1] - dot(w, ahsgp.mean_shift)
    lpdf = logpdf(hsgp.hyperprior[1], intercept) + sum(logpdf.(hsgp.hyperprior[2:3], parameters[2:3])) + sum(logpdf.(Normal(0., exp.(lsds .* ahsgp.centeredness)), xic))
    intercept .+ hsgp.X * w, lpdf
end

adapted(hsgp::HSGP, parameters::AbstractMatrix, optimizer) = begin
    AHSGP(hsgp, optimal_centeredness(hsgp, parameters, optimizer), optimal_mean_shift(hsgp, parameters, optimizer))
end

@views optimal_centeredness(hsgp::HSGP, parameters::AbstractMatrix, optimizer; c0=zeros(n_functions(hsgp))) = begin 
    xi = parameters[4:end, :]
    lsds = log_sds.([hsgp], eachcol(parameters))
    transform(c) = xi .* exp.(lsds .* (1 .- c)) 
    loss(c) = mean(-lsds .* c .+ log.(std(transform(c), dims=2)))
    optimizer(loss, c0)
end 

@views optimal_mean_shift(hsgp::HSGP, parameters::AbstractMatrix, optimizer; s0=zeros(n_functions(hsgp))) = begin 
    intercept = parameters[1, :]
    W = hcat(compute_w.([hsgp], eachcol(parameters))...)'
    transform(s) = intercept .+ W * s
    loss(s) = log(std(transform(s)))
    optimizer(loss, s0)
end 


end