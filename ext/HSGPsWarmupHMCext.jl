module HSGPWarmupHMCext
   
using HSGPs, WarmupHMC

WarmupHMC.reparametrization_parameters(source::HSGP) = hcat(source.centeredness, source.mean_shift)
WarmupHMC.reparametrize(source::HSGP, parameters::AbstractMatrix) = HSGP(
    source.hyperprior,
    source.pre_eig,
    source.X,
    parameters[:, 1],
    parameters[:, 2]
)

@views WarmupHMC.reparametrize(source::HSGP, target::HSGP, draw::AbstractVector) = begin 
    sxic = draw[4:end]
    lsds = HSGPs.log_sds(source, draw)
    w = sxic .* exp.(lsds .* (1 .- source.centeredness))
    txic = w .* exp.(lsds .* (target.centeredness .- 1))
    trintercept = draw[1] + sum(w .* (target.mean_shift - source.mean_shift))
    vcat(trintercept, draw[2:3], txic)
end

@views WarmupHMC.lja(source::HSGP, target::HSGP, draw::AbstractVector) = begin 
    -sum(HSGPs.log_sds(source, draw) .* target.centeredness)
end

end