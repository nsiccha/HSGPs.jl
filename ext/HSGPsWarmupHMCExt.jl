module HSGPsWarmupHMCExt
   
using HSGPs, WarmupHMC

WarmupHMC.reparametrization_parameters(source::HSGP) = vcat(source.centeredness, source.mean_shift)
WarmupHMC.reparametrize(source::HSGP, parameters::AbstractVector) = HSGP(
    source.hyperprior,
    source.pre_eig,
    source.X,
    collect.(eachcol(reshape(parameters, (:, 2))))...
)

# @views WarmupHMC.reparametrize(source::HSGP, target::HSGP, draw::AbstractVector) = begin 
#     sxic = draw[4:end]
#     lsds = HSGPs.log_sds(source, draw)
#     w = sxic .* exp.(lsds .* (1 .- source.centeredness))
#     txic = sxic .* exp.(lsds .* (target.centeredness .- source.centeredness))
#     trintercept = draw[1] + sum(w .* (target.mean_shift - source.mean_shift))
#     vcat(trintercept, draw[2:3], txic)
# end

# @views WarmupHMC.lja(source::HSGP, target::HSGP, draw::AbstractVector) = begin 
#     -sum(HSGPs.log_sds(source, draw) .* target.centeredness)
# end

@views WarmupHMC.lja_reparametrize(source::HSGP, target::HSGP, draw::AbstractVector, lja=0.) = begin  
    sxic = draw[4:end]
    lsds = HSGPs.log_sds(source, draw)
    w = sxic .* exp.(lsds .* (1 .- source.centeredness))
    txic = sxic .* exp.(lsds .* (target.centeredness .- source.centeredness))
    trintercept = draw[1] + sum(w .* (target.mean_shift - source.mean_shift))
    lja += -sum(HSGPs.log_sds(source, draw) .* target.centeredness)
    tdraw = vcat(trintercept, draw[2:3], txic)
    lja, tdraw
end

WarmupHMC.lpdf_and_invariants(source::HSGP, draw::AbstractVector) = logpdf_and_y(source, draw)

end