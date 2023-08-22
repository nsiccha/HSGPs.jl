module HSGPsLogDensityProblemsExt

using HSGPs, LogDensityProblems

LogDensityProblems.dimension(what::HSGP) = length(what)

end