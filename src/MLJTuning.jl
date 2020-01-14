module MLJTuning


## METHOD EXPORT

# defined in tuned_models.jl:
export Grid, TunedModel

# defined in strategies/:
export Explicit


## METHOD IMPORT

import MLJBase
using MLJBase
using Distributed
import ComputationalResources: CPU1, CPUProcesses,
    CPUThreads, AbstractResource

## INCLUDE FILES

include("utilities.jl")
include("tuning_strategy_interface.jl")
include("tuned_models.jl")
include("strategies/explicit.jl")
end

