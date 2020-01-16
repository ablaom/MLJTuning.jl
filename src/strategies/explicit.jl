mutable struct Explicit <: TuningStrategy end

# models! returns all models in the range at once:
MLJTuning.models!(tuning::Explicit, model, history::Nothing,
                  state, verbosity) = state
MLJTuning.models!(tuning::Explicit, model::M, history,
                  state, verbosity) where M = 
    state[length(history) + 1:end]

function MLJTuning.default_n(tuning::Explicit, range)
    try
        length(range)
    catch MethodError
        10
    end
end
         
