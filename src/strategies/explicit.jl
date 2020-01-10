mutable struct Explicit <: TuningStrategy end

# models! returns all models in the range at once:
MLJTuning.models!(tuned_model::Explicit, history, state) = state # the range

function MLJTuning.default_n(tuning::Explicit, range)
    try
        length(range)
    catch MethodError
        10
    end
end
         
