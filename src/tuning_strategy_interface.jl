abstract type TuningStrategy <: MLJBase.MLJType end
show_as_constructed(::Type{<:TuningStrategy}) = true

# for initialization of state (compulsory)
setup(tuning::TuningStrategy, model::M, range) where M =
    range

# for building each element of the history:
result(tuning::TuningStrategy, history, e) =
    (measure=e.measure, measurement=e.measurement)

result_type(tuning, model) =
    NamedTuple{(:measure, :measurement),Tuple{Float64,Float64}}

# for generating batches of new models and updating the state (but not
# history):
function models! end

# for extracting the optimal model from the history:
function best(tuning::TuningStrategy, history)
   measurements = [h[2].measurement[1] for h in history]
   measure = first(history)[2].measure[1]
   if orientation(measure) == :score
       measurements = -measurements
   end
   best_index = argmin(measurements)
   return history[best_index][1]
end

# for selecting what to report to the user apart from the optimal
# model:
tuning_report(tuning::TuningStrategy, history, state) = (history=history,)

# for declaring the default number of models to evaluate:
default_n(tuning::TuningStrategy, range) = 10
