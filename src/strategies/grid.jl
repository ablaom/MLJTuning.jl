const ParameterName=Union{Symbol,Expr}

"""
    Grid(goal=nothing, resolution=10, rng=Random.GLOBAL_RNG, shuffle=true)

Instantiate a Cartesian grid-based hyperparameter tuning strategy with
a specified `goal` for the number of grid points, or a specified
default `resolution` in each numeric dimension.

### Supported ranges:

- A single one-dimensional range (`ParamRange` object) `r`, or pair of
  the form `(r, res)` where `res` specifies a resolution to override
  the default `resolution`.

- Any vector of objects of the above form

`ParamRange` objects are constructed using the `range` method.

Example 1:

    range(model, :hyper1, lower=1, origin=2, unit=1)

Example 2:

    [(range(model, :hyper1, lower=1, upper=10), 15),
      range(model, :hyper2, lower=2, upper=4),
      range(model, :hyper3, values=[:ball, :tree]]

Note: All the `field` values of the `ParamRange` objects (`:hyper1`,
`:hyper2`, `:hyper3` in the precedng example) must refer to field
names a of single model (the `model` specified during `TunedModel`
construction).


### Algorithm

This is a standard grid search with the following specifics: In all
cases all `values` of each specified `NominalRange` are exhausted. If
`goal` is specified, then all resolutions are ignored, and a global
resolution is applied to the `NumericRange` objects that maximizes the
number of grid points, subject to the restriction that this not exceed
`goal`. Otherwise the default `resolution` and any parameter-specific
resolutions apply. In all cases the models generated are shuffled
using `rng`, unless `shuffle=false`.

See also [TunedModel](@ref), [range](@ref).

"""
mutable struct Grid <: TuningStrategy
    goal::Union{Nothing,Int}
    resolution::Int
    shuffle::Bool
    rng::Random.AbstractRNG
end

# Constructor with keywords
Grid(; goal=nothing, resolution=10, shuffle=true,
     rng=Random.GLOBAL_RNG) =
    Grid(goal, resolution, shuffle, rng)

"""
    process_user_range(user_specified_range, resolution, verbosity)

Utility to convert user-specified range (see [`Grid`](@ref)) into a
pair of tuples `(ranges, resolutions)`. 

For example, if `r1`, `r2` are `NumericRange`s and `s` is a
NominalRange`, then we have:

    julia> MLJTuning.process_user_range([(r1, 3), r2, s], 42, 1) ==
                            ((r1, r2, s), (3, 42, nothing))
    true

If `verbosity` > 0, then a warning is issued if a `Nominal` range is
paired with a resolution.  

"""
process_user_range(user_specified_range, resolution, verbosity) =
    process_user_range([user_specified_range, ], resolution, verbosity)
function process_user_range(user_specified_range::AbstractVector,
                    resolution, verbosity)
    stand(r) = throw(ArgumentError("Unsupported range. "))
    stand(r::NumericRange) = (r, resolution)
    stand(r::NominalRange) = (r, nothing)
    stand(t::Tuple{NumericRange,Integer}) = t
    function stand(t::Tuple{NominalRange,Integer})
        verbosity < 0 ||
            @warn  "Ignoring a resolution specified for a `NominalRange`. "
        return (first(t), nothing)
    end

    ret = zip(stand.(user_specified_range)...) |> collect
    return first(ret), last(ret)
end

function setup(tuning::Grid, model, user_range, verbosity)
    range, resolutions = process_user_range(user_range, tuning.resolution, verbosity)
    return range, resolution # temporary
end


# # for building each element of the history:
# result(tuning::TuningStrategy, history, e) =
#     (measure=e.measure, measurement=e.measurement)

# # for generating batches of new models and updating the state (but not
# # history):
# function models! end

# # for extracting the optimal model from the history:
# function best(tuning::TuningStrategy, history)
#    measurements = [h[2].measurement[1] for h in history]
#    measure = first(history)[2].measure[1]
#    if orientation(measure) == :score
#        measurements = -measurements
#        endusin
       
#    best_index = argmin(measurements)
#    return history[best_index][1]
# end

# # for selecting what to report to the user apart from the optimal
# # model:
# tuning_report(tuning::TuningStrategy, history, state) = (history=history,)

# # for declaring the default number of models to evaluate:
# default_n(tuning::TuningStrategy, range) = 10


# #######


# function MLJBase.fit(tuned_model::EitherTunedModel{Grid,M},
#                      verbosity::Integer, args...) where M

#     if tuned_model.ranges isa AbstractVector
#         ranges = tuned_model.ranges
#     else
#         ranges = [tuned_model.ranges,]
#     end

#     ranges isa AbstractVector{<:ParamRange} ||
#         error("ranges must be a ParamRange object or a vector of " *
#               "ParamRange objects. ")

#     # Build a vector of resolutions, one element per range. In case of
#     # NominalRange provide a dummy value of 5. In case of a dictionary
#     # with missing keys for the NumericRange`s, use fallback of 5.
#     resolution = tuned_model.tuning.resolution
#     if resolution isa Vector
#         val_given_field = Dict(resolution...)
#         fields = keys(val_given_field)
#         resolutions = map(ranges) do range
#             if range.field in fields
#                 return val_given_field[range.field]
#             else
#                 if range isa MLJBase.NumericRange && verbosity > 0
#                     @warn "No resolution specified for "*
#                     "$(range.field). Will use a value of 5. "
#                 end
#                 return 5
#             end
#         end
#     else
#         resolutions = fill(resolution, length(ranges))
#     end

#     if tuned_model.measure isa AbstractVector
#         measure = tuned_model.measure[1]
#         verbosity >=0 &&
#             @warn "Provided `meausure` is a vector. Using first element only. "
#     else
#         measure = tuned_model.measure
#     end

#     minimize = ifelse(orientation(measure) == :loss, true, false)

#     if verbosity > 0 && tuned_model.train_best
#         if minimize
#             @info "Mimimizing $measure. "
#         else
#             @info "Maximizing $measure. "
#         end
#     end

#     parameter_names = [string(r.field) for r in ranges]
#     scales = [scale(r) for r in ranges]

#     # We mutate a clone of the provided model but with any :rng field
#     # passed to the clone:
#     clone = deepcopy(tuned_model.model)
#     if isdefined(clone, :rng)
#         clone.rng = tuned_model.model.rng
#     end

#     resampler = Resampler(model=clone,
#                           resampling=tuned_model.resampling,
#                           measure=measure,
#                           weights=tuned_model.weights,
#                           operation=tuned_model.operation)

#     resampling_machine = machine(resampler, args...)

#     # tuple of iterators over hyper-parameter values:
#     iterators = map(eachindex(ranges)) do j
#         range = ranges[j]
#         if range isa MLJBase.NominalRange
#             MLJBase.iterator(range)
#         elseif range isa MLJBase.NumericRange
#             MLJBase.iterator(range, resolutions[j])
#         else
#             throw(TypeError(:iterator, "", MLJBase.ParamRange, range))
#         end
#     end

# #    nested_iterators = copy(tuned_model.ranges, iterators)

#     n_iterators = length(iterators) # same as number of ranges
#     A = MLJBase.unwind(iterators...)
#     N = size(A, 1)

#     if tuned_model.full_report
#         measurements = Vector{Float64}(undef, N)
#     end

#     # initialize search for best model:
#     best_model = deepcopy(tuned_model.model)
#     best_measurement = ifelse(minimize, Inf, -Inf)
#     s = ifelse(minimize, 1, -1)

#     # evaluate all the models using specified resampling:
#     # TODO: parallelize!

#     meter = Progress(N+1, dt=0, desc="Iterating over a $N-point grid: ",
#                      barglyphs=BarGlyphs("[=> ]"), barlen=25, color=:yellow)
#     verbosity != 1 || next!(meter)

#     for i in 1:N
#         verbosity != 1 || next!(meter)

#         A_row = Tuple(A[i,:])

#  #       new_params = copy(nested_iterators, A_row)

#         # mutate `clone` (the model to which `resampler` points):
#         for k in 1:n_iterators
#             field = ranges[k].field
#             recursive_setproperty!(clone, field, A_row[k])
#         end

#         if verbosity == 2
#             fit!(resampling_machine, verbosity=0)
#         else
#             fit!(resampling_machine, verbosity=verbosity-1)
#         end
#         e = evaluate(resampling_machine).measurement[1]

#         if verbosity > 1
#             text = prod("$(parameter_names[j])=$(A_row[j]) \t" for j in 1:length(A_row))
#             text *= "measurement=$e"
#             println(text)
#         end

#         if s*(best_measurement - e) > 0
#             best_model = deepcopy(clone)
#             best_measurement = e
#         end

#         if tuned_model.full_report
#  #           models[i] = deepcopy(clone)
#             measurements[i] = e
#         end

#     end

#     fitresult = machine(best_model, args...)
#     if tuned_model.train_best
#         verbosity < 1 || @info "Training best model on all supplied data."

#         # train best model on all the data:
#         # TODO: maybe avoid using machines here and use model fit/predict?
#         fit!(fitresult, verbosity=verbosity-1)
#         best_report = fitresult.report
#     else
#         verbosity < 1 || @info "Training of best model suppressed.\n "*
#         "To train tuning machine `mach` on all supplied data, call "*
#         "`fit!(mach.fitresult)`."
#         fitresult = tuned_model.model
#         best_report = missing
#     end

#     pre_report = (parameter_names= permutedims(parameter_names), # row vector
#                   parameter_scales=permutedims(scales),   # row vector
#                   best_measurement=best_measurement,
#                   best_report=best_report)

#     if tuned_model.full_report
#         report = merge(pre_report,
#                        (parameter_values=A,
#                         measurements=measurements,))
#     else
#         report = merge(pre_report,
#                        (parameter_values=missing,
#                         measurements=missing,))
#     end

#     cache = nothing

#     return fitresult, cache, report

# end 
