mutable struct DeterministicTunedModel{T,M<:Deterministic,R} <: MLJBase.Deterministic
    model::M
    tuning::T  # tuning strategy
    resampling # resampling strategy
    measure
    weights::Union{Nothing,Vector{<:Real}}
    operation
    range::R
    train_best::Bool
    repeats::Int
    n::Union{Int,Nothing}
    acceleration::AbstractResource
end

mutable struct ProbabilisticTunedModel{T,M<:Probabilistic,R} <: MLJBase.Probabilistic
    model::M
    tuning::T  # tuning strategy
    resampling # resampling strategy
    measure
    weights::Union{Nothing,AbstractVector{<:Real}}
    operation
    range::R
    train_best::Bool
    repeats::Int
    n::Union{Int,Nothing}
    acceleration::AbstractResource
end

const EitherTunedModel{T,M} =
    Union{DeterministicTunedModel{T,M},ProbabilisticTunedModel{T,M}}

MLJBase.is_wrapper(::Type{<:EitherTunedModel}) = true

#todo update:
"""
    tuned_model = TunedModel(; model=nothing,
                             tuning=Grid(),
                             resampling=Holdout(),
                             measure=nothing,
                             weights=nothing,
                             repeats=1,
                             operation=predict,
                             ranges=ParamRange[],
                             n=default_n(tuning, range),
                             train_best=true,
                             acceleration=default_resource())

Construct a model wrapper for hyperparameter optimization of a
supervised learner.

Calling `fit!(mach)` on a machine `mach=machine(tuned_model, X, y)` or
`mach=machine(tuned_model, X, y, w)` will:

- Instigate a search, over clones of `model`, with the hyperparameter
  mutations specified by `ranges`, for a model optimizing the
  specified `measure`, using performance evaluations carried out using
  the specified `tuning` strategy and `resampling` strategy. If
  `measure` supports weights (`supports_weights(measure) == true`)
  then any `weights` specified will be passed to the measure.

- Fit an internal machine, based on the optimal model
  `fitted_params(mach).best_model`, wrapping the optimal `model`
  object in *all* the provided data `X, y` (or in `task`). Calling
  `predict(mach, Xnew)` then returns predictions on `Xnew` of this
  internal machine. The final train can be supressed by setting
  `train_best=false`.

Specify `repeats > 1` for repeated resampling per model evaluation. See
[`evaluate!](@ref) options for details.

*Important.* If a custom measure `measure` is used, and the measure is
a score, rather than a loss, be sure to check that
`MLJ.orientation(measure) == :score` to ensure maximization of the
measure, rather than minimization. Override an incorrect value with
`MLJ.orientation(::typeof(measure)) = :score`.

*Important:* If `weights` are left unspecified, and `measure` supports
sample weights, then any weight vector `w` used in constructing a
corresponding tuning machine, as in `tuning_machine =
machine(tuned_model, X, y, w)` (which is then used in *training* each
model in the search) will also be passed to `measure` for evaluation.

In the case of two-parameter tuning, a Plots.jl plot of performance
estimates is returned by `plot(mach)` or `heatmap(mach)`.

Once a tuning machine `mach` has bee trained as above, one can access
the learned parameters of the best model, using
`fitted_params(mach).best_fitted_params`. Similarly, the report of
training the best model is accessed via `report(mach).best_report`.

"""
function TunedModel(;model=nothing,
                    tuning=Explicit(),
                    resampling=Holdout(),
                    measures=nothing,
                    measure=measures,
                    weights=nothing,
                    operation=predict,
                    ranges=nothing,
                    range=ranges,
                    train_best=true,
                    repeats=1,
                    n=default_n(tuning, range),
                    acceleration=default_resource())

    range === nothing && error("You need to specify `range=...` unless "*
                               "`tuning isa Explicit`. ")
    model == nothing && error("You need to specify model=... .\n"*
                              "If `tuning=Explicit()`, any model in the "*
                              "range will do. ")

    if model isa Deterministic
        tuned_model = DeterministicTunedModel(model, tuning, resampling,
                                       measure, weights, operation, range,
                                              train_best, repeats, n,
                                              acceleration)
    elseif model isa Probabilistic
        tuned_model = ProbabilisticTunedModel(model, tuning, resampling,
                                       measure, weights, operation, range,
                                              train_best, repeats, n,
                                              acceleration)
    else
        error("Only `Deterministic` and `Probabilistic` "*
              "model types supported.")
    end

    message = clean!(tuned_model)
    isempty(message) || @info message

    return tuned_model

end

function MLJBase.clean!(model::EitherTunedModel)
    message = ""
    if model.measure === nothing
        model.measure = default_measure(model)
        message *= "No measure specified. Setting measure=$(model.measure). "
    end
    return message
end

function build_history!(tuning, n, resampling_machine,
                         state, verbosity)
end

function event(model, resampling_machine, verbosity, tuning, history)
    resampling_machine.model.model = model
    fit!(resampling_machine, verbosity=verbosity - 1)
    e = evaluate(resampling_machine)
    r = result(tuning, history, e)
    return model, r
end

# history is intialized to `nothing` because it's type is not known.
_vcat(history, Δhistory) = vcat(history, Δhistory)
_vcat(history::Nothing, Δhistory) = Δhistory

# models may return `nothing` insead of an empty list:

function MLJBase.fit(tuned_model::EitherTunedModel{T,M},
                     verbosity::Integer, args...) where {T,M}
    tuning = tuned_model.tuning
    n = tuned_model.n
    domain = tuned_model.range
    model = tuned_model.model
    range = tuned_model.range
    n === Nothing && (n = default_n(tuning, range))

    # omitted: checks that measures are appropriate

    state = setup(tuning, model, range)

    # instantiate resampler (`model` to be replaced with mutated
    # clones during iteration below):
    resampler = Resampler(model=model,
                          resampling = tuned_model.resampling,
                          measure    = tuned_model.measure,
                          weights    = tuned_model.weights,
                          operation  = tuned_model.operation)
    resampling_machine = machine(resampler, args...)

    j = 0 # model counter
    models_exhausted = false
    history = nothing
    while j < n && !models_exhausted
        _models = models!(tuning, model, history, state)
        models = _models === nothing ? M[] : collect(_models)
        @show models
        Δj = length(models)
        Δj == 0 && (models_exhausted = true)
        shortfall = n - Δj
        if models_exhausted && shortfall > 0 && verbosity > -1
            @warn "Supply of models exhausted before specified number of "*
            "models (`n=$n`) could be evaluated. "
        end
        Δj == 0 && break
        shortfall < 0 && (models = models[1:n - j])
        j += Δj

        # batch processing (TODO: parallelize next line):
        Δhistory = [event(m, resampling_machine, verbosity, tuning, history)
                    for m in models]
        @show Δhistory
        history = _vcat(history, Δhistory)
    end

    best_model = best(tuning, history)
    fitresult = machine(best_model, args...)

    if tuned_model.train_best
        fit!(fitresult, verbosity=verbosity - 1)
        prereport = (best_model=best_model,
                     best_report=MLJBase.report(fitresult))
    else
        prereport = (best_model=best_model, best_report=missing)
    end

    report = merge(prereport, tuning_report(tuning, history, state))
    meta_state = (history, deepcopy(tuned_model), state)

    return fitresult, meta_state, report
end

function MLJBase.update(tuned_model::EitherTunedModel, verbosity::Integer,
                        old_fitresult, old_meta_state, args...)

    history, old_tuned_model, state = old_meta_state

    n = tuned_model.n

    if isequal_except(tuned_model, old_tuned_model, :n)

        if tuned_model.n > old_tuned_model.n
            tuned_model.n = n - old_model.n # temporarily mutate tuned_model
            update_history!(tuning, history, n, resampling_machine, state)
            tuned_model.n = n # restore tuned_model to original state
        else
            verbosity < 1 || @info "Number of tuning iterations `n` lowered.\n"*
            "Truncating existing tuning history and retraining new best model."
        end
        best_model = best(tuning, history)

        fitresult = machine(best_model, args...)

        if tuned_model.train_best
            fit!(fitresult, verbosity=verbosity-1)
            prereport = (best_model=best_model, best_report=report(fitresult))
        else
            prereport = (best_model=best_model, best_report=missing)
        end

        report = merge(prereport, tuning_report(tuning, history, state))

        meta_state = (history, deepcopy(tuned_model), state)

        return fitresult, meta_state, report

    else

        return fit(tuned_model, verbosity, args...)

    end

end

MLJBase.predict(tuned_model::EitherTunedModel, fitresult, Xnew) =
    predict(fitresult, Xnew)

function MLJBase.fitted_params(tuned_model::EitherTunedModel, fitresult)
    if tuned_model.train_best
        return (best_model=fitresult.model,
                best_fitted_params=fitted_params(fitresult))
    else
        return (best_model=fitresult.model,
                best_fitted_params=missing)
    end
end


## METADATA

MLJBase.supports_weights(::Type{<:EitherTunedModel{<:Any,M}}) where M =
    MLJBase.supports_weights(M)

MLJBase.load_path(::Type{<:DeterministicTunedModel}) =
    "MLJTuning.DeterministicTunedModel"
MLJBase.package_name(::Type{<:EitherTunedModel}) = "MLJTuning"
MLJBase.package_uuid(::Type{<:EitherTunedModel}) = "MLJTuning"
MLJBase.package_url(::Type{<:EitherTunedModel}) =
    "https://github.com/alan-turing-institute/MLJTuning.jl"
MLJBase.is_pure_julia(::Type{<:EitherTunedModel{T,M}}) where {T,M} =
    MLJBase.is_pure_julia(M)
MLJBase.input_scitype(::Type{<:EitherTunedModel{T,M}}) where {T,M} =
    MLJBase.input_scitype(M)
MLJBase.target_scitype(::Type{<:EitherTunedModel{T,M}}) where {T,M} =
    MLJBase.target_scitype(M)

