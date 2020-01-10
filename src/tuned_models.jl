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
                             full_report=true,
                             train_best=true)

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
                    n=default_n(tuning, range))

    range === nothing && error("You need to specify `range=...` unless "*
                               "`tuning isa Explicit`. ")
    model == nothing && error("You need to specify model=... "*
                              "If `tuning=Explicit()`, any model in the "*
                              "range will do. ")

    if model isa Deterministic
        tuned_model = DeterministicTunedModel(model, tuning, resampling,
                                       measure, weights, operation, range,
                                       train_best, repeats, n)
    elseif model isa Probabilistic
        tuned_model = ProbabilisticTunedModel(model, tuning, resampling,
                                       measure, weights, operation, range,
                                       train_best, repeats, n)
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

function update_history!(tuning, history, n, resampling_machine, state)
    j = length(history)
    models_exhausted = false
    while j < n && !models_exhausted
        models = models!(tuning, history, K, state)
        Δj = length(models)
        Δj == 0 && (models_exhausted = true)
        shortfall = n - Δj
        if models_exhausted && shortfall > 0 && verbosity > -1
            @warn "Supply of models pre-maturely exhausted. "
        end
        shortfall < 0 && (models = models[1:n - j])
        Δhistory = []
        # batch processing (TODO: parallize this!):
        for m in models
            resampling_machine.model = m
            fit!(resampling_machine)
            e = evaluate(resampling_machine)
            r = result(tuned_model.tuning, history, e)
            Δhistory = push!(Δhistory, (m, r))
        end
        history = vcat(history, Δhistory)
    end
end

function MLJBase.fit(tuned_model::EitherTunedModel, verbosity::Integer, X, y)
    tuning = tuned_model.tuning
    n = tuned_model.n
    n === Nothing && (n = default_n(tuning))
    batch_size = tuned_model.batch_size
    domain = tuned_model.range
    model = tuned_model.model

    # omitted: checks that measures are appropriate

    state = setup(tuning, model, range)

    # instantiate resampler (`model` to be replaced with mutated
    # clones during iteration below):
    resampler = Resampler(model=model,
                          resampling = tuned_model.resampling,
                          measure    = tuned_model.measure,
                          weights    = tuned_model.weights,
                          operation  = tuned_model.operation)
    resampling_machine = machine(resampler, X, y)

    history = []
    update_history!(tuning, history, n, resampling_machine, state)

    best_model = best(tuning, history)

    fitresult = machine(best_model, X, y)

    if tuned_model.train_best
        fit!(fitresult, verbosity=verbosity-1)
        prereport = (best_model=best_model, best_report=report(fitresult))
    else
        prereport = (best_model=best_model, best_report=missing)
    end

    report = merge(prereport, tuning_report(tuning, history, state))

    meta_state = (history, deepcopy(tuned_model), state)

    return fitresult, report, stuff
end

function MLJBase.update(tuned_model::EitherTunedModel, verbosity::Integer,
                        old_fitresult, old_meta_state, X, y)

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

        fitresult = machine(best_model, X, y)

        if tuned_model.train_best
            fit!(fitresult, verbosity=verbosity-1)
            prereport = (best_model=best_model, best_report=report(fitresult))
        else
            prereport = (best_model=best_model, best_report=missing)
        end

        report = merge(prereport, tuning_report(tuning, history, state))

        meta_state = (history, deepcopy(tuned_model), state)

        return fitresult, report, meta_state

    else

        return fit(tuned_model, verbosity, X, y)

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

