A work in progress


# Implementing a New Tuning Strategy

This document assumes familiarity with the [Evaluating Model
Performance](https://alan-turing-institute.github.io/MLJ.jl/dev/evaluating_model_performance/) and [Performance
  Measures](https://alan-turing-institute.github.io/MLJ.jl/dev/performance_measures/)
sections of the MLJ manual.

### Overview 

What follows is an overview of tuning in MLJ. After the overview is an
elaboration on those terms given in *italics*.

All tuning in MLJ is conceptualized as an iterative procedure, each
iteration corresponding to a performance *evaluation* of a single
*model*. Each such model is a mutation of a fixed *prototype*. In the
general case, this prototype is a composite model, i.e., a model with
other models as hyperparameters, and while the type of the prototype
is fixed, the types of the sub-models are allowed to vary.

When all iterations of the algorithm are complete, the optimal model
is selected based entirely on a *history* generated according to the
specified *tuning strategy*. Iterations are generally performed in
batches, which are evaluated in parallel (sequential tuning strategies
degenerating into semi-sequential strategies, unless the batch size is
one). At the beginning of each batch, both the history and an internal
*state* object are consulted, and, on the basis of the tuning
strategy, a new batch of models to be evaluated is generated. On the
basis of these evaluations, and the strategy, the history and internal
state are updated.

The tuning algorithm initializes the state object before iterations
begin, on the basis of the specific strategy and a user-specified
*range* object.

- Recall that in MLJ a *model* is an object storing the
  hyperparameters of some learning algorithm indicated by the name of
  the model type (e.g., `DecisionTreeRegressor`). Models do not
  store learned parameters.

- An *evaluation* is the value returned by some call to the
  `evaluate!` method, when passed the resampling strategy and
  performance measures specified by the user when specifying the
  tuning task. Recall that such a value is a named tuple of vectors
  with keys `measure`, `measurement`, `per_fold`, and
  `per_observation`. See [Evaluating Model
  Performance](https://alan-turing-institute.github.io/MLJ.jl/dev/evaluating_model_performance/)
  for details. Recall also that some measures in MLJ (e.g.,
  `cross_entropy`) report a loss (or score) for each provided
  observation, while others (e.g., `auc`) report only an aggregated
  value (the `per_observation` entries being recorded as
  `missing`). This and other behaviour can be inspected using trait
  functions. Do `info(rms)` to view the trait values for the `rms` loss, and
  see [Performance
  measures](https://alan-turing-institute.github.io/MLJ.jl/dev/performance_measures/)
  for details.
    
- The *history* is a vector of tuples generated by the tuning
  algorithm - one tuple per iteration - used to determine the optimal
  model and which also records other user-inspectable statistics that
  may be of interest - for example, evaluations of a measure (loss or
  score) different from one being explicitly optimized. Each tuple is
  of the form `(m, r)`, where `m` is a model instance and `r` is
  information
  about `m` extracted from an evaluation.

- A *tuning strategy* is an instance of some subtype `S <:
  TuningStrategy`, the name `S` (e.g., `Grid`) indicating the tuning
  algorithm to be applied. The fields of the tuning strategy - called
  *hyperparameters* - are those tuning parameters specific to the
  strategy that **do not refer to specific models or specific model
  hyperparameters**. So, for example, a default resolution to be used
  in a grid search is a hyperparameter of `Grid`, but the resolution
  to be applied to a *specific* model hyperparameter (such as the
  maximum depth of a decision tree) is **not**. This latter parameter
  would be part of the user-specified range object.

- A *range* is any object whose specification completes the
  specification of the tuning task, after the prototype, tuning
  strategy, resampling strategy, performance measure(s), and total
  iteration count are given - and is essentially the space of models
  to be searched. This definition is intentionally broad and the
  interface places no restriction on the allowed types of this
  object. As an example, the `Grid` tuning strategy type supports the
  following range objects:
  
  - one-dimensional `NumericRange` or `NominalRange` objects (these
  types are provided by MLJBase)
  
  - a tuple `(p, r)` where `p` is one of the above range objects, and
    `r` a resolution to overide the default `resolution` of the
    strategy
  
  - vectors of the above two objects

**Important note on history initialization:** The history is always
initialized to `nothing`, rather than an empty vector.
  
### Interface points for user input

Recall, for context, that in MLJ tuning is implemented as a model
wrapper. In setting up a tuning task, the user constructs an instance
of the `TunedModel` wrapper type, which has these principal fields:

- `model`: the prototype model instance mutated during tuning

- `tuning`: the tuning strategy, an instance of a concrete
  `TuningStrategy` subtype, such as `Grid`

- `resampling`: the resampling strategy used for performance
  evaluations, an instance of a concrete `ResamplingStrategy` subtype,
  such as `Holdout` or `CV`

- `measure`: a measure (loss or score) or vector of measures available
  to the tuning algorithm, the first of which is optimized in the
  common case of single-objective tuning strategies
  
- `range`: as defined above - roughly, the space of models to be searched

- `n`: the number of iterations (number of distinct models to be
  evaluated)
  
- `acceleration`: the computational resources to be applied (e.g.,
  `CPUProcesses()` for distributed computing and `CPUThreads()` for
  multithreaded processing)
  
- `acceleration_resampling`: the computational resources to be applied
  at the level of resampling (e.g., in cross-validation)
  

### Implementation requirements for new tuning strategies

#### Summary of functions

Several functions are part of the tuning strategy API:

- `setup`: for initialization of state (compulsory)

- `result`: for building each element of the history 

- `models!`: for generating batches of new models and updating the
  state (*Note:* The history is updated automatically) (compulsory)

- `best`: for extracting the optimal model (and its performance) from
  the history

- `tuning_report`: for selecting what to report to the user apart from
  the optimal model 

- `default_n`: to specify the number of models to be evaluated

- `result_type`: to declare the type of object returned by `result` method (for performance, optional)


These are outlined below, after discussing types.


#### The tuning strategy type

Each tuning algorithm must define a subtype of `TuningStrategy` whose
fields are the hyperparameters controlling the strategy that do not
directly refer to models or model hyperparameters. These would
include, for example, the default resolution of a grid search, or the
initial temperature in simulated annealing.

The algorithm implementation must include a keyword constructor with
defaults. Here's an example:

```julia
mutable struct Grid <: TuningStrategy
    resolution::Int
    acceleration::ComputationalResources.AbstractResource

end

# Constructor with keywords
Grid(; resolution=10, acceleration=MLJTuning.DEFAULT_RESOURCE[]) = 
    Grid(resolution, acceleration)
```


#### Range types

A type definition is required for each range object a tuning strategy
should like to handle. The following range types are available
out-of-the box (re-exported from MLJBase):

- The one-dimensional range types `NumericRange` and `OrdinalRange`
  (subtypes of `ParamRange`)

- `Vector{ParamRange}` for Cartesian products
  
Recall that `OrdinalRange` has a `values` field, while `NominalRange`
has the fields `upper`, `lower`, `scale`, `unit` and `origin`. The
`unit` field specifies a preferred length scale, while `origin` a
preferred "central value". These default to `(upper - lower)/2` and
`(upper + lower)/2`, respectively, in the bounded case (neither `upper
= Inf` nor `lower = -Inf`). The fields `origin` and `unit` are used in
generating grids for unbounded ranges but can also be used to assign
sensible one or two-parameter univariate pdf's to a specified range
(e.g., assign a shifted exponential with mean `lower + unit` to a
right-unbounded `NominalRange`).

A `ParamRange` object is always associated with a field name, stored
as `field`, but for composite models this might be a be a "nested
name", such as `:(atom.max_depth)`.

*Generating grids.** To generate a one-dimensional grid from a
`ParamRange` object `r`, use `iterator(r, n, [, rng])` where `n` is
the number of grid points (maybe less for integer grids, due to
rounding) and `rng` an optional random number generator to shuffle the
output. Query `OrdinalRange`, `NominalRange` and `iterator` doc
strings for further details. For multi-dimensional grids, use the
`unwind` function on the one-dimensional grids.


#### The `result` method: For declaring what parts of an evaluation goes into the history 

```julia
MLJTuning.result(tuning::MyTuningStrategy, history, e)
```

This method is for extracting from an evaluation `e` of some model `m`
(and possibly, through the `history`, previous values) the value of
`r` to be recorded in the corresponding tuple `(m, r)` of the
history. The fallback is

```julia
MLJTuning.result(tuning, history, e) = (measure=e.measure, measurement=e.measurement)
```

Note this is always a tuple of *vectors*, since multiple measures
can be specified.

The history must contain everything needed for the `best` method to
determine the optimal model, and everything needed by the
`report_history` method, which generates a report on tuning to the
user (for use in visualization, for example). These methods are
detailed below.


#### The `setup` method: To initialize state 

```julia
state = setup(tuning::MyTuningStrategy, model, range, verbosity)
```

The `setup` function is for initializing the mutable `state` of the
tuning algorithm (needed, by the algorithm's `models!` method; see
below) and an empty history object. The `state` generally stores, at
the least, the range or some processed version thereof. In
momentum-based gradient descent, for example, the state would include
the previous hyperparameter gradients, while in GP Bayesian
optimization, it would store the (evolving) Gaussian processes.

If a variable is to be reported as part of the user-inspectable
history, then it should be written to the history instead of stored in
state. An example of this might be the `temperature` in simulated
annealing.

The `verbosity` is an integer indicating the level of logging: `0`
means logging should be restricted to warnings, `-1`, completely
silent. 

The fallback for `setup` is:

```julia
setup(tuning::TuningStrategy, model, range, verbosity) = range
```

However, a tuning strategy will generally want to implement a `setup`
method for each range type it is going to support:

```julia 
MLJTuning.setup(tuning::MyTuningStrategy, model, range::RangeType1, verbosity) = ...
MLJTuning.setup(tuning::MyTuningStrategy, model, range::RangeType2, verbosity) = ...
etc.
```


#### The `models!` method: For generating model batches to evaluate

```julia
MLJTuning.models!(tuning::MyTuningStrategy, model, history, state, verbosity)
```

This is the core method of a new implementation. Given the existing
`history` and `state`, it must return a vector ("batch") of *new*
model instances to be evaluated. Any number of models can be returned
(and this includes an empty vector or `nothing`, if models have been
exhausted) and the evaluations will be performed in parallel (using
the mode of parallelization defined by the `acceleration` field of the
`TunedModel` instance). *An update of the history, performed
automatically under the hood, only occurs after these evaluations.*

Most sequential tuning strategies will want include the batch size as
a hyperparameter, which we suggest they call `batch_size`, but this
field is not part of the tuning interface. In tuning whatever number
of models are returned by `models!` get evaluated in parallel.

In a `Grid` tuning strategy, for example, `models!` returns a random
selection of `n - length(history)` models from the grid, so that
`models!` is called only once (in each call to
`MLJBase.fit(::TunedModel, ...)` or `MLJBase.update(::TunedModel,
...)`). In a bona fide sequential method which is generating models
non-deterministically (such as simulated annealing), `models!` might
return a single model, or return a small batch of models to make use
of parallelization (the method becoming "semi-sequential" in that
case). In sequential methods that generate new models
deterministically (such as those choosing models that optimize the
expected improvement of a surrogate statistical model) `models!` would
return a single model.

If the tuning algorithm exhausts it's supply of new models (because,
for example, there is only a finite supply) then `models!` should
return an empty vector. Under the hood, there is no fixed "batch-size"
parameter, and the tuning algororithm is happy to receive any number
of models.


#### The `best` method: To define what constitutes the "optimal model"

```julia
MLJTuning.best(tuning::MyTuningStrategy, history)
```

Returns the entry `(best_model, r)` from the history corresponding to
the optimal model `best_model`.

A fallback whose definition is given below may be used, *provided the
fallback for `result` detailed above has not been overloaded*. In this
fallback for `best`, the best model is the one optimizing performance
estimates for the first measure in the `TunedModel` field `measure`:

```julia
function best(tuning::TuningStrategy, history)
   measurements = [h[2].measurement[1] for h in history]
   measure = first(history)[2].measure[1]
   if orientation(measure) == :score
       measurements = -measurements
   end
   best_index = argmin(measurements)
   return history[best_index]
end
```

####  The `tuning_report` method: To build the user-accessible report

As with any model, fitting a `TunedModel` instance generates a
user-accessible report. In the case of tuning, the report is
constructed with this code:

```julia
report = merge((best_model=best_model, best_result=best_result, best_report=best_report,),
                tuning_report(tuning, history, state))
```

where:

- `best_model` is the optimal model instance

- `best_result` is the corresponding "result" entry in the history (e.g., performance evaluation)
 
- `best_report` is the report generated by fitting the optimal
model

- `tuning_report(::MyTuningStrategy, ...)` is a method the implementer
  may overload. It should return a named tuple. The fallback is to
  return the raw history:

```julia
MLJTuning.tuning_report(tuning, history, state) = (history=history,)
```

#### The `default_n` method: For declaring the default number of iterations

```julia
MLJTuning.default_n(tuning::MyTuningStrategy)
```

More precisely, the `methods!` method (which is allowed to return
mutliple models) is called until the number of models exceeds
`default_n(tuning)`, or `methods!` returns an empty list.

The fallback is 

```julia
MLJTuning.default_n(::TuningStrategy) = 10
```


### Implementation example: Search through explicit list 

The most rudimentary tuning strategy just evaluates every model in a
specified list, such lists constituting the only kind of supported
range. (In this special case `range` is an arbitrary iterator of models, which are `Probabilistic` or `Deterministic`, according to the type of the prototype `model`, which is otherwise ignored.) The fallback implementations for `result`,
`best` and `report_history` suffice.  Here's the complete
implementation:

```julia 
    
import MLJBase
    
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

```

For slightly less trivial example, see [here]().
> 
