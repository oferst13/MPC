# Architecture

## System data flow

```text
Observed rainfall ───────────────┐
                                │
Rainfall forecast ──┐           │
                    ▼           ▼
              SWMM prediction   Actual SWMM run
                    │           │
             lateral runoff     │
                    ▼           │
Tank clusters → kinematic routing model
       ▲                    │
       │                    ▼
candidate valve policy → predicted outfall Q(t)
       ▲                    │
       └── genetic algorithm ◄── objective Qobj
                                │
                                ▼
                      first hour of policy
                                │
                                ▼
                       updated system state
```

SWMM and the internal model have different roles. SWMM estimates runoff from
catchment surfaces that do not drain through the modeled RWH tanks. The
kinematic model routes that lateral runoff together with tank overflow and
controlled releases during optimization.

## Time scales

The defaults are defined in `cfg.py`:

| Quantity | Default |
| --- | ---: |
| Internal simulation step | 60 seconds |
| Rainfall input step | 10 minutes |
| Valve setting duration | 30 minutes |
| MPC sampling interval | 1 hour |
| Forecast/prediction/control horizon | 3 hours |

A three-hour policy therefore contains six decisions per tank cluster. The
decision vector contains 24 genes for the four-cluster case study. Only the
first two decisions per cluster are used before the next forecast update.

## Domain model

### Tanks

`Tank` represents a lumped cluster of identical tanks. It converts rainfall
depth to inflow volume using total connected roof area, supplies the configured
domestic demand, records overflow when capacity is exceeded, and calculates
orifice release using storage-dependent head.

Static case-study construction currently occurs near the bottom of `main.py`.
The cluster counts are 30, 35, 25, and 50 tanks. Each represented tank has a
nominal capacity of 10 m³.

### Drainage network

`Node` joins tank outlets, pipes, and SWMM lateral inflows. `Pipe` advances flow
using a kinematic-wave approximation based on pipe geometry, slope, Manning
roughness, and the one-minute time step.

The topology is assembled directly in `main.py`, ending at the `outfall` node.

### Scenario results

`Scenario` stores derived values for one simulation:

- maximum and complete outfall flow;
- last nonzero-flow and overflow positions;
- constant objective flow;
- objective-function value;
- supplied and remaining rainwater; and
- SWMM flow and peak.

## MPC cycle

For each forecast index, `main.py` performs the following sequence:

1. Reset tanks and pipes to the current cycle state.
2. Run `swmm_run_inflows()` with the forecast rainfall.
3. Attach the predicted lateral flows to network nodes.
4. Run the uncontrolled internal model to construct the baseline.
5. If flow and overflow thresholds are exceeded, run PyGAD.
6. Reshape the best gene vector into one row per tank cluster.
7. Retain the first hour of decisions in the event policy.
8. Reset to the current state and simulate one hour with observed rainfall.
9. Advance to the next forecast and repeat.

After all cycles, the policy is serialized below `policies/`.

## Genetic algorithm

`GA_params.py` defines the search:

- up to 250 generations;
- rank parent selection;
- uniform crossover;
- random mutation;
- elitism of two solutions; and
- termination after 75 generations without improvement.

The population includes four deterministic seeds: closed valves and constant
2.5%, 5%, and 10% openings. Remaining individuals are randomized.

`fitness_func()` resets the internal model, applies a candidate policy, runs the
three-hour prediction, and returns the reciprocal of the cumulative absolute
flow error. PyGAD therefore maximizes fitness while the scientific objective is
described as a minimization.

## Coupling and state concerns

The implementation depends on class registries such as `Tank.all_tanks`,
`Pipe.all_pipes`, and `Node.all_nodes`, as well as module globals including
`baseline`, `forecast_rain`, and `lat_flows`. This makes two simulations in one
process difficult to isolate and is the primary architectural constraint for a
future refactor.

The safe refactoring boundary is an explicit operation similar to:

```python
result = model.simulate(
    initial_state=state,
    rainfall=rainfall,
    lateral_inflows=lateral_inflows,
    valve_policy=policy,
)
```

Regression tests should be established before converting the current mutable
implementation to this interface.
