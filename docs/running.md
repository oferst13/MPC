# Running experiments

## Important behavior

`main.py` is both the implementation module and the executable script. It has
no `if __name__ == "__main__"` guard, so importing it starts the configured
experiment.

An optimization run may:

- invoke SWMM hundreds or thousands of times;
- overwrite `.out`, `.rpt`, `.ini`, and hot-start files;
- overwrite a serialized policy;
- open a Matplotlib window; and
- run for hours.

Run the code from a disposable working copy until these behaviors are
refactored.

## Configure an event

Experiment selection is currently controlled in `cfg.py`:

```python
forecast_mode = '-swap.csv'
files = glob.glob(rain_path + '/*.csv')
cur_file = files[37]
```

Available forecast modes are:

| Value | Meaning |
| --- | --- |
| `-perfect.csv` | Sliding windows of observed rainfall |
| `-plusMin.csv` | Synthetic rainfall-depth uncertainty |
| `-swap.csv` | Synthetic depth and temporal-placement uncertainty |

The positional `files[n]` selection is fragile because filesystem enumeration
order is not a stable event identifier. Before running, print
`cfg.event_dates` and confirm it is the intended event:

```powershell
@'
import cfg
print(cfg.event_dates)
print(cfg.forecast_mode)
print(cfg.sim_days)
print(len(cfg.forecast_array))
'@ | python -
```

## Generate a new policy

In `main.py`, use:

```python
optimize = True
real_rain = True
```

Then run:

```powershell
$env:MPLBACKEND = "Agg"  # optional for a noninteractive run
python -u main.py
```

The GA logs every generation. The event policy is written beneath
`policies/worse/` using the event dates and forecast mode.

The code currently has no random seed, so independently generated policies and
metrics may differ.

## Replay an existing policy

To test the simulation without regenerating a policy, set:

```python
optimize = False
real_rain = True
```

Confirm that the expected policy exists:

```text
policies/worse/<event dates><forecast suffix>
```

The replay performs baseline and controlled simulations and prints:

1. the internal mass-balance diagnostic;
2. peak-flow reduction;
3. rainwater-availability reduction; and
4. `end` after successful completion.

## Verified smoke test

An isolated replay was completed using the historical Python 3.8 environment:

```text
Event:             2013-01-05 - 2013-01-06
Forecast:          Depth & swap
Peak reduction:    17.290906900658374%
Water reduction:   24.40617860468646%
Process result:    completed
```

SWMM completed with continuity errors of approximately `-0.006%` and `0.019%`.

## Reproducibility notes

The same event row in `results-mae.csv` contains:

```text
Flow_reduction-swap:   19.904442%
Water_reduction-swap:  36.631790%
```

These values do not match the verified replay above. Possible causes include:

- a different historical code revision;
- different initial or hot-start state;
- a policy generated with different stochastic GA outcomes;
- different file-enumeration order changing event selection;
- locally modified configuration; or
- differences in intermediate SWMM files.

The replay also printed an internal-model mass-balance error of `266.17%`.
Because SWMM's continuity error was small, this points to the custom
kinematic/tank diagnostic or its units rather than a failed SWMM run. Do not use
the replay as a numerical regression baseline until this discrepancy is
resolved.

## Recommended reproducibility workflow

For every reference event, archive:

- the Git commit;
- full configuration;
- Python and package versions;
- random seed;
- selected input and forecast filenames;
- initial tank and SWMM state;
- serialized valve policy;
- outfall hydrograph;
- tank storage, overflow, release, and supply series; and
- PFR, RAR, and both mass-balance diagnostics.

A future command-line runner should write this information into a manifest for
every experiment.
