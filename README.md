# Model Predictive Control of Rainwater Harvesting Systems

Research code for simulating and optimizing real-time control of distributed
rainwater-harvesting (RWH) tanks in an urban drainage catchment.

The repository contains the implementation used in:

> Ofer Snir, Eran Friedler, and Luca Vezzaro, *Real-time Control of Rainwater
> Harvesting in Urban Catchments: A Model Predictive Control Approach under
> Forecasting Uncertainty*.

The model combines:

- a tank mass-balance model for rainfall capture, demand, overflow, and
  controlled release;
- EPA SWMM simulations for runoff from surfaces not connected to RWH tanks;
- an internal kinematic-wave model for routing tank and lateral flows;
- a genetic algorithm that chooses valve openings over a receding horizon; and
- event-level analysis of peak-flow reduction and retained rainwater.

## Research status

This branch preserves the original experimental implementation. It is runnable
with its historical Python environment, but it is not yet a packaged
application. Configuration is performed by editing module-level values, and
running `main.py` starts the experiment immediately.

For that reason, use an isolated working copy and read
[Running experiments](docs/running.md) before executing the model.

## Model summary

The case study represents 140 individual 10 m³ tanks as four controlled tank
clusters. At every hourly sampling instant, the controller:

1. receives a three-hour rainfall forecast and current tank/network state;
2. asks SWMM for predicted runoff from non-rooftop surfaces;
3. simulates the uncontrolled response and calculates a constant target flow
   `Qobj`;
4. uses a genetic algorithm to minimize the cumulative absolute difference
   between predicted outfall flow and `Qobj`;
5. produces six 30-minute valve settings for each tank cluster; and
6. applies only the first hour of the policy before repeating the process.

The optimization objective is:

```text
minimize Σ |Q(t) - Qobj|
```

See [Architecture](docs/architecture.md) for the code-level data flow.

## Repository map

| Path | Purpose |
| --- | --- |
| `main.py` | MPC loop, optimization, SWMM integration, evaluation, and plotting |
| `cfg.py` | Selected event, forecast scenario, time steps, demand, and SWMM files |
| `tank.py` | Tank storage, demand, overflow, and controlled release |
| `pipe.py` | Kinematic-wave pipe routing |
| `node.py` | Network connectivity and lateral-flow handling |
| `GA_params.py` | Genetic-algorithm settings and initial population |
| `clustered-no_roof*.inp` | SWMM models used at different MPC stages |
| `rain_files/` | Observed events and synthetic forecast scenarios |
| `policies/` | Serialized valve policies |
| `calc_results.py` | Batch result calculation |
| `plot_results.py` | Manuscript result plots |
| `results-mae.csv` | Event-level performance and forecast-error metrics |

## Historical environment

The code has been verified with:

```text
Python           3.8.2
NumPy            1.24.2
SciPy            1.10.1
pandas           1.5.3
Matplotlib       3.7.1
scikit-learn     1.2.2
PyGAD            2.19.2
PySWMM           1.3.0
```

Install the declared dependencies with:

```powershell
py -3.8 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The lower-bound-only requirements file does not guarantee compatibility with
modern Python or the newest package releases. In particular, the implementation
uses APIs that changed in later NumPy and SciPy releases.

## Outputs and performance metrics

For a controlled event, the code calculates:

```text
PFR = 100 × (baseline peak flow - controlled peak flow) / baseline peak flow

RAR = 100 × (baseline available water - controlled available water)
            / baseline available water
```

Saved policies are Python pickle files. Only load policies from a trusted
source.

## Known limitations

- Event selection currently depends on a positional file index in `cfg.py`.
- Windows-style relative paths are embedded in the implementation.
- Model construction and execution rely heavily on module-level mutable state.
- Importing `main.py` executes the experiment; it has no entry-point guard.
- Genetic-algorithm runs are not seeded and are therefore nondeterministic.
- A single event can require many expensive optimization cycles.
- Runtime SWMM files are generated in the current working directory.
- The present checkout does not exactly reproduce every value already stored in
  `results-mae.csv`; see [Running experiments](docs/running.md#reproducibility-notes).

These limitations are being retained here because this branch is the historical
research artifact. They should be addressed through regression-tested
refactoring rather than silent behavioral changes.

## License and data use

No explicit software or dataset license is currently included. Contact the
repository owner before redistributing or reusing the code or data outside the
terms applicable to the associated research.
