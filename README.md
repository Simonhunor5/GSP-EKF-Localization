# GSP-EKF Localization

Comparing Extended Kalman Filter methods for robot localization with range measurements:
- **(A) Fixed R**: Standard EKF with constant measurement covariance
- **(B) Heuristic Adaptive R**: Per-anchor weighting based on residual history
- **(C) GSP Smoothing**: Graph Signal Processing with learned Laplacian

## Project Structure

```
GSP/
├── src/
│   ├── __init__.py
│   ├── utils.py         # Utility functions (angle wrapping)
│   ├── models.py        # Motion & measurement models
│   ├── ekf.py           # EKF predict/update
│   ├── adaptive.py      # Method B: Heuristic weighting
│   ├── gsp.py           # Method C: Laplacian learning & smoothing
│   ├── experiment.py    # Experiment runner & config
│   └── plotting.py      # Visualization
├── run_experiment.py    # Main entry point
├── analyze_results.py   # Analyze sweep results
├── generate_figures.py  # Generate figures
├── results/             # Output JSON files
└── README.md
```

## Usage

### Single Experiment (default parameters)
```bash
python run_experiment.py --plot
```

### Custom Parameters
```bash
python run_experiment.py --sigma_r 0.25 --outlier_prob 0.1 --heuristic_beta 3.0 --plot
```

### Parameter Sweep
Run a series of experiments with varying random seeds:
```bash
python run_experiment.py --sweep
```

### Analyze Results
Calculate statistics from the sweep results:
```bash
python analyze_results.py
```

### Generate Figures
Create publication-quality figures for the LaTeX paper:
```bash
python generate_figures.py
```

### All Options
```
--seed              Random seed (default: 0)
--T                 Number of time steps (default: 350)
--dt                Time step (default: 0.1)

--sigma_r           Measurement noise std (default: 0.18)
--outlier_prob      Outlier probability (default: 0.06)
--outlier_sigma     Outlier noise std (default: 1.2)

--heuristic_T_win   Window size for Method B (default: 15)
--heuristic_beta    Sensitivity for Method B (default: 2.0)

--gsp_T_win         Window size for Method C (default: 20)
--gsp_learn_every   Learning frequency for L (default: 10)
--gsp_alpha         Regularization for L (default: 0.01)
--gsp_mu            Temporal smoothness for L (default: 0.2)
--gsp_gamma         Smoothing strength (default: 1.0)

--plot              Show plots
--sweep             Run parameter sweep
--output, -o        Output directory (default: results)
```

## Output

Results are saved as JSON in the `results/` directory:
- `experiment_YYYYMMDD_HHMMSS.json` - Single run metrics
- `sweep_YYYYMMDD_HHMMSS.json` - Parameter sweep results

## Requirements

- numpy
- matplotlib
- cvxpy (optional, for Method C)

Install with:
```bash
pip install numpy matplotlib cvxpy
```
