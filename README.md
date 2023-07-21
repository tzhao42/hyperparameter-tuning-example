# hyperparameter-tuning-example

Minimal example repo for hyperparameter tuning, with `optuna`, `ray[tune]`, and `syne_tune`.

Due to conflicting dependencies, we install each library in its own virtual environment.

To create the virtual environments:
```bash
cd scripts
./create_env_{optuna,ray,syne}.sh
```

Examples are located in `notebooks/`. To run a notebook, make sure to launch your Jupyter server in the proper virtual environment.
