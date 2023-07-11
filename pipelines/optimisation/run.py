import numpy as np
import optuna
from optuna.samplers import TPESampler
from os.path import join as path_join

from model_calibration_lib.gaussian_sim import (
    read_sim_config, simulator_func, load_data, DATA_DIR, INPUT_FILE, 
    CONFIG_FILE, make_dir, write_sim_config
)

def objective(trial, obs_data, mu_lb, mu_ub, sigma_lb, sigma_ub, n, seed):
    mu = trial.suggest_float("mu", mu_lb, mu_ub) 
    sigma = trial.suggest_float("sigma", sigma_lb, sigma_ub) 
    sim_data = simulator_func(mu, sigma, n, seed)
    discrepency = np.linalg.norm(sim_data - obs_data).item()
    return discrepency

def save_model(study, output_dir: str, config: dict):
    trial_out = path_join(output_dir, "trial_results.csv")
    trials_df = study.trials_dataframe()
    trials_df.sort_values("value", ascending = True).to_csv(trial_out, index = False)
    print(f"Trial results written to {trial_out}")

    def __plot_results(plot_func, plot_name: str):
        img_file = path_join(output_dir, f"{plot_name}.png")
        plot_func(study).write_image(img_file)

    # __plot_results(optuna.visualization.plot_contour, "contour")
    __plot_results(optuna.visualization.plot_edf, "edf")
    __plot_results(optuna.visualization.plot_optimization_history, "optimization_history")
    __plot_results(optuna.visualization.plot_parallel_coordinate, "parallel_coordinate")
    __plot_results(optuna.visualization.plot_param_importances, "param_importances")
    __plot_results(optuna.visualization.plot_slice, "slice")
    
    outfile = path_join(output_dir, CONFIG_FILE)
    write_sim_config(outfile, config)

def main():
    infile = path_join(DATA_DIR, INPUT_FILE)
    obs_data = load_data(infile)
    config = read_sim_config()
    n, seed = config["n"], config["seed"]

    sampler = TPESampler(
        consider_prior = True,
        prior_weight = 1.0,
        consider_endpoints = False,
        n_startup_trials = 10,
        seed = seed,
    )

    mu_lb, mu_ub, sigma_lb, sigma_ub = config["mu_lb"], config["mu_ub"], config["sigma_lb"], config["sigma_ub"]
    study = optuna.create_study(sampler = sampler, study_name = "gaussian", direction = "minimize")
    study.optimize(
        lambda trial: objective(trial, obs_data, mu_lb, mu_ub, sigma_lb, sigma_ub, n, seed), 
        n_trials = 500, n_jobs = -1, gc_after_trial = True, 
        show_progress_bar = True
    )

    output_dir = path_join(DATA_DIR, "optimisation")
    make_dir(output_dir)
    save_model(study, output_dir, config)

if __name__ == "__main__":
    main()