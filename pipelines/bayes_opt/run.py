import numpy as np
from os.path import join as path_join
from skopt import gp_minimize
import skopt.plots as opt_plts
from matplotlib import pyplot as plt

from model_calibration_lib.gaussian_sim import (
    read_sim_config, simulator_func, load_data, DATA_DIR, INPUT_FILE, 
    CONFIG_FILE, make_dir, write_sim_config
)

def main():
    infile = path_join(DATA_DIR, INPUT_FILE)
    obs_data = load_data(infile)
    config = read_sim_config()
    n, seed = config["n"], config["seed"]

    mu_lb, mu_ub, sigma_lb, sigma_ub = config["mu_lb"], config["mu_ub"], config["sigma_lb"], config["sigma_ub"]

    def objective(x):
        mu, sigma = x
        sim_data = simulator_func(mu, sigma, n, seed)
        discrepency = np.linalg.norm(sim_data - obs_data).item()
        return discrepency

    np.int = np.int64
    opt = gp_minimize(
        objective, [(mu_lb, mu_ub), (sigma_lb, sigma_ub)], acq_func = "EI",
        n_calls = 20, n_initial_points = 10, initial_point_generator = "lhs", verbose = True,
        n_jobs = -1, acq_optimizer = "lbfgs"
    )
    
    output_dir = path_join(DATA_DIR, "bayes_opt")
    make_dir(output_dir)

    for func in [
        opt_plts.plot_convergence, opt_plts.plot_evaluations, opt_plts.plot_objective, opt_plts.plot_regret
    ]:
        outfile = path_join(output_dir, f"{func.__name__}.png")
        func(opt)
        plt.savefig(outfile)

    outfile = path_join(output_dir, CONFIG_FILE)
    write_sim_config(outfile, config)

if __name__ == "__main__":
    main()