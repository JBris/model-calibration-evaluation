import elfi
from matplotlib import pyplot as plt
import numpy as np
from os.path import join as path_join

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

    mu =  elfi.Prior('uniform', mu_lb, mu_ub - mu_lb, name = "mu")
    sigma =  elfi.Prior('uniform', sigma_lb, sigma_ub - sigma_lb, name = "sigma")

    def simulator_model(mu, sigma, size = None, batch_size = 1, random_state = None):
        return simulator_func(mu, sigma, n, seed)

    simulator = elfi.Simulator(simulator_model, mu, sigma, observed = obs_data, name = 'simulator')
    mean_stat = elfi.Summary(lambda y: np.mean(y), simulator, name = 'mean_stat')
    variance_stat = elfi.Summary(lambda y: np.var(y), simulator, name = 'variance_stat')

    euclidean_distance = elfi.Distance(
        lambda XA, XB: np.expand_dims(
            np.linalg.norm(XA.flatten() - XB.flatten()), axis = 0
        ), 
        mean_stat, variance_stat, name = 'euclidean_distance'
    )

    log_distance = elfi.Operation(np.log, euclidean_distance, name = 'log_distance')

    bounds = {}
    bounds["mu"] = (mu_lb, mu_ub)
    bounds["sigma"] = (sigma_lb, sigma_ub)

    initial_evidence = 1000
    n_evidence = 1000
    bolfi = elfi.BOLFI(
        log_distance, batch_size = 1, initial_evidence = initial_evidence, update_interval = int(initial_evidence / 10), 
        bounds = bounds, seed = seed
    )
    post = bolfi.fit(n_evidence = n_evidence)

    output_dir = path_join(DATA_DIR, "bolfi")
    make_dir(output_dir)

    for plot_func in [bolfi.plot_state, bolfi.plot_discrepancy, bolfi.plot_gp]:
        plot_func()
        outfile = path_join(output_dir, f"{plot_func.__name__}.png")
        plt.savefig(outfile)
        
    n_draws = 1000
    n_chains = 4
    sampler = "metropolis"
    result_BOLFI = bolfi.sample(
        n_draws, algorithm = sampler, n_chains = n_chains, 
    )

    for plot_func in [ result_BOLFI.plot_traces, result_BOLFI.plot_marginals ]:
        plot_func()
        outfile = path_join(output_dir, f"{plot_func.__name__}.png")
        plt.savefig(outfile)

    outfile = path_join(output_dir, CONFIG_FILE)
    write_sim_config(outfile, config)

if __name__ == "__main__":
    main()