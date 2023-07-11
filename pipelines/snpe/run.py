from matplotlib import pyplot as plt
import numpy as np
from os.path import join as path_join
import pandas as pd
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi import utils as utils
from sbi import analysis as analysis
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

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

    def simulator_model(theta):
        mu, sigma = theta
        return simulator_func(mu, sigma, n, seed)

    num_simulations = 300
    priors = [
        dist.Uniform(torch.tensor([mu_lb], dtype = torch.float64), torch.tensor([mu_ub], dtype = torch.float64)),
        dist.Uniform(torch.tensor([sigma_lb], dtype = torch.float64), torch.tensor([sigma_ub], dtype = torch.float64))
    ] 
    simulator, prior = prepare_for_sbi(simulator_model, priors)
    inference = SNPE(prior = prior)
    theta, x = simulate_for_sbi(simulator, proposal = prior, num_simulations = num_simulations)
    inference = inference.append_simulations(theta, x, data_device = "cpu")

    n_draws = 300
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)
    posterior.set_default_x(obs_data)
    posterior_samples = posterior.sample((n_draws,), x = obs_data)

    output_dir = path_join(DATA_DIR, "snpe")
    make_dir(output_dir)

    parameter_labels = ["mu", "sigma"]
    for plot_func in [analysis.pairplot, analysis.marginal_plot]:
        outfile = path_join(output_dir, f"{plot_func.__name__}.png")
        plt.rcParams.update({'font.size': 8})
        fig, _ = plot_func(posterior_samples, figsize = (20, 12), labels = parameter_labels)
        fig.savefig(outfile)

    limits = [(mu_lb, mu_ub), (sigma_lb, sigma_ub)]
    for plot_func in [analysis.conditional_pairplot, analysis.conditional_marginal_plot]:
        outfile = path_join(output_dir, f"{plot_func.__name__}.png")
        plt.rcParams.update({'font.size': 8})
        fig, _ = plot_func(
            density = posterior, condition = posterior.sample((1,)), figsize = (20, 12), 
            labels = parameter_labels, limits = limits
        )
        fig.savefig(outfile)

    thetas = prior.sample((n_draws,))
    xs = simulator(thetas)

    ranks, dap_samples = analysis.run_sbc(
        thetas, xs, posterior, num_posterior_samples = n_draws
    )

    check_stats = analysis.check_sbc(
        ranks, thetas, dap_samples, num_posterior_samples = n_draws
    )

    check_stats_processed = []
    for metric in check_stats:
        metric_dict = { "metric": metric }
        check_stats_processed.append(metric_dict)
        scores = check_stats[metric].detach().cpu().numpy()
        for i, score in enumerate(scores):
            col_name = parameter_labels[i]
            metric_dict[col_name] = score
    
    outfile = path_join(output_dir, "diagnostics.csv")
    pd.DataFrame(check_stats_processed).to_csv(outfile, index = False)
        
    outfile = path_join(output_dir, CONFIG_FILE)
    write_sim_config(outfile, config)

if __name__ == "__main__":
    main()