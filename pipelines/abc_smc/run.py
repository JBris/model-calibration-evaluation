import arviz as az
from matplotlib import pyplot as plt
from os.path import join as path_join
import pymc as pm

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

    with pm.Model() as model:
        mu = pm.Uniform("mu", mu_lb, mu_ub)
        sigma = pm.Uniform("sigma", sigma_lb, sigma_ub)

        def simulator_model(rng, mu, sigma, size = None):
            return simulator_func(mu, sigma, n, seed)

        pm.Simulator(
            "Y_obs",
            simulator_model,
            params = (mu, sigma),
            distance = "gaussian",
            sum_stat = "sort",
            epsilon = 1,
            observed = obs_data,
        )

    output_dir = path_join(DATA_DIR, "abc_smc")
    make_dir(output_dir)

    with model:
        trace = pm.sample_smc(
            draws = 1000, 
            kernel = "abc",
            chains = 4,
            cores = -1,
            compute_convergence_checks = True,
            return_inferencedata = True,
            random_seed = seed,
            progressbar = True
        )

        textsize = 7
        for plot in ["trace", "rank_vlines", "rank_bars"]:
            az.plot_trace(trace, kind = plot, plot_kwargs = {"textsize": textsize})
            outfile = path_join(output_dir, f"{plot}.png")
            plt.tight_layout()
            plt.savefig(outfile)

        def __create_plot(trace, plot_func, plot_name, kwargs):
            plot_func(trace, **kwargs)
            outfile = path_join(output_dir, f"{plot_name}.png")
            plt.tight_layout()
            plt.savefig(outfile)
        
        kwargs = {"figsize": (12, 12), "scatter_kwargs": dict(alpha = 0.01), "marginals": True, "textsize": textsize}
        __create_plot(trace, az.plot_pair, "marginals", kwargs)

        kwargs = {"figsize": (12, 12), "textsize": textsize}
        __create_plot(trace, az.plot_violin, "violin", kwargs)

        kwargs = {"figsize": (12, 12), "textsize": 5}
        __create_plot(trace, az.plot_posterior, "posterior", kwargs)

        outfile = path_join(output_dir, "summary.csv")
        az.summary(trace).to_csv(outfile, index = False)
        
    outfile = path_join(output_dir, CONFIG_FILE)
    write_sim_config(outfile, config)

if __name__ == "__main__":
    main()