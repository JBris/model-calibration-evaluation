from matplotlib import pyplot as plt
from os.path import join as path_join
import numpy as np
from pygpc.AbstractModel import AbstractModel
import pygpc
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

from model_calibration_lib.gaussian_sim import (
    read_sim_config, simulator_func, load_data, DATA_DIR, INPUT_FILE, 
    CONFIG_FILE, make_dir, write_sim_config
)

class GaussianModel(AbstractModel):
    def __init__(self, obs_data , seed, n):
        super(type(self), self).__init__(matlab_model = False)
        self.obs_data = obs_data
        self.seed = seed
        self.n = n

    def validate(self):
        pass

    def simulate(self, process_id = None, matlab_engine = None):
        mu_arr, sigma_arr = self.p["mu"], self.p["sigma"]

        N = mu_arr.shape[0]
        discrepencies = np.empty(N)
        for i in range(N):
            mu, sigma = mu_arr[i], sigma_arr[i]
            sim_data = simulator_func(mu, sigma, self.n, self.seed)
            discrepencies[i] = np.linalg.norm(sim_data - self.obs_data).item()

        discrepencies = discrepencies[:, np.newaxis]
        return discrepencies
    
def main():
    infile = path_join(DATA_DIR, INPUT_FILE)
    obs_data = load_data(infile)
    config = read_sim_config()
    n, seed = config["n"], config["seed"]

    mu_lb, mu_ub, sigma_lb, sigma_ub = config["mu_lb"], config["mu_ub"], config["sigma_lb"], config["sigma_ub"]

    mu_sum = mu_lb + mu_ub
    parameters = OrderedDict()
    parameters["mu"] = pygpc.Norm(pdf_shape = [mu_sum / 2, mu_sum / 6])
    sigma_sum = sigma_lb + sigma_ub
    parameters["sigma"] = pygpc.Norm(pdf_shape = [sigma_sum / 2, sigma_sum / 6])
    model = GaussianModel(obs_data, seed, n)
    problem = pygpc.Problem(model = model, parameters = parameters)

    options = {}
    options["order_start"] = 5
    options["interaction_order"] = 2
    options["matrix_ratio"] = 3

    options["order_end"] = 20
    options["solver"] = "Moore-Penrose"
    options["order_max_norm"] = 1.0
    options["n_cpu"] = 0
    options["adaptive_sampling"] = False
    options["eps"] = 0.05
    options["fn_results"] = None
    options["basis_increment_strategy"] = "anisotropic"
    options["grid"] = pygpc.Random
    options["grid_options"] = { "seed": seed }

    algorithm = pygpc.RegAdaptive(problem = problem, options = options)
    session = pygpc.Session(algorithm = algorithm)
    session, coeffs, results = session.run()

    output_dir = path_join(DATA_DIR, "poly_chaos")
    make_dir(output_dir)

    lin_len = results.shape[0]
    pygpc.plot_gpc(
        session = session,
        coeffs = coeffs,
        random_vars = ["mu", "sigma"],
        output_idx = 0,
        n_grid = [100, 100],
        coords = np.vstack((np.linspace(mu_lb, mu_ub, lin_len) , np.linspace(sigma_lb, sigma_ub , lin_len))).T,
        results = results,
        fn_out = None
    )

    outfile = path_join(output_dir, "poly_chaos.png")
    plt.tight_layout()
    plt.savefig(outfile)

    outfile = path_join(output_dir, CONFIG_FILE)
    write_sim_config(outfile, config)

if __name__ == "__main__":
    main()