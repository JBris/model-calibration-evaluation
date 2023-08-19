from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.initial_designs.random_design import RandomDesign
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from emukit.experimental_design.experimental_design_loop import ExperimentalDesignLoop
import GPy
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as path_join
import pandas as pd

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

    space = ParameterSpace([
        ContinuousParameter('mu', mu_lb, mu_ub),
        ContinuousParameter('sigma', sigma_lb, sigma_ub)
    ])

    random_design = RandomDesign(space)
    initial_points_count = 100
    X_init = random_design.get_samples(initial_points_count)

    def target_function(input_rows):
        discrepencies = []
        for row in input_rows:
            mu = row[0]
            sigma = row[1]
            sim_data = simulator_func(mu, sigma, n, seed)
            discrepency = np.linalg.norm(sim_data - obs_data).item()            
            discrepencies.append([discrepency])
        
        return np.array(discrepencies)

    Y_init = target_function(X_init)

    gp = GPy.models.GPRegression(X_init, Y_init, GPy.kern.RBF(2), noise_var=1e-10)
    emulator = GPyModelWrapper(gp)

    ed = ExperimentalDesignLoop(space = space, model = emulator)
    ed.run_loop(target_function, 30)

    output_dir = path_join(DATA_DIR, "experimental_design")
    make_dir(output_dir)

    X_samples = random_design.get_samples(1000)

    discrepencies_mu, discrepencies_var = ed.model.predict(X_samples)
    df = pd.DataFrame(X_samples, columns = ["mu", "sigma"])
    df["discrepencies_mu"] = discrepencies_mu
    df["discrepencies_var"] = discrepencies_var
    df.sort_values("discrepencies_mu", inplace = True)
    outfile = path_join(output_dir, "discrepencies.csv")
    df.to_csv(outfile, index = False)

    for col in ["mu", "sigma"]:
        df.plot.scatter(x = col, y = "discrepencies_mu")
        outfile = path_join(output_dir, f"{col}.png")
        plt.savefig(outfile)

    outfile = path_join(output_dir, CONFIG_FILE)
    write_sim_config(outfile, config)

if __name__ == "__main__":
    main()