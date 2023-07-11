from matplotlib import pyplot as plt
from os.path import join as path_join
from SALib import ProblemSpec

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

    problem = {
        'num_vars': 2,
        'names': ['mu', 'sigma'],
        'bounds': [
            [mu_lb, mu_ub],
            [sigma_lb, sigma_ub]
        ]
    }

    sp = ProblemSpec(problem)

    def wrapped_func(X):
        import numpy as np
        N, _ = X.shape
        discrepencies = np.empty(N)
        for i in range(N):
            mu, sigma = X[i, :]
            sim_data = simulator_func(mu, sigma, n, seed)
            discrepencies[i] = np.linalg.norm(sim_data - obs_data).item()
        return discrepencies

    (sp.sample_sobol(2**5)
    .evaluate(wrapped_func)
    .analyze_sobol())

    output_dir = path_join(DATA_DIR, "sobol_sa")
    make_dir(output_dir)

    sp_list = sp.to_df() 
    outfile = path_join(output_dir, "sobol_st.csv")
    sp_list[0].to_csv(outfile, index = False)
    outfile = path_join(output_dir, "sobol_s1.csv")
    sp_list[1].to_csv(outfile, index = False)
        
    outfile = path_join(output_dir, "sobol_bar.png")
    sp.plot()
    plt.tight_layout()
    plt.savefig(outfile)
    
    outfile = path_join(output_dir, "sobol_heatmap.png")
    sp.heatmap()
    plt.tight_layout()
    plt.savefig(outfile)
    
    outfile = path_join(output_dir, CONFIG_FILE)
    write_sim_config(outfile, config)

if __name__ == "__main__":
    main()