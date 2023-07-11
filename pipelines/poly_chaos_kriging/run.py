from matplotlib import pyplot as plt
from os.path import join as path_join
import numpy as np
import numpy
import chaospy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LarsCV
import gstools

from model_calibration_lib.gaussian_sim import (
    read_sim_config, simulator_func, load_data, DATA_DIR, INPUT_FILE, 
    CONFIG_FILE, make_dir, write_sim_config
)

def model_sample(parameters, obs_data, n, seed):
    mu, sigma = parameters
    sim_data = simulator_func(mu, sigma, n, seed)
    discrepency = np.linalg.norm(sim_data - obs_data).item()
    return discrepency

def main():
    infile = path_join(DATA_DIR, INPUT_FILE)
    obs_data = load_data(infile)
    output_dir = path_join(DATA_DIR, "poly_chaos_kriging")
    make_dir(output_dir)
    config = read_sim_config()
    n, seed = config["n"], config["seed"]

    mu_lb, mu_ub, sigma_lb, sigma_ub = config["mu_lb"], config["mu_ub"], config["sigma_lb"], config["sigma_ub"]

    mu = chaospy.Uniform(mu_lb, mu_ub)
    sigma = chaospy.Uniform(sigma_lb, sigma_ub)
    joint = chaospy.J(mu, sigma)

    n_samples = 100
    expansion = chaospy.generate_expansion(10, joint, normed = True)
    samples = joint.sample(n_samples, rule = "sobol")
    evaluations = numpy.array([ model_sample(sample, obs_data, n, seed) for sample in samples.T ])

    lars = LarsCV(fit_intercept = False, max_iter = 5)
    pce, coeffs = chaospy.fit_regression(
        expansion, samples, evaluations, model = lars, retall = True
    )
    expansion_ = expansion[coeffs != 0]

    coordinates = np.linspace(evaluations.min(), evaluations.max(), n_samples)
    model = gstools.Gaussian(dim = 2, var = 10)
    pck = gstools.krige.Universal(model, samples, evaluations, list(expansion_))
    pck(samples)

    uk = gstools.krige.Universal(model, samples, evaluations, "linear")
    uk(samples)

    mu, sigma = pck.field, numpy.sqrt(pck.krige_var)
    plt.plot(coordinates, mu, label = "pck")
    plt.fill_between(coordinates, mu - sigma, mu + sigma, alpha = 0.5)

    mu, sigma = uk.field, numpy.sqrt(uk.krige_var)
    plt.plot(coordinates, mu, label = "uk")
    plt.fill_between(coordinates, mu - sigma, mu + sigma, alpha = 0.5)

    plt.legend(loc="upper left")

    outfile = path_join(output_dir, "poly_chaos_kriging.png")
    plt.tight_layout()
    plt.savefig(outfile)

    outfile = path_join(output_dir, CONFIG_FILE)
    write_sim_config(outfile, config)

if __name__ == "__main__":
    main()