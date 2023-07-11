import numpy as np
from os.path import join as path_join
from matplotlib import pyplot as plt
import spotpy
import matplotlib.pyplot as plt
from spotpy.likelihoods import gaussianLikelihoodMeasErrorOut as GaussianLik
from spotpy.analyser import plot_parameter_trace
from spotpy.analyser import plot_posterior_parameter_histogram
from spotpy.parameter import Uniform

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

    class SPOTSetup:
        mu = Uniform(low = mu_lb, high = mu_ub)
        sigma = Uniform(low = sigma_lb, high = sigma_ub)

        def __init__(self, obj_func = GaussianLik):
            self.obj_func = obj_func
  
        def simulation(self, x):
            mu, sigma = x
            sim_data = simulator_func(mu, sigma, n, seed)
            return sim_data

        def evaluation(self):
            return obs_data

        def objectivefunction(self, simulation, evaluation, params = None):
            like = self.obj_func(evaluation, simulation)
            return like
    
    spot_setup = SPOTSetup(GaussianLik)

    output_dir = path_join(DATA_DIR, "dream")
    make_dir(output_dir)

    dbname = path_join(output_dir, "dream_gauss")
    sampler = spotpy.algorithms.dream(spot_setup, dbname = dbname, dbformat = 'csv')

    rep = 1000 
    r_hat = sampler.sample(rep, nChains = 4, nCr = 3, delta = 1, c = 0.1, eps = 10e-6, convergence_limit = 1)

    results = spotpy.analyser.load_csv_results(dbname)
    fields=[word for word in results.dtype.names if word.startswith('sim')]

    fig= plt.figure(figsize=(16,9))
    ax = plt.subplot(1,1,1)
    q5,q25,q75,q95=[],[],[],[]
    for field in fields:
        q5.append(np.percentile(results[field][-100:-1],2.5))
        q95.append(np.percentile(results[field][-100:-1],97.5))
    ax.plot(q5,color='dimgrey',linestyle='solid')
    ax.plot(q95,color='dimgrey',linestyle='solid')
    ax.fill_between(np.arange(0,len(q5),1),list(q5),list(q95),facecolor='dimgrey',zorder=0,
                    linewidth=0,label='parameter uncertainty')  
    ax.plot(spot_setup.evaluation(),'r.',label='data')
    ax.legend()
    outfile = path_join(output_dir, "uncertainty.png")
    fig.savefig(outfile,dpi=300)

    outfile = path_join(output_dir, "convergence.png")
    spotpy.analyser.plot_gelman_rubin(results, r_hat, fig_name = outfile)

    parameters = spotpy.parameter.get_parameters_array(spot_setup)

    fig, ax = plt.subplots(nrows = 2, ncols = 2)
    for par_id in range(len(parameters)):
        plot_parameter_trace(ax[par_id][0], results, parameters[par_id])
        plot_posterior_parameter_histogram(ax[par_id][1], results, parameters[par_id])

    ax[-1][0].set_xlabel('Iterations')
    ax[-1][1].set_xlabel('Parameter range')
    outfile = path_join(output_dir, "parameters.png")
    fig.savefig(outfile, dpi = 300)

    outfile = path_join(output_dir, CONFIG_FILE)
    write_sim_config(outfile, config)

if __name__ == "__main__":
    main()