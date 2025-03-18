import numpy as np
from os.path import join as path_join
from evotorch import Problem
from evotorch.logging import PandasLogger
from evotorch.algorithms import GeneticAlgorithm, Cosyne, PyCMAES 
from evotorch.operators import OnePointCrossOver, GaussianMutation, PolynomialMutation 
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
    
    trials = []
    def objective(x):
        mu, sigma = x
        trial = dict(mu=mu.item(), sigma=sigma.item())
        sim_data = simulator_func(mu, sigma, n, seed)
        discrepency = np.linalg.norm(sim_data - obs_data).item()
        trial["discrepency"] = discrepency
        trials.append(trial)
        return [discrepency]

    problem = Problem(
        ["min"],
        objective,
        solution_length=2,
        bounds=([mu_lb, sigma_lb], [mu_ub, sigma_ub]),
        vectorized=False,
    )

    searcher = GeneticAlgorithm(
        problem,
        popsize=100,
        operators=[
            OnePointCrossOver(problem, tournament_size=4),
            GaussianMutation(problem, stdev=0.1),
            PolynomialMutation(problem, eta=20.0) 
        ],
    )

    # Can only handle single objectives?
    # searcher = PyCMAES (
    #     problem, 
    #     stdev_init=1,
    #     popsize=50,
    #     center_learning_rate=1.0,
    #     cov_learning_rate=1.0,
    #     rankmu_learning_rate=1.0,
    #     separable=True,
    #     obj_index=0
    # )
    # searcher = Cosyne(
    #     problem,
    #     num_elites = 1,
    #     popsize=50,  
    #     tournament_size = 4,
    #     mutation_stdev = 0.3,
    #     mutation_probability = 0.5,
    #     permute_all = True, 
    # )

    pandas_logger = PandasLogger(searcher, interval=50)
    searcher.run(num_generations=100)

    output_dir = path_join(DATA_DIR, "evolutionary")
    make_dir(output_dir)
    outfile = path_join(output_dir, CONFIG_FILE)
    write_sim_config(outfile, config)

    progress = pandas_logger.to_dataframe()
    progress.to_csv(path_join(output_dir, "trace.csv"), index=False)

    results = []
    names = ["mu", "sigma"]
    best_results = searcher.status["best"]
    worst_results = searcher.status["worst"]
    
    # Will be list of Solution objects for multi-objective
    for result in [best_results, worst_results]:
        row = {}
        for i, value in enumerate(result.values):
            name = names[i]
            row[name] = value.item()
        # Will be a list for multi-objective
        row["eval"] = result.evals.item()
        results.append(row)
        
    trials_df = pd.DataFrame(trials)
    trials_df.to_csv(path_join(output_dir, "trials.csv"), index=False)
    
if __name__ == "__main__":
    main()