from ax import Data, Experiment, ParameterType, RangeParameter, SearchSpace
from ax.modelbridge.cross_validation import cross_validate
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner
from matplotlib import pyplot as plt
import numpy as np
from os.path import join as path_join
import pandas as pd
import torch

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

    class EuclideanObjective(NoisyFunctionMetric):
        def f(self, x: np.ndarray) -> float:
            mu, sigma = x
            sim_data = simulator_func(mu, sigma, n, seed)
            discrepency = np.linalg.norm(sim_data - obs_data).item()
            return discrepency

    torch.manual_seed(seed)  
    tkwargs = {
        "dtype": torch.double, 
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    search_space = SearchSpace(
        parameters = [
            RangeParameter(
                name = "mu", parameter_type = ParameterType.FLOAT, lower = mu_lb, upper = mu_ub
            ),  
            RangeParameter(
                name = "sigma", parameter_type = ParameterType.FLOAT, lower = sigma_lb, upper = sigma_ub
            ) 
        ]
    )

    optimization_config = OptimizationConfig(
        objective = Objective(
            metric = EuclideanObjective(
                name = "euclidean_distance", 
                param_names = ["mu", "sigma"], 
                noise_sd = None,  
            ),
            minimize = True,
        )
    )

    N_INIT = 10
    BATCH_SIZE = 8
    N_BATCHES = 10
    print(f"{N_INIT + N_BATCHES * BATCH_SIZE} evaluations")

    experiment = Experiment(
        name = "saasbo_experiment",
        search_space = search_space,
        optimization_config = optimization_config,
        runner = SyntheticRunner(),
    )

    sobol = Models.SOBOL(search_space = experiment.search_space)
    for _ in range(N_INIT):
        experiment.new_trial(sobol.gen(1)).run()

    data = experiment.fetch_data()

    num_samples = 16
    warmup_steps = 16
    gp_kernel = "matern"

    trials = []
    for i in range(N_BATCHES):
        model = Models.FULLYBAYESIAN(
            experiment = experiment, 
            data = data,
            num_samples = num_samples,   
            warmup_steps = warmup_steps,  
            gp_kernel = gp_kernel,  
            torch_device = tkwargs["device"],
            torch_dtype = tkwargs["dtype"],
            verbose = False,   
            disable_progbar = False,  
        )
        generator_run = model.gen(BATCH_SIZE)
        trial = experiment.new_batch_trial(generator_run=generator_run)
        trial.run()
     
        data = Data.from_multiple_data([data, trial.fetch_data()])
        new_value = trial.fetch_data().df["mean"].min()

        for arm in trial.arms:
            parameters = arm.parameters
            parameters["trial_number"] = i + 1
            parameters["batch_id"] = arm.name
            parameters["score"] = new_value
            trials.append(parameters)

        print(f"Iteration: {i + 1}, Best in iteration {new_value:.3f}, Best so far: {data.df['mean'].min():.3f}")
    
    output_dir = path_join(DATA_DIR, "saasbo")
    make_dir(output_dir)

    outfile = path_join(output_dir, "trials.csv")
    (
        pd.DataFrame(trials)
        .sort_values("score", ascending = True)
        .to_csv(outfile, index = False)
    )
    
    outfile = path_join(output_dir, "results.csv")
    (
        experiment
        .fetch_data()
        .df
        .sort_values("mean", ascending = True)
        .to_csv(outfile, index = False)
    )

    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(8, 6))
    res_saasbo = data.df['mean']
    ax.plot(np.minimum.accumulate(res_saasbo), color="b", label="SAASBO")
    ax.grid(True)
    ax.set_xlabel("Number of evaluations", fontsize=20)
    ax.set_xlim([0, len(res_saasbo)])
    ax.set_ylabel("Best value found", fontsize=20)
    ax.set_ylim([0, 8])
    ax.legend(fontsize=18)
    outfile = path_join(output_dir, "optimisation_results.png")
    fig.savefig(outfile)
       
    model = Models.FULLYBAYESIAN(
        experiment = experiment, 
        data = data,
        use_saas = True,
        num_samples = num_samples,
        warmup_steps = warmup_steps,
        gp_kernel = gp_kernel,
        torch_dtype = tkwargs["dtype"],
        torch_device = tkwargs["device"],
        disable_progbar = False,
        verbose = False
    )

    cv = cross_validate(model)
    y_true = np.stack([cv_.observed.data.means for cv_ in cv]).ravel()
    y_saas_mean = np.stack([cv_.predicted.means for cv_ in cv]).ravel()
    y_saas_std = np.stack([np.sqrt(np.diag(cv_.predicted.covariance)) for cv_ in cv]).ravel()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    min_val, max_val = -5, 120
    ax.plot([min_val, max_val], [min_val, max_val], "b--", lw=2)
    _, caps, bars = ax.errorbar(
        y_true,
        y_saas_mean,
        yerr=1.96 * y_saas_std,
        fmt=".",
        capsize=4,
        elinewidth=2.0,
        ms=14,
        c="k",
        ecolor="gray",
    )
    [bar.set_alpha(0.8) for bar in bars]
    [cap.set_alpha(0.8) for cap in caps]
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    ax.set_xlabel("True value", fontsize=20)
    ax.set_ylabel("Predicted value", fontsize=20)
    ax.grid(True)
    outfile = path_join(output_dir, "cv_results.png")
    fig.savefig(outfile)

    outfile = path_join(output_dir, CONFIG_FILE)
    write_sim_config(outfile, config)

if __name__ == "__main__":
    main()