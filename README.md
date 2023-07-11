# model-calibration-evaluation

Evaluating model calibration methods for sensitivity analysis, uncertainty analysis, optimisation

See [config.yaml](config.yaml) for the ground-truth simulation parameters.

The following model calibration methods have been evaluated.

* [Approximate Bayesian Computation - Sequential Monte Carlo](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/abc_smc/run.py)
* [Bayesian Optimisation](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/bayes_opt/run.py)
* [Differential Evolution Adaptive Metropolis](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/dream/run.py)
* [Tree-structured Parzen Estimator](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/optimisation/run.py)
* [Polynomial Chaos Expansion](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/poly_chaos/run.py)
* [Polynomial Chaos Kriging](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/poly_chaos_kriging/run.py)
* [Shuffled Complex Evolution Algorithm Uncertainty Analysis](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/sceua/run.py)
* [Sobol Sensitivity Analysis](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/sobol_sa/run.py)
  
