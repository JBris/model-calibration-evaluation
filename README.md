# model-calibration-evaluation

[![pages-build-deployment](https://github.com/JBris/model-calibration-evaluation/actions/workflows/pages/pages-build-deployment/badge.svg?branch=main)](https://github.com/JBris/model-calibration-evaluation/actions/workflows/pages/pages-build-deployment)
[![CodeQL](https://github.com/JBris/model-calibration-evaluation/actions/workflows/github-code-scanning/codeql/badge.svg?branch=main)](https://github.com/JBris/model-calibration-evaluation/actions/workflows/github-code-scanning/codeql)

Evaluating model calibration methods for sensitivity analysis, uncertainty analysis, optimisation, and Bayesian inference 

See [config.yaml](config.yaml) for the ground-truth simulation parameters.

The following model calibration methods have been evaluated.

* [Approximate Bayesian Computation - Sequential Monte Carlo](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/abc_smc/run.py)
* [Bayesian Optimisation](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/bayes_opt/run.py)
* [Bayesian Optimisation for Likelihood-Free Inference](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/bolfi/run.py)
* [Differential Evolution Adaptive Metropolis](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/dream/run.py)
* [Experimental Design via Gaussian Process Emulation](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/experimental_design/run.py)
* [Flow Matching Posterior Estimation](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/fmpe/run.py)
* [Tree-structured Parzen Estimator](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/optimisation/run.py)
* [Polynomial Chaos Expansion](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/poly_chaos/run.py)
* [Polynomial Chaos Kriging](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/poly_chaos_kriging/run.py)
* [Sparse Axis-Aligned Subspace Bayesian Optimization](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/saasbo/run.py)
* [Shuffled Complex Evolution Algorithm Uncertainty Analysis](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/sceua/run.py)
* [Sequential Neural Posterior Estimation](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/snpe/run.py)
* [Sobol Sensitivity Analysis](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/sobol_sa/run.py)
* [Truncated Marginal Neural Ratio Estimation](https://github.com/JBris/model-calibration-evaluation/tree/main/pipelines/tmnre/run.py)
