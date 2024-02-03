from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import swyft
import torch
from os.path import join as path_join

from model_calibration_lib.gaussian_sim import (
    read_sim_config, simulator_func, load_data, DATA_DIR, INPUT_FILE, 
    CONFIG_FILE, make_dir, write_sim_config
)

class Simulator(swyft.Simulator):
    def __init__(
            self, mu_lb: float, mu_ub: float, sigma_lb: float, 
            sigma_ub: float, n: int, seed: int
        ):
        super().__init__()
        bounds = np.array([[mu_lb, mu_ub], [sigma_lb, sigma_ub]])
        self.transform_samples = swyft.to_numpy32
        self.n = n
        self.seed = seed

        self.sample_z = swyft.RectBoundSampler(
            [ 
                stats.uniform(mu_lb, mu_ub - mu_lb), 
                stats.uniform(sigma_lb, sigma_ub - sigma_lb) 
            ],
            bounds = bounds
        )

    def simulator_func(self, z):
        mu, sigma = z
        y = simulator_func(mu, sigma, self.n, self.seed)
        return y

    def build(self, graph):
        z = graph.node('z', self.sample_z)
        y = graph.node('y', self.simulator_func, z)

class SimulatorObs(swyft.Simulator):
    def __init__(self, obs):
        super().__init__()
        self.obs = obs
        self.transform_samples = swyft.to_numpy32

    def build(self, graph):
        y = graph.node('y', lambda: self.obs)

class Network(swyft.SwyftModule):
    def __init__(self, n, num_features = 10, num_params = 2, lr = 1e-3):
        super().__init__()
        self.learning_rate = lr
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n, 32),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(32, num_features),
            torch.nn.ReLU()
        )

        self.logratios = swyft.LogRatioEstimator_1dim(
            num_features = num_features, num_params = num_params, varnames = 'z'
        )

    def forward(self, A, B):
        f = self.net(A['y'])
        logratios = self.logratios(f, B['z'])
        return logratios

def main():
    infile = path_join(DATA_DIR, INPUT_FILE)
    obs_data = load_data(infile)
    config = read_sim_config()
    
    n_samples = 10000
    batch_size = 64

    n, seed = config["n"], config["seed"]
    mu_lb, mu_ub, sigma_lb, sigma_ub = config["mu_lb"], config["mu_ub"], config["sigma_lb"], config["sigma_ub"]
    
    sim = Simulator(mu_lb, mu_ub, sigma_lb, sigma_ub, n, seed)
    samples = sim.sample(n_samples)
    dm = swyft.SwyftDataModule(samples, batch_size = batch_size)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta = 0., patience=10, verbose=False, mode='min')
            
    trainer = swyft.SwyftTrainer(accelerator = 'gpu', max_epochs = 30, callbacks=[lr_monitor, early_stopping_callback])
    network = Network(n)
    trainer.fit(network, dm)

    prior_samples = sim.sample(n_samples, targets = ['z'])
    obs = SimulatorObs(obs_data).sample()
    predictions = trainer.infer(network, obs, prior_samples)

    output_dir = path_join(DATA_DIR, "tmnre")
    make_dir(output_dir)

    truth = { "z[0]": config["mu"], "z[1]": config["sigma"]}

    for func in [ 
        swyft.plot_posterior, swyft.plot_corner, 
    ]:
        func(predictions, ('z[0]', 'z[1]'), bins = n, smooth = 3, truth = truth) 
        plt.savefig(path_join(output_dir, f"{func.__name__}.png"))

    coverage_samples = trainer.test_coverage(network, samples[-500:], prior_samples)
    for func in [ swyft.plot_zz, swyft.plot_pp ]:
        _, axes = plt.subplots(1, 2, figsize = (12, 4))
        for i in range(2):
            func(coverage_samples, f"z[{i}]", ax = axes[i]) 
        plt.tight_layout()
        plt.savefig(path_join(output_dir, f"{func.__name__}.png"))

    outfile = path_join(output_dir, CONFIG_FILE)
    write_sim_config(outfile, config)

if __name__ == "__main__":
    main()