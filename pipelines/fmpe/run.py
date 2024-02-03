from lampe.data import JointLoader, JointDataset
from lampe.diagnostics import expected_coverage_mc, expected_coverage_ni
from lampe.inference import FMPE, FMPELoss
from lampe.plots import nice_rc, corner, mark_point, coverage_plot
from lampe.utils import GDStep
import torch
import torch.nn as nn
import torch.optim as optim
import zuko
from itertools import islice
from matplotlib import pyplot as plt
import numpy as np
import torch
from os.path import join as path_join

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

    def simulator(theta: torch.Tensor) -> torch.Tensor:
        mu, sigma = theta.cpu().detach().numpy().T
        y = simulator_func(mu, sigma, n, seed)
        return torch.Tensor(y).float()

    lower = torch.Tensor([mu_lb, sigma_lb])
    upper = torch.Tensor([mu_ub, sigma_ub])
    prior = zuko.distributions.BoxUniform(lower, upper)

    loader = JointLoader(
        prior, simulator, batch_size=1, vectorized=True
    )

    estimator = FMPE(
        2, n, hidden_features=[64] * 5, activation=nn.ELU
    )

    loss = FMPELoss(estimator)
    optimizer = optim.AdamW(estimator.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 128)
    step = GDStep(optimizer, clip=1.0)  # gradient descent step with gradient clipping

    estimator.train()

    for epoch in range(120):
        losses = torch.stack([
            step(loss(theta, x))
            for theta, x in islice(loader, 128)   
        ])
        scheduler.step()
        print(f"Epoch {epoch}; Loss {losses.mean().item()}")

    theta_star = torch.Tensor(
        np.array([config["mu"], config["sigma"]])
    )
    x_star = torch.Tensor(obs_data).float()
    estimator.eval()
    with torch.no_grad():
        samples = estimator.flow(x_star).sample((2**14,))

    output_dir = path_join(DATA_DIR, "fmpe")
    make_dir(output_dir)
    labels = ["mu", "sigma"]

    fig = corner(
        samples,
        smooth=2,
        domain=(lower, upper),
        labels=labels,
        legend=r'$p_\phi(\theta | x^*)$',
        figsize=(4.8, 4.8),
    )

    mark_point(fig, theta_star)
    plt.savefig(path_join(output_dir, "corner.png"))

    joint_dataset = JointDataset(theta_star.reshape((1, 1, 2)), x_star.reshape(1, n))
    fmpe_levels, fmpe_coverages = expected_coverage_mc(estimator.flow, joint_dataset)
    fig = coverage_plot(fmpe_levels, fmpe_coverages, legend='FMPE')
    plt.savefig(path_join(output_dir, "coverage.png"))

    outfile = path_join(output_dir, CONFIG_FILE)
    write_sim_config(outfile, config)

if __name__ == "__main__":
    main()