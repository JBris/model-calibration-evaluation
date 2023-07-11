import numpy as np
from pathlib import Path
import yaml

CONFIG_FILE = "config.yaml"
DATA_DIR = "data"
INPUT_FILE = "input.csv"

def read_sim_config() -> dict:
    with open(CONFIG_FILE) as f:
        options: dict = yaml.safe_load(f)

    return options

def write_sim_config(outfile, config) -> dict:
    with open(outfile, 'w') as f:
        yaml.dump(config, f, default_flow_style = False)
    return outfile

def simulator_func(mu: float, sigma: float, n: int, seed: int = None) -> np.array: 
    rng = np.random.default_rng(seed)
    draws = rng.normal(loc = mu, scale = sigma, size = n)
    return draws

def load_data(infile: str, delimiter: str = ',') -> np.array:
    arr = np.loadtxt(infile, delimiter = delimiter)
    return arr

def save_data(outfile: str, arr: np.array, delimiter: str = ',') -> str:
    np.savetxt(outfile, arr, delimiter = delimiter, fmt='%f')
    return outfile

def make_dir(outdir: str) -> str:
    Path(outdir).mkdir(parents = True, exist_ok = True)
    return outdir