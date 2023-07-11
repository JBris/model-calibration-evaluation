from os.path import join as path_join
from model_calibration_lib.gaussian_sim import (read_sim_config, simulator_func, save_data, DATA_DIR, INPUT_FILE)

def main():
    config = read_sim_config()
    mu, sigma, n, seed = config["mu"], config["sigma"], config["n"], config["seed"]
    draws = simulator_func(mu, sigma, n, seed)
    outfile = path_join(DATA_DIR, INPUT_FILE)
    save_data(outfile, draws)

if __name__ == "__main__":
    main()