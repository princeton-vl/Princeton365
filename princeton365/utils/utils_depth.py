import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture
from scipy.stats import gamma
from pomegranate.distributions import Normal, LogNormal, Gamma, Exponential
from pomegranate.gmm import GeneralMixtureModel
from scipy.optimize import minimize
from scipy.special import logsumexp
import os 
import sys
import warnings
from tqdm import tqdm
import torch
import random

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

def read_zed_depth(file_path):
    """
    Read depth data from a ZED camera depth file in H5 format.
    """
    with h5py.File(file_path, 'r') as f:
        print("Available keys in the file:", list(f.keys()))
        depth_data = f['data'][:]
        print(f"Data shape: {depth_data.shape}")
    return depth_data

def preprocess_depth_frame(frame):
    """
    Preprocess depth frame by removing invalid values and reshaping.
    """
    valid_depths = frame[frame > 0].flatten()
    return valid_depths


def calculate_bic(model, X):
    """Calculate BIC score for a mixture model"""
    n_samples = len(X)
    if hasattr(model, 'priors'): 
        n_params = 3 * len(model.priors) - 1
    else: 
        n_params = 2
    log_likelihood = torch.sum(torch.tensor(model.log_probability(X)).clone().detach())
    return -2 * log_likelihood.item() + n_params * np.log(n_samples)

def analyze_depth_distribution(depth_data, n_samples=10000, experiment=None):
    """Analyze depth distribution using GMMs and Gamma mixture models"""
    # np.random.seed(0)

    n_frames, height, width = depth_data.shape
    frame_indices = np.random.randint(0, n_frames, n_samples)
    row_indices = np.random.randint(0, height, n_samples)
    col_indices = np.random.randint(0, width, n_samples)
    
    # Extract sampled depths
    sampled_depths = depth_data[frame_indices, row_indices, col_indices]
    valid_depths = sampled_depths[sampled_depths > 0]
    
    # Convert to 2D array and to torch tensor
    X = torch.tensor(valid_depths.reshape(-1, 1), dtype=torch.float32)
    
    # Fit single component models first
    print("Fitting single component models...")
    single_gaussian = Normal()
    single_gaussian.fit(X)
    single_gaussian_bic = calculate_bic(single_gaussian, X)
    
    single_gamma = Gamma()
    single_gamma.fit(X)
    single_gamma_bic = calculate_bic(single_gamma, X)
    
    # Try different numbers of components
    gaussian_models = [single_gaussian]
    gamma_models = [single_gamma]
    gaussian_bics = [single_gaussian_bic]
    gamma_bics = [single_gamma_bic]
    n_components_range = list(range(2, 8))  # Test more components
    
    # Fit mixture models
    for n in n_components_range:
        print(f"Fitting models with {n} components...")
        try: 
            # Fit GMM
            gmm = GeneralMixtureModel([Normal() for _ in range(n)], verbose=False, random_state=0)
            gmm.fit(X)
            gaussian_models.append(gmm)
            gaussian_bics.append(calculate_bic(gmm, X))
            
            # Fit Gamma mixture
            gamma_mix = GeneralMixtureModel([Gamma() for _ in range(n)], verbose=False, random_state=0)
            gamma_mix.fit(X)
            gamma_models.append(gamma_mix)
            gamma_bics.append(calculate_bic(gamma_mix, X))
        except Exception as e:
            print(f"Error fitting models with {n} components: {e}")
            continue
                  
    
    # Find best models
    n_components_range_all = [1] + n_components_range
    opt_n_gaussian = n_components_range_all[np.argmin(gaussian_bics)]
    opt_n_gamma = n_components_range_all[np.argmin(gamma_bics)]
    
    best_gaussian = gaussian_models[np.argmin(gaussian_bics)]
    best_gamma = gamma_models[np.argmin(gamma_bics)]

    x = torch.linspace(float(valid_depths.min()), float(valid_depths.max()), 1000).reshape(-1, 1)
    x_np = x.numpy()
    
    # Plot best models
    if opt_n_gaussian == 1:
        gaussian_pdf = torch.exp(best_gaussian.log_probability(x).clone().detach())
    else:
        gaussian_pdf = torch.exp(best_gaussian.log_probability(x).clone().detach())
    
    if opt_n_gamma == 1:
        gamma_pdf = torch.exp(best_gamma.log_probability(x).clone().detach())
    else:
        gamma_pdf = torch.exp(best_gamma.log_probability(x).clone().detach())
        
    # Plot components of best model (if mixture)
    if min(gaussian_bics) < min(gamma_bics):
        best_model = best_gaussian
        n_best = opt_n_gaussian
        model_type = "Gaussian"
    else:
        best_model = best_gamma
        n_best = opt_n_gamma
        model_type = "Gamma"
    
    if n_best > 1:
        for i, dist in enumerate(best_model.distributions):
            log_probs = dist.log_probability(x).clone().detach()
            component_pdf = best_model.priors[i] * torch.exp(log_probs)
            if model_type == "Gaussian":
                mean = dist.means[0].item()
                std = np.sqrt(dist.covs[0].item())
                param_str = f'μ={mean:.2f}, σ={std:.2f}'
            else:
                shape = dist.shapes[0].item()
                rate = dist.rates[0].item()
                param_str = f'shape={shape:.2f}, rate={rate:.2f}'
                
    
    # Summary statistics
    summary_text = [
        f'Number of samples: {len(valid_depths)}',
        f'\nGaussian:',
        f'  Optimal n: {opt_n_gaussian}',
        f'  BIC Score: {min(gaussian_bics):.2f}',
        f'\nGamma:',
        f'  Optimal n: {opt_n_gamma}',
        f'  BIC Score: {min(gamma_bics):.2f}',
        f'\nBest Model: {model_type}{"Mix" if n_best > 1 else ""}',
        f'\nData Statistics:',
        f'  Mean: {torch.mean(X).item():.2f}',
        f'  Std Dev: {torch.std(X).item():.2f}',
        f'  Min: {torch.min(X).item():.2f}',
        f'  Max: {torch.max(X).item():.2f}'
    ]
    
    print("Analysis complete.")
    return {
        'gaussian_models': gaussian_models,
        'gamma_models': gamma_models,
        'gaussian_bics': gaussian_bics,
        'gamma_bics': gamma_bics,
        'opt_n_gaussian': opt_n_gaussian,
        'opt_n_gamma': opt_n_gamma,
        'best_type': "Gaussian" if min(gaussian_bics) < min(gamma_bics) else "Gamma",
        'n_best': n_best,
        'best_model': best_model
    }

def visualize_depth_frame(depth_array, frame_idx, min_depth=None, max_depth=None, colormap='viridis', experiment = None):
    """
    Convert a depth array to a colored image visualization.
    
    Parameters:
    depth_array (numpy.ndarray): 2D array of depth values in mm
    min_depth (float): Minimum depth value for normalization (optional)
    max_depth (float): Maximum depth value for normalization (optional)
    colormap (str): Matplotlib colormap name (default: 'viridis')
    
    Returns:
    numpy.ndarray: RGB image array
    """

    frame = depth_array[frame_idx]
    # Auto-compute min and max if not provided
    if min_depth is None:
        min_depth = np.min(frame[frame > 0])
    if max_depth is None:
        max_depth = np.max(frame)
    
    # Normalize the depth values to [0, 1] range
    normalized_depth = np.clip((frame - min_depth) / (max_depth - min_depth), 0, 1)
    
    # Convert to colored image using matplotlib's colormap
    cmap = plt.get_cmap(colormap)
    colored_image = cmap(normalized_depth)
    
    # Convert to uint8 RGB image (0-255 range)
    rgb_image = (colored_image[:, :, :3] * 255).astype(np.uint8)

    # save image
    if experiment: 
        plt.imsave(f"depth/{experiment}/depth_image_{experiment}.png", rgb_image)
    else:
        plt.imsave("depth_image.png", rgb_image)

    # visualize histogram of depth values and save it
    plt.hist(frame[frame > 0].flatten(), bins=100)
    if experiment:
        plt.savefig(f"depth/{experiment}/depth_histogram_{experiment}.png")
    else:
        plt.savefig("depth_histogram.png")
    plt.close()

def sample(model, n):
    """ 
    Sample positive values from a distribution model.
    """
    samples = []
    while len(samples) < n:
        sample = model.sample(1)
        if sample > 0:
            samples.append(sample)
    return samples
