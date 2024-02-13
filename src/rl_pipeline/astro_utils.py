#resolution
#sensitivity
#Briggs weigthing (+Uniform, +Natural)
#Tapered
import numpy as np

# SKIPPING CONSTANTS
def resolution(wavelength, baselines):
    # Baselines is a list of baselines. Resolution depends on the longest baseline
    #return (wavelength / max(baselines))
    return 1 / max(baselines)

# SKIPPING CONSTANTS
def sensitivity(num_antennas, system_temp=1, integration_time=3600, bandwidth=200e6, individual_radius=7.5):
    # The SKA has 197 antennas, the new ones have radius 7.5m, the MeerKat ones 6.75m
    # Bandwidth 50-350MHz
    #numerator = 2*1.380649e-23 * system_temp
    #denominator = num_antennas * np.pi * individual_radius**2 * \
     #               np.sqrt(integration_time * bandwidth)
    
    #return (numerator / denominator)
    return 1/ num_antennas


def briggs_weighting(baselines, source_vis=1):
    # Recreating eqn 3.13 from Dan Briggs thesis. Set delta S_uw to 1
    return [1/((source_vis**2) * sum(baselines) + 2) for _ in baselines] # 8.2 e-06

def tapered_weighting(baselines, taper_threshold=0.8):
    # Calculate the weights based on the tapered regime (long baselines weighted less)
    weights = np.ones_like(baselines)
    taper_indices = np.where(baselines > taper_threshold * max(baselines))

    for i in taper_indices[0]:
        weights[i] = taper_threshold * max(baselines) / baselines[i]
    
    return weights # Works
