import numpy as np 
import math 

def generate_ap_locations(ap_nums, area_size):
    n_rows = max(1, round(math.sqrt(ap_nums))) #number of rows 
    base = ap_nums // n_rows 
    remainder = ap_nums % n_rows
    row_counts = np.full(n_rows, base, dtype = int)
    
    #Add the remainder AP to the row being closet to center 
    row_order = np.argsort(
        np.abs(np.arange(n_rows) - (n_rows - 1) / 2)
    )
    for i in range(remainder):
        row_counts[row_order[i]] += 1

    #y location 
    dy = area_size / n_rows 
    y_coords = (np.arange(n_rows) + 0.5) * dy
    positions = [] 
    for row_idx, num_aps in enumerate(row_counts):
        # AP in each rows in the Ox 
        dx = area_size / num_aps
        x_coords = (np.arange(num_aps) + 0.5) * dx

        for x in x_coords:
            positions.append([
                x,
                y_coords[row_idx]
            ])

    return np.array(positions, dtype=np.float32) #(AP, 2)

def generate_ue_locations(area_size, user_equipment_size, margin = 5.0):
    return np.random.uniform(
        low = margin, 
        high = area_size - margin, 
        size = (user_equipment_size, 2)
    ) #(UE, 2)

#for one AP
def generate_path_angles(num_tx_paths, num_rx_paths, seed=None, dtype=np.float32):
    rng = np.random.default_rng(seed)

    def sample_angles(num_paths):
        # Elevation:
        # f(theta) = 0.5 * sin(theta), theta in [0, pi]
        u = rng.uniform(
            low=0.0,
            high=1.0,
            size=num_paths
        )

        elevation = np.arccos(
            1.0 - 2.0 * u
        )

        # Azimuth:
        # Uniform over [0, pi]
        azimuth = rng.uniform(
            low=0.0,
            high=np.pi,
            size=num_paths
        )

        return (
            elevation.astype(dtype),
            azimuth.astype(dtype)
        )

    # Transmit paths
    tx_elevation, tx_azimuth = sample_angles(
        num_tx_paths
    )

    # Receive paths
    rx_elevation, rx_azimuth = sample_angles(
        num_rx_paths
    )

    return (
        tx_elevation,
        tx_azimuth,
        rx_elevation,
        rx_azimuth
    )

def generate_path_angles_for_all_AP(num_aps, num_tx_paths, num_rx_paths, seed=None, dtype=np.float32):
    """
    elevation theta in [0, pi]
    azimuth   phi   in [0, pi]
    """
    rng = np.random.default_rng(seed)

    def sample_angles(shape):
        # Elevation:
        # f_theta(theta) = 0.5 * sin(theta), theta in [0, pi]
        u = rng.uniform(
            low=0.0,
            high=1.0,
            size=shape,
        )

        elevation = np.arccos(
            1.0 - 2.0 * u
        )

        # Azimuth:
        # Uniform over [0, pi]
        azimuth = rng.uniform(
            low=0.0,
            high=np.pi,
            size=shape,
        )

        return (
            elevation.astype(dtype),
            azimuth.astype(dtype),
        )

    # Transmit paths
    tx_elevation, tx_azimuth = sample_angles(
        (num_aps, num_tx_paths)
    )

    # Receive paths
    rx_elevation, rx_azimuth = sample_angles(
        (num_aps, num_rx_paths)
    )

    return (
        tx_elevation,
        tx_azimuth,
        rx_elevation,
        rx_azimuth,
    )
    
def generate_path_response_matrix(num_paths, 
                                  distance, 
                                  pathloss_ref_db, #C_0 
                                  reference_distance, 
                                  pathloss_exponent,
                                  rician_factor,
                                  rng = None):
    if rng is None:
        rng = np.random.default_rng()

    C_0 = 10.0 ** (pathloss_ref_db / 10.0)
    path_gain = C_0 * (distance / reference_distance) ** (-pathloss_exponent)
    los_variance = (path_gain * rician_factor / (rician_factor + 1)) #LoS path
    nlos_variance = (path_gain / ((rician_factor + 1.0) * (num_paths - 1))) #NLoS path 
    
    sigma = np.zeros(
        num_paths,
        dtype = np.complex128
    )
    
    sigma[0] = rng.normal(0.0, np.sqrt(los_variance / 2.0)) + 1j * rng.normal(0.0, np.sqrt(los_variance / 2.0))
    
    for p in range(1, num_paths):
        sigma[p] = rng.normal(0.0, np.sqrt(nlos_variance / 2.0)) + 1j * rng.normal(0.0, np.sqrt(nlos_variance / 2.0))
    
    Sigma = np.diag(sigma)
    return Sigma