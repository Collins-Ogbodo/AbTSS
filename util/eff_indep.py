import numpy as np

def effective_independence(mode_shape, num_sensors, spatial_cov):
    """
    Perform Effective Independence (EfI) sensor placement.
    
    Parameters:
    mode shape matric (numpy.ndarray).
    num_sensors (int): Number of sensors to place.
    spatial_cov(numpy.ndarray): spatial correclation matrix
    
    Returns:
    numpy.ndarray: Indices of the selected sensor locations.
    """
    #Compute FIM
    FIM = mode_shape.T @ mode_shape
    # Initialize sensor indices
    sensor_indices = np.arange(mode_shape.shape[0])
    spatial_cov_inv = np.linalg.inv(spatial_cov)
    # Calculate the initial EfI values
    efi_vector = np.diag(spatial_cov_inv @ mode_shape @ np.linalg.inv(FIM) @ mode_shape.T @ spatial_cov_inv)
    efi_vector = np.multiply(efi_vector, np.diag(spatial_cov_inv))
    while len(sensor_indices) != num_sensors:
        # Select the sensor with the lowest EfI value
        min_index = np.argmin(efi_vector)

        # Remove the selected sensor from the list
        sensor_indices = np.delete(sensor_indices, (min_index))

        # Update the Mode shape matrix by removing the selected sensor
        mode_shape = np.delete(mode_shape, min_index, axis=0)
        spatial_cov_inv = np.delete(spatial_cov_inv, min_index, axis=0)
        spatial_cov_inv = np.delete(spatial_cov_inv, min_index, axis=1)
        # Recalculate the EfI values
        FIM = mode_shape.T @ mode_shape
        efi_vector = np.diag(spatial_cov_inv @ mode_shape @ np.linalg.inv(FIM) @ mode_shape.T @ spatial_cov_inv)
        efi_vector = np.multiply(efi_vector, np.diag(spatial_cov_inv))
    return sensor_indices, efi_vector