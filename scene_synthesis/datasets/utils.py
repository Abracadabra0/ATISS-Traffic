import numpy as np


def crop_image(image: np.array,
               x_px: int,
               y_px: int,
               axes_limit_px: int) -> np.array:
    x_min = int(x_px - axes_limit_px)
    x_max = int(x_px + axes_limit_px)
    y_min = int(y_px - axes_limit_px)
    y_max = int(y_px + axes_limit_px)
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image


def get_homogeneous_matrix(translation: np.array,
                           rotation: np.array) -> np.array:
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation
    homogeneous_matrix[:3, 3] = translation
    return homogeneous_matrix


def cartesian_to_polar(vector: np.array) -> np.array:
    rho = np.linalg.norm(vector)
    if rho == 0:
        return np.array([0., 0.])
    theta = np.arctan(vector[1] / vector[0]) + (vector[0] < 0) * np.pi
    theta = theta + (theta < 0) * np.pi * 2
    return np.array([rho, theta])
