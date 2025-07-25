import numpy as np
from skimage.metrics import hausdorff_distance
from scipy.spatial.distance import cdist

def mask_iou(mask1, mask2):
    """ Input 
    mask1: numpy array of shape (H, W) with boolean values
    mask2: numpy array of shape (H, W) with boolean values
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou_score = intersection / union if union != 0 else 0.0
    return iou_score

def dice_coefficient(mask1, mask2):
    #mask1 = mask1.astype(bool)
    #mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    size1 = mask1.sum()
    size2 = mask2.sum()

    if size1 + size2 == 0:
        return 1.0  # Both masks are empty
    dice = 2. * intersection / (size1 + size2)
    return dice


def average_surface_distance(contour1, contour2):
    """
    contour1: numpy array of shape (N, 2) representing points in contour1
    contour2: numpy array of shape (M, 2) representing points in contour2
    """
    # Compute all pairwise distances between contour points
    dists_1_to_2 = cdist(contour1, contour2)
    dists_2_to_1 = cdist(contour2, contour1)

    # For each point in contour1, find distance to closest point in contour2
    min_dists_1_to_2 = dists_1_to_2.min(axis=1)
    # For each point in contour2, find distance to closest point in contour1
    min_dists_2_to_1 = dists_2_to_1.min(axis=1)

    # Average the minimal distances for both directions
    asd = (min_dists_1_to_2.mean() + min_dists_2_to_1.mean()) / 2.0
    return asd



# Usage example
"""
mask1 = None #numpy array of shape (H, W) with boolean (0,1) values
mask2 = None #numpy array of shape (H, W) with boolean (0,1) values

iou = mask_iou(mask1, mask2)
print("IOU:", iou)

hd = hausdorff_distance(mask1, mask2)
print("Hausdorff Distance (HD):", hd, "pixels")

contour1 = None  # numpy array of shape (N, 2) representing points in contour1
contour2 = None  # numpy array of shape (M, 2) representing points in contour2

asd_value = average_surface_distance(contour1, contour2)
print(f"Average Surface Distance (ASD): {asd_value:.2f} pixels")
"""

""" Find the two points causing the HD and the distance and plot them """
"""
def find_hd_points(u, v):
    # Distances from each u to closest v
    d_uv = cdist(u, v)
    u_to_v_min = d_uv.min(axis=1)
    max_u_idx = np.argmax(u_to_v_min)
    max_u_dist = u_to_v_min[max_u_idx]
    closest_v_idx_for_u = d_uv[max_u_idx].argmin()

    # Distances from each v to closest u
    d_vu = cdist(v, u)
    v_to_u_min = d_vu.min(axis=1)
    max_v_idx = np.argmax(v_to_u_min)
    max_v_dist = v_to_u_min[max_v_idx]
    closest_u_idx_for_v = d_vu[max_v_idx].argmin()

    # Decide which direction is the Hausdorff distance
    if max_u_dist > max_v_dist:
        return (u[max_u_idx], v[closest_v_idx_for_u]), max_u_dist
    else:
        return (u[closest_u_idx_for_v], v[max_v_idx]), max_v_dist

# Find the two points causing the HD and the distance
(pointA, pointB), hd_value = find_hd_points(coords1, coords2)
print(f"Hausdorff Distance: {hd_value}")

# Plot contours and highlight these points
plt.figure(figsize=(6, 6))
plt.plot(coords1[:, 0], coords1[:, 1], label="Mask 1 contour", color='blue')
plt.plot(coords2[:, 0], coords2[:, 1], label="Mask 2 contour", color='green')

plt.scatter(*pointA, color='red', s=50, label="Max dist point on Mask 1")
plt.scatter(*pointB, color='orange', s=50, label="Closest point on Mask 2")

plt.legend()
plt.gca().invert_yaxis()  # Invert y if image coordinates
plt.axis("off")
plt.show()
"""