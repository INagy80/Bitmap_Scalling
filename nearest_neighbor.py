import numpy as np
import numpy.typing as npt


#############################################
# implement the NEAREST NEIGHBOR scaling here
#############################################
def nearest_neighbor_scaling(scaling: float, img_array: npt.ArrayLike) -> npt.NDArray:
    """
    Resize an image with nearest-neighbor interpolation.

    Parameters
    ----------
    scaling : float
        Scaling factor (>1 upscales, <1 downscales).
    img_array : array-like
        Input image (H×W×C or H×W).

    Returns
    -------
    scaled : ndarray
        Resized image.
    """
    if scaling == 1.0:
        return np.asarray(img_array).copy()

    img = np.asarray(img_array)
    in_h, in_w = img.shape[:2]

    # neue Auflösung berechnen
    out_h = max(1, int(round(in_h * scaling)))
    out_w = max(1, int(round(in_w * scaling)))

    # Ursprungskoord. der Zielpixel (Vektorisierung)
    y_idx = (np.arange(out_h) / scaling).round().astype(int)
    x_idx = (np.arange(out_w) / scaling).round().astype(int)

    # Randkorrektur
    y_idx = np.clip(y_idx, 0, in_h - 1)
    x_idx = np.clip(x_idx, 0, in_w - 1)

    # Broadcasting: (out_h, out_w[, C])
    scaled = img[y_idx[:, None], x_idx[None, :]]

    return scaled