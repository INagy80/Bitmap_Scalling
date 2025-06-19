import numpy as np
import numpy.typing as npt


def bilinear_interpolation_scaling(scaling: float,
                                   img_array: npt.ArrayLike) -> npt.NDArray:
    if scaling == 1.0:
        return np.asarray(img_array).copy()

    img = np.asarray(img_array)
    in_h, in_w = img.shape[:2]

    # Bild ggf. auf (H, W, 1) bringen, damit wir einheitlich 3-D rechnen
    squeezed = False
    if img.ndim == 2:
        img = img[..., np.newaxis]
        squeezed = True

    channels = img.shape[2]

    # Zielgröße
    out_h = max(1, int(round(in_h * scaling)))
    out_w = max(1, int(round(in_w * scaling)))

    # Gitter der Zielkoordinaten
    y = np.linspace(0, in_h - 1, out_h)
    x = np.linspace(0, in_w - 1, out_w)

    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    x1 = np.clip(x0 + 1, 0, in_w - 1)

    # Gewichte (jetzt mit Kanal-Achse)
    wy = (y - y0).reshape(out_h, 1, 1)      # (out_h, 1, 1)
    wx = (x - x0).reshape(1, out_w, 1)      # (1, out_w, 1)

    # Als 2-D Matrizen zum Indexieren
    y0 = y0.reshape(out_h, 1)
    y1 = y1.reshape(out_h, 1)
    x0 = x0.reshape(1, out_w)
    x1 = x1.reshape(1, out_w)

    # Vier Nachbarpixel
    Ia = img[y0, x0]        # links-oben
    Ib = img[y0, x1]        # rechts-oben
    Ic = img[y1, x0]        # links-unten
    Id = img[y1, x1]        # rechts-unten

    # Bilineare Interpolation
    top = Ia * (1 - wx) + Ib * wx
    bottom = Ic * (1 - wx) + Id * wx
    out = top * (1 - wy) + bottom * wy

    # Datentyp wiederherstellen
    if np.issubdtype(img_array.dtype, np.integer):
        out = np.rint(out).astype(img_array.dtype)
    else:
        out = out.astype(img_array.dtype)

    # Bei Graustufen die Kanal-Achse entfernen
    if squeezed:
        out = out.squeeze(-1)

    return out