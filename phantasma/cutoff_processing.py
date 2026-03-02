"""
cutoff_processing.py
Utilities to preprocess astronomical maps: smooth to a target resolution
accounting for pixel window functions, reproject to a target WCS, and
extract a cutout.

Supports both FITS (flat-sky, provided as data+WCS) and HEALPix input maps.
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.nddata import Cutout2D
from astropy.convolution import Gaussian2DKernel, convolve
from reproject import reproject_exact, reproject_from_healpix
import healpy as hp
import warnings


# Conversion factor: FWHM = 2 * sqrt(2 * ln2) * sigma
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ≈ 0.4247


def _pixel_window_fwhm(pixel_size_arcmin):
    """
    Approximate FWHM of the pixel window function for a square pixel.

    A square pixel of angular size delta acts as a 2D top-hat convolution.
    Its Gaussian approximation has sigma = delta / sqrt(12) per axis,
    giving FWHM = 2*sqrt(2*ln2) * delta / sqrt(12) ≈ 0.6798 * delta.

    Parameters
    ----------
    pixel_size_arcmin : float
        Pixel angular size in arcmin.

    Returns
    -------
    float
        Approximate pixel window FWHM in arcmin.
    """
    return (1.0 / _FWHM_TO_SIGMA) * pixel_size_arcmin / np.sqrt(12.0)


def _compute_smoothing_kernel_sigma_pix(
    original_res_arcmin,
    target_res_arcmin,
    orig_pixel_size_arcmin,
    new_pixel_size_arcmin,
    current_pixel_scale_arcmin,
):
    """
    Compute the Gaussian smoothing kernel sigma in pixels.

    The kernel accounts for:
    - The original Gaussian beam (already applied to the data)
    - The original pixel window (already baked into the data)
    - The new pixel window (which ``reproject_exact`` will apply)

    .. math::

        \\mathrm{FWHM_{kernel}}^2 =
            \\mathrm{FWHM_{target}}^2
          - \\mathrm{FWHM_{beam}}^2
          - \\mathrm{FWHM_{pix,orig}}^2
          - \\mathrm{FWHM_{pix,new}}^2

    Parameters
    ----------
    original_res_arcmin : float
        Original *beam* FWHM in arcmin (Gaussian; does **not** include the
        pixel window — that is accounted for separately).
    target_res_arcmin : float
        Desired total effective FWHM after smoothing + reprojection [arcmin].
    orig_pixel_size_arcmin : float
        Pixel size of the *original* map [arcmin].  Its pixel window is
        already present in the data.
    new_pixel_size_arcmin : float
        Pixel size of the *output* grid [arcmin].  ``reproject_exact`` will
        effectively convolve with this pixel window.
    current_pixel_scale_arcmin : float
        Pixel scale of the intermediate grid on which the smoothing kernel
        is applied [arcmin].  Used to convert the kernel from arcmin to
        pixels.

    Returns
    -------
    sigma_pix : float or None
        Gaussian kernel sigma in pixels.  ``None`` if no smoothing is needed.
    fwhm_kernel_arcmin : float
        FWHM of the smoothing kernel in arcmin (0 if no smoothing needed).
    """
    pw_orig = _pixel_window_fwhm(orig_pixel_size_arcmin)
    pw_new  = _pixel_window_fwhm(new_pixel_size_arcmin)

    fwhm_sq = (
        target_res_arcmin**2
        - original_res_arcmin**2
        - pw_orig**2
        - pw_new**2
    )

    if fwhm_sq <= 0:
        eff = np.sqrt(original_res_arcmin**2 + pw_orig**2 + pw_new**2)
        warnings.warn(
            f"Target resolution ({target_res_arcmin:.2f}') is not larger than "
            f"sqrt(beam^2 + pix_window_orig^2 + pix_window_new^2) = "
            f"{eff:.2f}'.  No additional smoothing will be applied.",
            stacklevel=3,
        )
        return None, 0.0

    fwhm_kernel = np.sqrt(fwhm_sq)
    sigma_kernel_arcmin = fwhm_kernel * _FWHM_TO_SIGMA
    sigma_pix = sigma_kernel_arcmin / current_pixel_scale_arcmin

    return sigma_pix, fwhm_kernel


def _create_intermediate_wcs(center_l, center_b, size_deg, pixel_size_arcmin):
    """
    Create a gnomonic (TAN) WCS in Galactic coordinates centred on (l, b).

    Parameters
    ----------
    center_l, center_b : float
        Centre of the projection in Galactic degrees.
    size_deg : float
        Angular size of the field in degrees.
    pixel_size_arcmin : float
        Pixel size in arcmin.

    Returns
    -------
    wcs : astropy.wcs.WCS
    shape : tuple of int
        (ny, nx)
    """
    pixel_size_deg = pixel_size_arcmin / 60.0
    n_pix = int(np.ceil(size_deg / pixel_size_deg))
    if n_pix % 2 == 1:
        n_pix += 1  # keep even for symmetry

    w = WCS(naxis=2)
    w.wcs.crpix = [n_pix / 2.0 + 0.5, n_pix / 2.0 + 0.5]
    w.wcs.cdelt = [-pixel_size_deg, pixel_size_deg]
    w.wcs.crval = [center_l, center_b]
    w.wcs.ctype = ["GLON-TAN", "GLAT-TAN"]
    return w, (n_pix, n_pix)


def make_target_wcs(center_l, center_b, pixel_size_arcmin, cutout_size_deg):
    """
    Create a gnomonic (TAN) WCS in Galactic coordinates to use as the
    output grid for ``preprocess_and_cutout``.

    Parameters
    ----------
    center_l : float
        Galactic longitude of the cutout centre [deg].
    center_b : float
        Galactic latitude of the cutout centre [deg].
    pixel_size_arcmin : float
        Desired output pixel size [arcmin].
    cutout_size_deg : float
        Angular size of the cutout [deg].  The grid will be square with
        ``ceil(cutout_size_deg / pixel_size_deg)`` pixels per side
        (rounded up to the nearest even number).

    Returns
    -------
    wcs : astropy.wcs.WCS
        2-D WCS centred on (center_l, center_b) with TAN projection.
    shape : tuple of int
        ``(ny, nx)`` shape of the corresponding output grid.

    Examples
    --------
    >>> target_wcs, shape = make_target_wcs(17.0, 0.8, pixel_size_arcmin=2.0, cutout_size_deg=2.0)
    >>> data_out, wcs_out = ph.smooth_cutout(..., target_wcs=target_wcs, target_shape=shape)
    """
    return _create_intermediate_wcs(center_l, center_b, cutout_size_deg, pixel_size_arcmin)


def _sanitise(data, remove_healpix_unseen=False):
    """Replace non-finite values (inf, etc.) with NaN.

    Parameters
    ----------
    data : ndarray
    remove_healpix_unseen : bool
        If True, also replace the HEALPix UNSEEN sentinel (~-1.6e30) with NaN.
        Only needed for HEALPix input.
    """
    out = np.array(data, dtype=np.float64)
    out[~np.isfinite(out)] = np.nan
    if remove_healpix_unseen:
        out[np.isclose(out, hp.UNSEEN, atol=1e-6)] = np.nan
    return out


def smooth_cutout(
    data,
    map_format="fits",
    original_wcs=None,
    center_l=0.0,
    center_b=0.0,
    cutout_size_deg=1.0,
    original_res_arcmin=5.0,
    target_res_arcmin=10.0,
    pixel_size_arcmin=1.0,
    target_wcs=None,
    target_shape=None,
    healpix_coord="G",
):
    """
    Smooth an astronomical map to a target resolution and reproject it onto
    a target WCS grid.

    The function handles both flat-sky FITS images (provided as a numpy array
    + WCS) and HEALPix maps (provided as a 1-D pixel array).  Reprojection is
    done with ``reproject_exact`` (flux-conserving).

    The original pixel scale is read automatically from ``original_wcs`` (for
    FITS) or computed from the HEALPix Nside (for HEALPix).  The output grid
    shape is computed from ``pixel_size_arcmin`` and ``cutout_size_deg``.

    Parameters
    ----------
    data : numpy.ndarray
        - For ``map_format='fits'``: 2-D (or higher, extra axes are dropped)
          pixel array of the original map.
        - For ``map_format='healpix'``: 1-D HEALPix pixel array.
    map_format : {'fits', 'healpix'}
        Format of the input map.
    original_wcs : astropy.wcs.WCS, optional
        WCS of the original FITS map.  Required when ``map_format='fits'``;
        ignored for ``'healpix'`` (Nside is inferred from the array length).
    center_l : float
        Galactic longitude of the cutout centre [deg].
    center_b : float
        Galactic latitude of the cutout centre [deg].
    cutout_size_deg : float
        Angular size of the cutout [deg].
    original_res_arcmin : float
        Original *beam* FWHM of the input map [arcmin].  This is the
        Gaussian beam only — the pixel window is accounted for separately
        using the pixel scales.
    target_res_arcmin : float
        Desired *total effective* FWHM after smoothing and reprojection
        [arcmin].
    pixel_size_arcmin : float
        Pixel size of the *output* grid [arcmin].  Used both to compute the
        pixel window correction and to determine the output grid shape.
    target_wcs : astropy.wcs.WCS
        WCS of the output grid.
    healpix_coord : str, optional
        Coordinate system of the HEALPix map: ``'G'`` (Galactic), ``'C'``
        (Celestial/Equatorial), ``'E'`` (Ecliptic).  Default ``'G'``.

    Returns
    -------
    cutout_data : numpy.ndarray
        2-D map reprojected onto ``target_wcs`` with NaN for invalid pixels.
    target_wcs : astropy.wcs.WCS
        The WCS of the returned map (same object as the input ``target_wcs``).

    Notes
    -----
    **Pixel window correction**

    The input data has already been smoothed by the original beam *and* the
    original pixel window.  ``reproject_exact`` will add the output pixel
    window.  The smoothing kernel subtracts all three contributions so that
    the final effective resolution equals ``target_res_arcmin``::

        FWHM_kernel^2 = FWHM_target^2 - FWHM_beam^2
                      - FWHM_pix_orig^2 - FWHM_pix_new^2

    where  FWHM_pix ≈ 0.68 × pixel_size  (Gaussian approximation of a
    square top-hat pixel window).
    """
    if target_wcs is None:
        raise ValueError("target_wcs must be provided.")
    if original_res_arcmin <= 0:
        raise ValueError("original_res_arcmin must be positive.")
    if target_res_arcmin <= 0:
        raise ValueError("target_res_arcmin must be positive.")
    if pixel_size_arcmin <= 0:
        raise ValueError("pixel_size_arcmin must be positive.")
    if cutout_size_deg <= 0:
        raise ValueError("cutout_size_deg must be positive.")

    # Output grid shape derived from output pixel scale and cutout size.
    # Can be overridden by passing target_shape explicitly (e.g. from make_target_wcs).
    if target_shape is None:
        pixel_size_deg = pixel_size_arcmin / 60.0
        n_pix = int(np.ceil(cutout_size_deg / pixel_size_deg))
        if n_pix % 2 == 1:
            n_pix += 1
        target_shape = (n_pix, n_pix)

    # ------------------------------------------------------------------
    # 1.  Read / prepare the input map on a flat-sky intermediate grid
    # ------------------------------------------------------------------
    # Padding around the cutout to avoid edge effects from convolution.
    # 5× the target FWHM is generous (kernel is ~4σ ≈ 1.7 FWHM).
    padding_deg = 5.0 * target_res_arcmin / 60.0
    padded_size_deg = cutout_size_deg + 2.0 * padding_deg

    if map_format == "healpix":
        data_proc, current_wcs, current_pixel_arcmin = _read_healpix(
            data, healpix_coord,
            center_l, center_b, padded_size_deg, pixel_size_arcmin,
        )

    elif map_format == "fits":
        if original_wcs is None:
            raise ValueError(
                "original_wcs must be provided when map_format='fits'."
            )
        data_proc, current_wcs, current_pixel_arcmin = _read_fits(
            data, original_wcs,
            center_l, center_b, padded_size_deg,
        )

    else:
        raise ValueError(f"Unknown map_format: '{map_format}'")

    # ------------------------------------------------------------------
    # 2.  Sanitise – replace sentinels / inf with NaN
    # ------------------------------------------------------------------
    data_proc = _sanitise(data_proc, remove_healpix_unseen=(map_format == "healpix"))

    # ------------------------------------------------------------------
    # 3.  Smooth to the target resolution
    # ------------------------------------------------------------------
    sigma_pix, fwhm_kernel = _compute_smoothing_kernel_sigma_pix(
        original_res_arcmin,
        target_res_arcmin,
        orig_pixel_size_arcmin=current_pixel_arcmin,
        new_pixel_size_arcmin=pixel_size_arcmin,
        current_pixel_scale_arcmin=current_pixel_arcmin,
    )

    if sigma_pix is not None and sigma_pix > 0:
        # Use astropy convolve to properly handle NaN pixels.
        kernel = Gaussian2DKernel(x_stddev=sigma_pix)
        data_proc = convolve(
            data_proc, kernel,
            boundary="fill",
            fill_value=np.nan,
            nan_treatment="interpolate",
            preserve_nan=True,
        )

    # ------------------------------------------------------------------
    # 4.  Reproject to the target WCS  (exact / flux-conserving)
    # ------------------------------------------------------------------
    input_hdu = fits.PrimaryHDU(data=data_proc, header=current_wcs.to_header())

    reprojected, footprint = reproject_exact(
        input_hdu, target_wcs, shape_out=target_shape,
    )

    # ------------------------------------------------------------------
    # 5.  Mask invalid / zero-footprint pixels with NaN
    # ------------------------------------------------------------------
    reprojected = np.where(footprint > 0, reprojected, np.nan)
    reprojected[~np.isfinite(reprojected)] = np.nan

    return reprojected, target_wcs


# ======================================================================
#  Private helpers for reading different input formats
# ======================================================================

def _read_healpix(
    input_map, coord,
    center_l, center_b, padded_size_deg, pixel_size_arcmin,
):
    """
    Take a 1-D HEALPix pixel array and project it onto an intermediate
    flat-sky grid (gnomonic / TAN projection in Galactic coordinates).

    The intermediate pixel scale is set to half the smaller of the original
    HEALPix pixel size and the requested output pixel size, so that the
    HEALPix resolution is well-sampled.

    Parameters
    ----------
    input_map : 1-D numpy.ndarray
        HEALPix pixel values.
    coord : str
        Coordinate system of the HEALPix map ('G', 'C', or 'E').

    Returns
    -------
    data : 2-D ndarray
    wcs  : WCS
    pixel_scale_arcmin : float
    """
    hpx_data = np.asarray(input_map, dtype=np.float64).ravel()

    nside = hp.npix2nside(len(hpx_data))
    hpx_pixel_arcmin = np.degrees(hp.nside2resol(nside)) * 60.0

    # Intermediate pixel scale: well-sample the original HEALPix grid
    intermediate_pix = min(hpx_pixel_arcmin, pixel_size_arcmin) / 2.0

    # --- build intermediate WCS ---
    inter_wcs, inter_shape = _create_intermediate_wcs(
        center_l, center_b, padded_size_deg, intermediate_pix,
    )

    # --- reproject from HEALPix using nearest-neighbour sampling ---
    # (smoothing is done afterwards on the flat grid)
    hpx_input = (hpx_data, coord)
    data, footprint = reproject_from_healpix(
        hpx_input, inter_wcs, shape_out=inter_shape,
        order="nearest-neighbor", nested=False,
    )
    data[footprint == 0] = np.nan

    return data, inter_wcs, intermediate_pix


def _read_fits(data, wcs, center_l, center_b, padded_size_deg):
    """
    Extract a padded cutout from a FITS map (provided as array + WCS).

    The original pixel scale is read directly from the provided WCS.

    Parameters
    ----------
    data : 2-D (or higher) ndarray
        Pixel data of the original FITS map.
    wcs : astropy.wcs.WCS
        WCS of the original FITS map (used to determine pixel scale and
        to convert sky coordinates to pixel positions).

    Returns
    -------
    data : 2-D ndarray
    wcs  : WCS
    pixel_scale_arcmin : float
    """
    raw_data = np.asarray(data, dtype=np.float64)
    raw_wcs = wcs

    # Collapse extra dimensions (e.g. freq / Stokes)
    while raw_data.ndim > 2:
        raw_data = raw_data[0]

    # Pixel scale — read from WCS, no user input required
    pixel_scales_deg = proj_plane_pixel_scales(raw_wcs)  # deg per pixel
    pixel_scale_arcmin = np.mean(pixel_scales_deg) * 60.0

    # --- cut out a padded region around the requested centre ---
    center = SkyCoord(l=center_l * u.deg, b=center_b * u.deg, frame="galactic")

    # Convert Galactic → whatever frame the map WCS uses
    try:
        pix_x, pix_y = raw_wcs.world_to_pixel(center)
    except Exception:
        center_icrs = center.icrs
        pix_x, pix_y = raw_wcs.world_to_pixel(center_icrs)

    size_pix = int(np.ceil(padded_size_deg / np.mean(pixel_scales_deg)))
    position = (float(pix_x), float(pix_y))

    try:
        from astropy.nddata.utils import PartialOverlapError
        cutout = Cutout2D(
            raw_data, position=position, size=size_pix,
            wcs=raw_wcs, mode="partial", fill_value=np.nan,
        )
        data_out = cutout.data
        wcs_out = cutout.wcs
    except PartialOverlapError:
        # Centre is outside the map — return NaN array with original WCS
        warnings.warn(
            "Cutout region falls outside the input FITS map; "
            "returning NaN-filled array.",
            stacklevel=3,
        )
        data_out = np.full((size_pix, size_pix), np.nan)
        wcs_out = raw_wcs

    return data_out, wcs_out, pixel_scale_arcmin
