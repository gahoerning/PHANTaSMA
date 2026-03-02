"""
PHANTaSMA — Standard astronomy map preprocessing utilities.

Usage
-----
    import phantasma as ph
    result, wcs = ph.preprocess_and_cutout(...)

    from phantasma import cutoff_processing
    from phantasma.cutoff_processing import preprocess_and_cutout
"""

from .cutoff_processing import smooth_cutout, make_target_wcs

__all__ = ["smooth_cutout", "make_target_wcs"]
