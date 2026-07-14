This document contains the CMC_hybrid release history in reverse chronological order.

0.2.1 (Tuesday 14th July 2026)
------------------------------
- Performance and memory optimisations.
- General code maintenance and cleanup.
- Added linting test in CI.

0.2.0 (Friday 15th May 2026)
----------------------------
- Added hybrid modelling for 'sagittal' datasets.
- Expanded code to work with PSOCT and dMRI data in different world spaces (provided with the transformations).
- Code supports linear and non-linear transformations between spaces.
- Restriction to only translational registration has been rectified.
- Optimised methods to use the slidedeck header instead of multiple slices' ones.
- Added new output data types `psoct-fod` and `inplane-dMRI`.
- `Utils` have been expanded to support anisotropic resolution.

0.1.0 (Monday 6th October 2025)
-------------------------------
- Initial release to combine PS-OCT and dMRI fibre orientation data for joint analysis.
- Implementation for 'coronal' dataset only.
- Implementation with only cross-correlation registration method (i.e. translation-only).
