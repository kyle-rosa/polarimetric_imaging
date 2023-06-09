# Polarimetric Imaging

![alt text](https://github.com/kyle-rosa/polarisation_imaging/blob/main/gallery/glasses.png?raw=true)
![alt text](https://github.com/kyle-rosa/polarisation_imaging/blob/main/gallery/watergun.png?raw=true)
![alt text](https://github.com/kyle-rosa/polarisation_imaging/blob/main/gallery/matchbox.png?raw=true)

## Overview
Initialise camera stream and video writer, then loop:
1. Load frame from stream buffer.
2. Demosaic frame and convert to floating point dtype.
3. Generate Stokes parameter visualisations.
4. Display visualisations on screen and write frame to video output file.

## Capture Settings
1. Video streaming.
2. Long exposure.

## Demosaicing and Interpolation
1. Channel alignment and interpolation.

## Visualisation
1. Angular (0, 45, 90, and 135 degrees) intensity.
2. Stokes parameters (I, Q, and U) intensity.
3. Linear polarisation (L = Q + iU) domain colouring.
4. HSV-based Stokes colourmapping.

# TODO
- Replace generators with async coroutines.
- Make compatible with torch.compile.
- Variational demosaicking.

# Requirements
## Software:
- ArenaSDK: https://thinklucid.com/downloads-hub/.
- Arena-API Python package: https://thinklucid.com/downloads-hub/.

## Sensors:
One of:
- https://thinklucid.com/product/triton-5-mp-polarization-camera/.
- https://thinklucid.com/product/triton-5-0-mp-polarization-model-imx264mzrmyr/.

# Acknowledgements:
- Colourmaps:
    - https://www.fabiocrameri.ch/.
    - https://cgg.mff.cuni.cz/~wilkie/Website/Home_files/polvis_sccg_2010.pdf