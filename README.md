# Polarimetric Imaging
<p align="center">
  <img src="gallery/matchbox.png?raw=truee" width="250">
  <img src="gallery/watergun.png?raw=true" width="250">
  <img src="gallery/glasses.png?raw=true" width="250">
</p>

This repository contains code I wrote for controlling a polarimetric machine camera, and processing its raw data into vibrant false-colour visualisations of light polarisation.

The pixels on a standard RGB camera are covered with a mosaic of differently coloured filters in order to differntiate red, blue, and green light. Similarly, a polarimetric camera works by using a mosaic of polarising filters that alternate between $0$, $45$, $90$, and $135$ degree offsets. 

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
1. Replace generators with async coroutines.
2. Make compatible with torch.compile.
3. Variational demosaicking.

# Requirements
## Software:
1. ArenaSDK: https://thinklucid.com/downloads-hub/.
2. Arena-API Python package: https://thinklucid.com/downloads-hub/.

## Sensors:
One of:
1. https://thinklucid.com/product/triton-5-mp-polarization-camera/.
2. https://thinklucid.com/product/triton-5-0-mp-polarization-model-imx264mzrmyr/.

# Acknowledgements:
1. Colourmaps:
    1. https://www.fabiocrameri.ch/.
    2. https://cgg.mff.cuni.cz/~wilkie/Website/Home_files/polvis_sccg_2010.pdf