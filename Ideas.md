# Ideas.md

Contains ideas planned out at various stages.

## New Panoptic Segmentation Head

- Learn object classification masks as usual.
- Learn the object centroids as keypoints.
- Each of their pixels have an offset towards the centroid.
- Clustering during postprocess:
    - Number of clusters = number of centroids
    - Pixel XY is offset by its offset.
    - Just do an eager nearest cluster.
    - Background pixels should tend to norm=0.

## New Architecture

- Stagewise refinement worked well in OpenPose.
- Make stages autoregressive.
- Test-time scaling.

## Architecture for video processing

- Utilize key-frames:
- Big network runs of the key-frames (X), every n frames, predicts output t0
- Small diff network:
  - input is sparse dX_t + output at t > 0. 
  
  
## Random idea - computer control

This is a tangential idea.

Gesture-based computer inputs. Since mouse is still the king, try to aid
mouse control with gaze first. Gaze direction + face gesture would be used to
make large scale adjustment to mouse position, then mouse is used for finer
control. Setup could be used to extract data for further refinement trainings.
