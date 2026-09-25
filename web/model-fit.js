// Full edge lengths of the built-in box primitive, in meters.
export const LAB_BOX_SIZE = [0.5, 0.28, 0.22];

// Physical longest side to apply when testing models/ace.glb.
export const ACE_LENGTH_M = 8.3;

export function longestSide(size) {
  return Math.max(size[0], size[1], size[2]);
}

// Uniform scale that makes the model's longest side equal targetMeters.
export function uniformScaleForLength(naturalLongest, targetMeters) {
  if (!(naturalLongest > 0) || !(targetMeters > 0)) return null;
  return targetMeters / naturalLongest;
}
