// 256×256 segmentation masks as row-major bits, LSB first.
// Bit (y * width + x) is the least significant bit of byte (bit >> 3).
// 1 = object, 0 = background. y grows downward in image space.

export function emptyMask(width, height) {
  return new Uint8Array((width * height) >> 3);
}

export function setBit(bytes, width, x, y) {
  const bit = y * width + x;
  bytes[bit >> 3] |= 1 << (bit & 7);
}

export function maskBit(bytes, width, x, y) {
  const bit = y * width + x;
  return (bytes[bit >> 3] >> (bit & 7)) & 1;
}

export function countBits(bytes) {
  let n = 0;
  for (let i = 0; i < bytes.length; i++) {
    let v = bytes[i];
    while (v) {
      n += v & 1;
      v >>>= 1;
    }
  }
  return n;
}

export function bytesToBase64(bytes) {
  let bin = "";
  const chunk = 4096;
  for (let i = 0; i < bytes.length; i += chunk) {
    const end = Math.min(i + chunk, bytes.length);
    let part = "";
    for (let j = i; j < end; j++) part += String.fromCharCode(bytes[j]);
    bin += part;
  }
  return btoa(bin);
}

// Move foreground pixels by integer detector pixels. Pixels that leave the
// image are dropped; nothing wraps around.
export function shiftMask(bytes, width, height, dx, dy) {
  const out = emptyMask(width, height);
  const ix = Math.trunc(dx);
  const iy = Math.trunc(dy);
  if (ix === 0 && iy === 0) return bytes;
  for (let y = 0; y < height; y++) {
    const ny = y + iy;
    if (ny < 0 || ny >= height) continue;
    for (let x = 0; x < width; x++) {
      if (!maskBit(bytes, width, x, y)) continue;
      const nx = x + ix;
      if (nx < 0 || nx >= width) continue;
      setBit(out, width, nx, ny);
    }
  }
  return out;
}

export function maskSpec(bytes, width, height, scale) {
  return {
    width,
    height,
    scale_to_image: scale,
    encoding: "row-major-bits-lsb",
    data: bytesToBase64(bytes),
  };
}
