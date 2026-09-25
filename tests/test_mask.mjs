import assert from "node:assert/strict";
import test from "node:test";
import { bytesToBase64, countBits, emptyMask, maskBit, maskSpec, setBit, shiftMask } from "../web/mask.js";

test("segmentation bits are row-major and LSB first", () => {
  const width = 256;
  const height = 256;
  const bytes = emptyMask(width, height);
  assert.equal(bytes.length, 8192);
  setBit(bytes, width, 3, 5);
  setBit(bytes, width, 0, 0);
  assert.equal(maskBit(bytes, width, 3, 5), 1);
  assert.equal(maskBit(bytes, width, 0, 0), 1);
  assert.equal(maskBit(bytes, width, 4, 5), 0);
  assert.equal(maskBit(bytes, width, 255, 255), 0);
  const bit = 5 * width + 3;
  assert.equal((bytes[bit >> 3] >> (bit & 7)) & 1, 1);
  assert.equal(countBits(bytes), 2);
  const spec = maskSpec(bytes, width, height, 4);
  assert.equal(spec.encoding, "row-major-bits-lsb");
  assert.equal(spec.scale_to_image, 4);
  assert.equal(spec.data, Buffer.from(bytes).toString("base64"));
  assert.equal(bytesToBase64(bytes), spec.data);
});

test("a filled block counts interior pixels, not an outline", () => {
  const width = 256;
  const bytes = emptyMask(width, 256);
  for (let y = 40; y < 80; y++) {
    for (let x = 10; x < 50; x++) setBit(bytes, width, x, y);
  }
  assert.equal(countBits(bytes), 40 * 40);
  assert.equal(maskBit(bytes, width, 30, 60), 1);
});

test("mask shift moves a pixel and does not wrap", () => {
  const width = 256;
  const bytes = emptyMask(width, 256);
  setBit(bytes, width, 3, 5);
  setBit(bytes, width, 255, 255);
  const moved = shiftMask(bytes, width, 256, 2, -1);
  assert.equal(maskBit(moved, width, 5, 4), 1);
  assert.equal(maskBit(moved, width, 3, 5), 0);
  assert.equal(maskBit(moved, width, 0, 0), 0);
  assert.equal(maskBit(moved, width, 255, 255), 0);
  assert.equal(countBits(moved), 1);
  assert.equal(shiftMask(bytes, width, 256, 0, 0), bytes);
});
