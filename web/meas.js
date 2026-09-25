// Minimum-area rectangle of a convex hull in image space (Y down).
// theta is the edge angle from atan2, which is the angle canvas.rotate expects.

export function obbFromHull(hull) {
  let best = null;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i];
    const b = hull[(i + 1) % hull.length];
    const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
    const c = Math.cos(-ang);
    const s = Math.sin(-ang);
    let minx = Infinity;
    let maxx = -Infinity;
    let miny = Infinity;
    let maxy = -Infinity;
    for (const p of hull) {
      const x = p[0] * c - p[1] * s;
      const y = p[0] * s + p[1] * c;
      if (x < minx) minx = x;
      if (x > maxx) maxx = x;
      if (y < miny) miny = y;
      if (y > maxy) maxy = y;
    }
    const area = (maxx - minx) * (maxy - miny);
    if (!best || area < best.area) best = { area, ang, minx, maxx, miny, maxy };
  }
  const cxR = (best.minx + best.maxx) / 2;
  const cyR = (best.miny + best.maxy) / 2;
  const c = Math.cos(best.ang);
  const s = Math.sin(best.ang);
  return {
    cx: cxR * c - cyR * s,
    cy: cxR * s + cyR * c,
    width: best.maxx - best.minx,
    height: best.maxy - best.miny,
    theta: best.ang,
  };
}
