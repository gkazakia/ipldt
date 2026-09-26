"""pm_moore -- counter-clockwise (on screen) Moore boundary trace of an 8-connected component, entering the
raster-first pixel from the west (the convention that reproduces IPL's gobj start points and step directions)."""
# image-frame steps (dx, dy) for Freeman code c with y up  ==  (DX[c], -DY[c])
STEP = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
CODE = {s: c for c, s in enumerate(STEP)}
# counter-clockwise-on-screen Moore order starting from W: W, SW, S, SE, E, NE, N, NW
CCW = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def _next(sl, x, y, b, order):
    H, W = sl.shape
    for k in range(8):
        d = order[(b + k) % 8]
        nx, ny = x + d[0], y + d[1]
        if 0 <= nx < W and 0 <= ny < H and sl[ny, nx]:
            pd = order[(b + k - 1) % 8]
            return nx, ny, (x + pd[0] - nx, y + pd[1] - ny)
    return None


def moore_trace(sl, start, backtrack=(-1, 0), ccw=True):
    """list of (x, y) boundary pixels (closed; the step back to the start is implicit).
    Stops when the start pixel is re-entered such that the next move would repeat the first move."""
    order = CCW if ccw else [CCW[0]] + CCW[1:][::-1]
    index = {d: i for i, d in enumerate(order)}
    x, y = start
    b = index[backtrack]
    pts = [(x, y)]
    r = _next(sl, x, y, b, order)
    if r is None:
        return pts
    x, y, bt = r; b = index[bt]
    if (x, y) == start:
        return pts
    pts.append((x, y))
    second = (x, y)
    while True:
        r = _next(sl, x, y, b, order)
        x, y, bt = r; b = index[bt]
        if (x, y) == start:
            r2 = _next(sl, x, y, b, order)
            if r2 is not None and (r2[0], r2[1]) == second:
                return pts
        pts.append((x, y))
        if len(pts) > 4 * sl.size + 8:
            raise RuntimeError("runaway trace")


def codes_of(pts):
    return [CODE[(pts[(i + 1) % len(pts)][0] - pts[i][0], pts[(i + 1) % len(pts)][1] - pts[i][1])] for i in range(len(pts))]
