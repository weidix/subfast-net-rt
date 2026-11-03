const H = __REGION_H__u;
const W = __REGION_W__u;
const THREADS = 128u;

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> evidence: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> maxima: array<f32, THREADS>;

@compute @workgroup_size(THREADS)
fn main(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
    let base = group.y * H * W;
    var best = -10000.0;
    for (var pixel = lane; pixel < H * W; pixel += THREADS) {
        let y = pixel / W;
        let x = pixel % W;
        let y0 = max(y, 2u) - 2u;
        let x0 = max(x, 2u) - 2u;
        let y1 = min(y + 3u, H);
        let x1 = min(x + 3u, W);
        var sum = 0.0;
        for (var yy = y0; yy < y1; yy += 1u) {
            for (var xx = x0; xx < x1; xx += 1u) {
                sum += logits[base + yy * W + xx];
            }
        }
        best = max(best, sum / f32((y1 - y0) * (x1 - x0)));
    }
    maxima[lane] = best;
    workgroupBarrier();
    for (var stride = THREADS / 2u; stride != 0u; stride >>= 1u) {
        if (lane < stride) {
            maxima[lane] = max(maxima[lane], maxima[lane + stride]);
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        output[group.y] = fma(evidence[0], maxima[0], evidence[1]);
    }
}
