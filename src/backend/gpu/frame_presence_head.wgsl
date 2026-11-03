const REGION_H = __REGION_H__u;
const REGION_W = __REGION_W__u;
const CONTEXT_H = __CONTEXT_H__u;
const CONTEXT_W = __CONTEXT_W__u;

@group(0) @binding(0) var<storage, read> region: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> context: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> head: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> logits: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= REGION_W * REGION_H) {
        return;
    }
    let position = vec2<f32>(f32(id.x % REGION_W), f32(id.x / REGION_W));
    let ratio = vec2<f32>(f32(CONTEXT_W) / f32(REGION_W), f32(CONTEXT_H) / f32(REGION_H));
    let tap = max(fma(position + 0.5, ratio, vec2<f32>(-0.5)), vec2<f32>(0.0));
    let lo = vec2<u32>(tap);
    let hi = min(lo + 1u, vec2<u32>(CONTEXT_W - 1u, CONTEXT_H - 1u));
    let alpha = fract(tap);
    let sample_base = id.y * CONTEXT_H * CONTEXT_W;
    let upper_left = (sample_base + lo.y * CONTEXT_W + lo.x) * 12u;
    let upper_right = (sample_base + lo.y * CONTEXT_W + hi.x) * 12u;
    let lower_left = (sample_base + hi.y * CONTEXT_W + lo.x) * 12u;
    let lower_right = (sample_base + hi.y * CONTEXT_W + hi.x) * 12u;
    let pixel = id.y * REGION_W * REGION_H + id.x;
    var logit = head[20].x;
    for (var c = 0u; c < 8u; c += 1u) {
        logit += dot(region[pixel * 8u + c], head[c]);
    }
    for (var c = 0u; c < 12u; c += 1u) {
        let upper = mix(context[upper_left + c], context[upper_right + c], alpha.x);
        let lower = mix(context[lower_left + c], context[lower_right + c], alpha.x);
        logit += dot(mix(upper, lower, alpha.y), head[8u + c]);
    }
    logits[pixel] = logit;
}
