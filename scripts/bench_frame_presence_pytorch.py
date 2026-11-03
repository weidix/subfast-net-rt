"""Measure the sibling project's Frame Presence checkpoint on preprocessed tensors.

Set PYTHONPATH to the sibling project's src directory and run with its Python.
The input generator and warmup/measurement units match gpu-bench.
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from subfast_frame_presence.inference import (
    checkpoint_default_input_size,
    load_frame_presence_inference_checkpoint,
)


def make_input(batch, height, width):
    values = np.empty(batch * 3 * height * width, dtype=np.float32)
    state = 0x9E3779B9
    for index in range(values.size):
        state = (state * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF
        values[index] = (state >> 8) * (2.0 / 16_777_216.0) - 1.0
    return torch.from_numpy(values.reshape(batch, 3, height, width))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()
    batches = [int(batch) for batch in args.batch_sizes.split(",")]
    if (
        min(batches) <= 0
        or min(args.iterations, args.rounds) <= 0
        or args.warmup < 0
        or (args.threads is not None and args.threads <= 0)
    ):
        parser.error("batch sizes, threads, iterations and rounds must be positive; warmup can be zero")

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model, checkpoint = load_frame_presence_inference_checkpoint(args.checkpoint, device)
    width, height = checkpoint_default_input_size(checkpoint)
    print(f"pytorch={torch.__version__} device={device} threads={torch.get_num_threads()}")

    for batch in batches:
        host = make_input(batch, height, width)
        warmup_calls = (args.warmup + batch - 1) // batch
        calls = (args.iterations + batch - 1) // batch
        for _ in range(warmup_calls):
            model(host.to(device)).cpu()

        elapsed = 0.0
        for round_index in range(args.rounds):
            round_calls = calls // args.rounds + (round_index < calls % args.rounds)
            start = perf_counter()
            for _ in range(round_calls):
                # Keep the host-to-device upload and synchronous logit readback
                # inside the measurement, as in the Rust GPU path.
                model(host.to(device)).cpu()
            elapsed += perf_counter() - start
        print(
            f"batch={batch} samples={calls * batch} "
            f"ms={elapsed * 1_000 / calls:.4f} samples_per_second={calls * batch / elapsed:.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
