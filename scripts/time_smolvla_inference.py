"""Time SmolVLA chunk inference on this GPU: total, VLM prefix, and per flow step.

Uses our camera setup (two 640x480 colour images, SmolVLA pads to 512x512), a 6-D state and a
short task prompt; weights from lerobot/smolvla_base (a fine-tune has the same architecture, so
the same latency). The prefix (images + language + state through the VLM, filling the KV cache)
runs once per chunk; each flow step runs the action expert against that cache. Timing a run with
1 flow step and one with ``num_steps`` = N gives both:

  total(N) = prefix + N * step   ->   step = (total(N) - total(1)) / (N - 1)

  uv run --no-sync python scripts/time_smolvla_inference.py   # -> outputs/smolvla_latency.json
  ... --batch 1 4 8 --precision native float32 --reps 30

Precision: "native" is the checkpoint as LeRobot loads it (VLM and action expert in bfloat16,
small projections in float32), the deployment setting; "float32" upcasts every weight.

GPU only; never touches cameras or motors. One heavy GPU job at a time.
"""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ("observation.images.camera1", "observation.images.camera2")
TASK = "Pick up the red block and place it in the bowl.\n"


def load(precision: str):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    # Our fine-tune sees two cameras (the base checkpoint was trained with three).
    policy.config.input_features.pop("observation.images.camera3", None)
    policy.config.input_features.update(
        {k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)) for k in CAMERAS}
    )
    if precision == "float32":
        policy = policy.to(dtype=torch.float32)
    return policy.to("cuda").eval()


def make_batch(policy, bsize: int) -> dict:
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    tokens = tokenizer(
        [TASK] * bsize,
        padding="max_length",
        max_length=policy.config.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    gen = torch.Generator().manual_seed(0)
    batch = {k: torch.rand(bsize, 3, 480, 640, generator=gen) for k in CAMERAS}
    batch["observation.state"] = torch.randn(bsize, 6, generator=gen)
    batch = {k: v.to("cuda") for k, v in batch.items()}
    batch["observation.language.tokens"] = tokens["input_ids"].to("cuda")
    batch["observation.language.attention_mask"] = tokens["attention_mask"].to("cuda").bool()
    return batch


def time_chunks(policy, batch, num_steps: int, reps: int, warmup: int = 5) -> list[float]:
    policy.config.num_steps = num_steps
    out = []
    for i in range(warmup + reps):
        policy.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        if i >= warmup:
            out.append((time.perf_counter() - t0) * 1000)
    return out


def summary(ms: list[float]) -> dict:
    return {
        "median_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "min_ms": float(np.min(ms)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--precision", nargs="+", default=["native", "float32"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "smolvla_latency.json")
    args = parser.parse_args()

    results = []
    for precision in args.precision:
        policy = load(precision)
        for bsize in args.batch:
            batch = make_batch(policy, bsize)
            with torch.inference_mode():
                full = time_chunks(policy, batch, args.steps, args.reps)
                one = time_chunks(policy, batch, 1, args.reps)
            step = (np.median(full) - np.median(one)) / (args.steps - 1)
            row = {
                "precision": precision,
                "batch": bsize,
                "num_steps": args.steps,
                "chunk": summary(full),
                "one_flow_step": summary(one),
                "prefix_ms": float(np.median(one) - step),
                "per_flow_step_ms": float(step),
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
            results.append(row)
            print(
                f"{precision:7s} batch {bsize}: chunk {row['chunk']['median_ms']:6.1f} ms "
                f"(p95 {row['chunk']['p95_ms']:6.1f}), prefix "
                f"{row['prefix_ms']:6.1f} ms, per flow step {step:5.1f} ms, "
                f"peak {row['peak_mem_gb']:.2f} GB",
                flush=True,
            )
            torch.cuda.reset_peak_memory_stats()
            save(args.out, results)
        del policy
        torch.cuda.empty_cache()
    print(f"Saved {args.out}")


def save(out: Path, results: list[dict]) -> None:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": "lerobot/smolvla_base",
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "host": platform.node(),
                "git_sha": sha,
                "cameras": list(CAMERAS),
                "image_hw": [480, 640],
                "results": results,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
