import argparse
import ctypes
import json
import math
import os
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from itertools import product
from pathlib import Path
from typing import Callable, Dict, Optional

os.environ["CXX"] = "g++"

import numpy as np
import torch

try:
    from torch._inductor import config as inductor_config
except ModuleNotFoundError as inductor_import_error:
    inductor_config = None
else:
    inductor_import_error = None
from transformers import AutoModelForCausalLM, AutoTokenizer, masking_utils

if inductor_config is not None:
    inductor_config.cpp.cxx = (None, "g++")
    inductor_config.triton.cudagraphs = False

BATCH_SWEEP_CONFIG = {
    "num_cores": [1, 4, 8, 16],
    "prompt": ["hello", "hello, my name is"],
}


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LIBTORCH_RUNNER_SOURCE = SCRIPT_DIR / "libtorch_script_runner.cpp"
DEFAULT_LIBTORCH_BUILD_DIR = REPO_ROOT / "build"
LIBTORCH_RUNNER_NAME = "libtorch_script_runner"
TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile") and inductor_config is not None

if inductor_config is None:
    print(
        "Warning: torch._inductor is unavailable (likely due to missing dependencies); "
        "torch.compile benchmarks will be skipped."
    )

DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float64: 3,
    torch.int64: 4,
    torch.int32: 5,
    torch.int16: 6,
    torch.int8: 7,
    torch.uint8: 8,
    torch.bool: 9,
}

CODE_TO_DTYPE = {code: dtype for dtype, code in DTYPE_TO_CODE.items()}

LIBTORCH_INPUT_IDS_KEY = "libtorch_input_ids.bin"
LIBTORCH_ATTENTION_MASK_KEY = "libtorch_attention_mask.bin"


def resolve_dtype(dtype_str: str) -> torch.dtype:
    normalized = dtype_str.lower()
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "f32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "f16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float64": torch.float64,
        "double": torch.float64,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return aliases[normalized]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TorchScript vs TorchDynamo for Qwen3 0.6B")
    parser.add_argument(
        "--prompt",
        default="Hello, my name is",
        help="Prompt used to create the example input.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model id to benchmark.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of timed iterations for each configuration.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup runs before timing.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run the benchmark on (default: cpu).",
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=None,
        help="Override the number of CPU cores used by PyTorch threads.",
    )
    parser.add_argument(
        "--run-batch",
        action="store_true",
        help="Run the predefined batch sweep instead of a single configuration.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Model parameter dtype (e.g. float32, bfloat16)",
    )
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="Use Qwen's thinking mode when preparing inputs (default)",
    )
    parser.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable Qwen's thinking mode",
    )
    parser.set_defaults(enable_thinking=True)
    parser.add_argument(
        "--libtorch-build-dir",
        default=None,
        help=(
            "Path to the PyTorch build directory containing libtorch artifacts. "
            "Defaults to <repo root>/build."
        ),
    )
    return parser.parse_args()


def ensure_libtorch_runner(build_dir: Path) -> Path:
    if not build_dir.exists():
        raise FileNotFoundError(f"libtorch build directory not found: {build_dir}")

    runner_source = LIBTORCH_RUNNER_SOURCE
    if not runner_source.exists():
        raise FileNotFoundError(f"Missing C++ runner source file: {runner_source}")

    binary_path = build_dir / "bin" / LIBTORCH_RUNNER_NAME
    binary_path.parent.mkdir(parents=True, exist_ok=True)

    needs_build = not binary_path.exists()
    if not needs_build:
        needs_build = runner_source.stat().st_mtime > binary_path.stat().st_mtime

    if needs_build:
        cxx = os.environ.get("CXX", "g++")
        include_dirs = [
            build_dir / "include",
            REPO_ROOT / "torch" / "include",
            REPO_ROOT / "torch" / "include" / "torch" / "csrc" / "api" / "include",
        ]
        compile_cmd = [
            cxx,
            "-O3",
            "-std=c++17",
            "-DNDEBUG",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
        ]
        for inc in include_dirs:
            if inc.exists():
                compile_cmd.extend(["-I", str(inc)])

        compile_cmd.extend([
            str(runner_source),
            "-o",
            str(binary_path),
        ])

        lib_dir = build_dir / "lib"
        if not lib_dir.exists():
            raise FileNotFoundError(f"Expected lib directory inside build: {lib_dir}")

        compile_cmd.extend(
            [
                "-L",
                str(lib_dir),
                "-Wl,-rpath," + str(lib_dir),
                "-ltorch",
                "-ltorch_cpu",
                "-lc10",
                "-ltorch_global_deps",
                "-pthread",
            ]
        )

        subprocess.run(compile_cmd, check=True, cwd=str(REPO_ROOT))

    return binary_path


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    array = tensor.detach().cpu().contiguous()
    dtype_code = DTYPE_TO_CODE.get(array.dtype)
    if dtype_code is None:
        raise ValueError(f"Unsupported dtype for libtorch serialization: {array.dtype}")

    rank = array.dim()
    header = struct.pack("<ii", dtype_code, rank)
    dims = struct.pack("<" + "q" * rank, *array.shape)
    byte_length = array.element_size() * array.nelement()
    data = ctypes.string_at(array.data_ptr(), byte_length)
    return header + dims + data


def bytes_to_tensor(blob: bytes) -> torch.Tensor:
    if len(blob) < 8:
        raise ValueError("Tensor payload too small")

    dtype_code, rank = struct.unpack_from("<ii", blob, 0)
    dtype = CODE_TO_DTYPE.get(dtype_code)
    if dtype is None:
        raise ValueError(f"Unsupported dtype code from libtorch runner: {dtype_code}")

    offset = 8
    dims = []
    for _ in range(rank):
        (dim,) = struct.unpack_from("<q", blob, offset)
        offset += 8
        dims.append(dim)

    tensor = torch.empty(dims, dtype=dtype)
    expected_bytes = tensor.element_size() * tensor.nelement()
    payload = blob[offset:]
    if len(payload) != expected_bytes:
        raise ValueError(
            f"Tensor payload size mismatch (expected {expected_bytes}, got {len(payload)})"
        )
    if expected_bytes:
        ctypes.memmove(tensor.data_ptr(), payload, expected_bytes)
    return tensor


def run_libtorch_mode(
    label: str,
    scripted_module: torch.jit.ScriptModule,
    example_inputs: tuple[torch.Tensor, torch.Tensor],
    warmup: int,
    iterations: int,
    build_dir: Path,
    num_threads: Optional[int],
) -> Dict[str, float]:
    runner_path = ensure_libtorch_runner(build_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        module_path = tmpdir_path / "module.pt"
        output_path = tmpdir_path / "output.pt"
        extra_files = {
            LIBTORCH_INPUT_IDS_KEY: tensor_to_bytes(example_inputs[0]),
            LIBTORCH_ATTENTION_MASK_KEY: tensor_to_bytes(example_inputs[1]),
        }
        torch.jit.save(scripted_module, module_path, _extra_files=extra_files)

        env = os.environ.copy()
        lib_dir = build_dir / "lib"
        ld_paths = [str(lib_dir)]
        if env.get("LD_LIBRARY_PATH"):
            ld_paths.append(env["LD_LIBRARY_PATH"])
        env["LD_LIBRARY_PATH"] = ":".join(ld_paths)

        cmd = [
            str(runner_path),
            str(module_path),
            str(warmup),
            str(iterations),
            str(num_threads if num_threads and num_threads > 0 else 0),
            str(output_path),
        ]
        try:
            completed = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=env,
                cwd=str(REPO_ROOT),
            )
        except subprocess.CalledProcessError as err:
            stderr = err.stderr.strip()
            stdout = err.stdout.strip()
            detail_lines = []
            if stdout:
                detail_lines.append("stdout:\n" + stdout)
            if stderr:
                detail_lines.append("stderr:\n" + stderr)
            detail = "\n".join(detail_lines)
            raise RuntimeError(
                "libtorch runner failed with non-zero exit status" +
                (f":\n{detail}" if detail else "")
            ) from err

        stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not stdout_lines:
            raise RuntimeError("libtorch runner produced no output")

        try:
            payload = json.loads(stdout_lines[-1])
        except json.JSONDecodeError as err:
            raise RuntimeError(
                "Failed to parse libtorch runner output:\n" + completed.stdout
            ) from err

        timings = [float(value) for value in payload.get("timings_s", [])]
        output_blob = Path(payload["output_path"]).read_bytes()
        result_tensor = bytes_to_tensor(output_blob)

        mean = statistics.mean(timings) if timings else 0.0
        stdev = statistics.stdev(timings) if len(timings) > 1 else 0.0
        print(f"{label:>12}: {mean * 1000:.2f} ms ± {stdev * 1000:.2f} ms (n={iterations})")

        return summarize_timings(label, timings, result_tensor)

def prepare_inputs(
    tokenizer,
    prompt: str,
    device: torch.device,
    enable_thinking: bool,
) -> Dict[str, torch.Tensor]:
    encoded = None

    if enable_thinking:
        apply_chat = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_chat):
            messages = [{"role": "user", "content": prompt}]
            try:
                text = apply_chat(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=True,
                )
            except TypeError:
                text = apply_chat(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            encoded = tokenizer(text, return_tensors="pt")

    if encoded is None:
        encoded = tokenizer(prompt, return_tensors="pt")

    return {k: v.to(device) for k, v in encoded.items()}


def install_torchscript_friendly_mask() -> None:
    """Patch masking_utils to avoid sdpa mask helpers that break tracing."""

    def eager_mask(
        batch_size: int,
        cache_position: torch.Tensor,
        kv_length: int,
        kv_offset: int = 0,
        mask_function=None,
        attention_mask: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> torch.Tensor:
        q_positions = cache_position.unsqueeze(-1)
        kv_positions = torch.arange(kv_length, device=cache_position.device)
        base_mask = kv_positions.unsqueeze(0) <= q_positions
        base_mask = base_mask.unsqueeze(0).expand(batch_size, -1, -1)

        if attention_mask is not None:
            slice_end = min(attention_mask.size(-1), kv_offset + kv_length)
            attn_slice = attention_mask[:, kv_offset:slice_end]
            if attn_slice.size(-1) != kv_length:
                pad = torch.zeros(
                    (attention_mask.size(0), kv_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                pad[:, : attn_slice.size(-1)] = attn_slice
                attn_slice = pad
            attn_slice = attn_slice.bool().unsqueeze(1).expand_as(base_mask)
            base_mask = base_mask & attn_slice

        zeros = torch.zeros((), dtype=dtype, device=base_mask.device)
        neg_inf = torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=base_mask.device)
        return torch.where(base_mask.unsqueeze(1), zeros, neg_inf)

    masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["eager"] = eager_mask


class CausalLMForwardWrapper(torch.nn.Module):
    """Wrap a causal LM to return logits tensor only."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )
        return outputs[0]


def run_timed(
    run_callable: Callable[[], torch.Tensor],
    label: str,
    warmup: int,
    iterations: int,
) -> tuple[list[float], float, float, torch.Tensor]:
    for _ in range(warmup):
        run_callable()

    timings: list[float] = []
    last_result: Optional[torch.Tensor] = None
    for _ in range(iterations):
        start = time.perf_counter()
        result = run_callable()
        timings.append(time.perf_counter() - start)
        last_result = result.detach()

    mean = statistics.mean(timings) if timings else 0.0
    stdev = statistics.stdev(timings) if len(timings) > 1 else 0.0
    print(f"{label:>12}: {mean * 1000:.2f} ms ± {stdev * 1000:.2f} ms (n={iterations})")
    return timings, mean, stdev, last_result if last_result is not None else torch.empty(0)


def summarize_timings(
    label: str,
    timings: list[float],
    output: torch.Tensor,
) -> Dict[str, float]:
    total_outputs = 0
    tokens_per_sequence = 0
    if output.numel() > 0:
        total_outputs = output.shape[0] * output.shape[1]
        tokens_per_sequence = output.shape[1]

    first_latency = timings[0] if timings else float("nan")
    remaining_timings = timings[1:] if len(timings) > 1 else []
    avg_time_ex_first = (
        statistics.mean(remaining_timings) if remaining_timings else float("nan")
    )
    avg_per_token = (
        avg_time_ex_first / tokens_per_sequence
        if tokens_per_sequence > 0 and not math.isnan(avg_time_ex_first)
        else float("nan")
    )
    tokens_per_second = (
        (tokens_per_sequence / avg_time_ex_first)
        if tokens_per_sequence > 0 and avg_time_ex_first and not math.isnan(avg_time_ex_first)
        else float("nan")
    )
    for i, timing_value in enumerate(timings):
        print(f"  time {i}: {timing_value * 1000:.2f} ms")

    print(
        f"{label:>12} details - total outputs: {total_outputs}, "
        f"first token latency: {first_latency * 1000:.2f} ms, "
        f"avg/token (excl first): {avg_per_token * 1000:.2f} ms, "
        f"tokens/s: {tokens_per_second:.2f}"
    )

    return {
        "total_outputs": float(total_outputs),
        "first_token_latency_s": first_latency,
        "avg_latency_excl_first_s": avg_time_ex_first,
        "avg_per_token_latency_s": avg_per_token,
        "tokens_per_second": tokens_per_second,
    }


def configure_threads(num_cores: Optional[int]) -> None:
    if num_cores is None or num_cores <= 0:
        return
    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(max(1, num_cores))


def run_single_mode(
    label: str,
    runner: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    example_inputs: tuple[torch.Tensor, torch.Tensor],
    warmup: int,
    iterations: int,
) -> Dict[str, float]:
    zero_arg = lambda: runner(*example_inputs)
    timings, _, _, output = run_timed(zero_arg, label, warmup, iterations)
    return summarize_timings(label, timings, output)


def benchmark_triplet(
    prompt: str,
    num_cores: Optional[int],
    model_id: str,
    model_dtype: torch.dtype,
    enable_thinking: bool,
    iterations: int,
    warmup: int,
    device: torch.device,
    libtorch_build_dir: Path,
) -> Dict[str, Dict[str, float]]:
    configure_threads(num_cores)

    install_torchscript_friendly_mask()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=model_dtype,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.eval()

    wrapper = CausalLMForwardWrapper(model)
    wrapper.eval()

    inputs = prepare_inputs(tokenizer, prompt, device, enable_thinking)
    example_inputs = (inputs["input_ids"], inputs["attention_mask"])

    results: Dict[str, Dict[str, float]] = {}

    eager_runner = lambda input_ids, attention_mask: wrapper(input_ids, attention_mask)
    results["Eager"] = run_single_mode(
        "Eager", eager_runner, example_inputs, warmup, iterations
    )

    scripted = torch.jit.trace(wrapper, example_inputs, strict=False)
    scripted = torch.jit.freeze(scripted)
    if device.type == "cpu":
        scripted_cpu = scripted.cpu()
        cpu_inputs = (example_inputs[0].cpu(), example_inputs[1].cpu())
        results["TorchScript"] = run_libtorch_mode(
            "TorchScript",
            scripted_cpu,
            cpu_inputs,
            warmup,
            iterations,
            libtorch_build_dir,
            num_threads=num_cores,
        )
    else:
        print(
            "TorchScript libtorch runner currently supports CPU devices only; "
            "falling back to Python execution."
        )
        script_runner = lambda input_ids, attention_mask: scripted(
            input_ids, attention_mask
        )
        results["TorchScript"] = run_single_mode(
            "TorchScript", script_runner, example_inputs, warmup, iterations
        )

    if TORCH_COMPILE_AVAILABLE:
        try:
            compiled = torch.compile(wrapper, backend="inductor")
            compiled_runner = lambda input_ids, attention_mask: compiled(
                input_ids, attention_mask
            )
            results["torch.compile"] = run_single_mode(
                "torch.compile",
                compiled_runner,
                example_inputs,
                warmup + 1,
                iterations,
            )
        except Exception as compile_error:
            print(f"torch.compile benchmark failed: {compile_error}")
    else:
        print(
            "Skipping torch.compile benchmark because torch._inductor is unavailable."
        )

    return results


def run_batch_configs(
    model_id: str,
    iterations: int,
    warmup: int,
    device: torch.device,
    dtype: str,
    enable_thinking: bool,
    libtorch_build_dir: Path,
) -> None:
    script_path = os.path.abspath(__file__)
    device_str = str(device)
    for num_core, prompt in product(
        BATCH_SWEEP_CONFIG["num_cores"], BATCH_SWEEP_CONFIG["prompt"]
    ):
        print("=" * 80)
        print(f"Batch config -> prompt: {prompt!r}, num_cores: {num_core}")
        cmd = [
            sys.executable,
            script_path,
            "--prompt",
            prompt,
            "--model",
            model_id,
            "--iterations",
            str(iterations),
            "--warmup",
            str(warmup),
            "--device",
            device_str,
            "--num-cores",
            str(num_core),
            "--dtype",
            dtype,
            "--libtorch-build-dir",
            str(libtorch_build_dir),
        ]
        cmd.append("--enable-thinking" if enable_thinking else "--disable-thinking")
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model_dtype = resolve_dtype(args.dtype)

    if args.libtorch_build_dir:
        libtorch_build_dir = Path(args.libtorch_build_dir).expanduser()
    else:
        libtorch_build_dir = DEFAULT_LIBTORCH_BUILD_DIR
    libtorch_build_dir = libtorch_build_dir.resolve()
    if not libtorch_build_dir.exists():
        raise FileNotFoundError(
            f"libtorch build directory not found: {libtorch_build_dir}"
        )

    torch.set_grad_enabled(False)
    torch._dynamo.config.suppress_errors = False

    if args.run_batch:
        run_batch_configs(
            model_id=args.model,
            iterations=args.iterations,
            warmup=args.warmup,
            device=device,
            dtype=args.dtype,
            enable_thinking=args.enable_thinking,
            libtorch_build_dir=libtorch_build_dir,
        )
    else:
        print(
            f"Benchmarking {args.model} on {device} using prompt: {args.prompt!r} "
            f"with num_cores={args.num_cores or 'default'}"
        )

        results = benchmark_triplet(
            prompt=args.prompt,
            num_cores=args.num_cores,
            model_id=args.model,
            model_dtype=model_dtype,
            enable_thinking=args.enable_thinking,
            iterations=args.iterations,
            warmup=args.warmup,
            device=device,
            libtorch_build_dir=libtorch_build_dir,
        )

        print()
        for mode, metrics in results.items():
            print(f"{mode:>12} summary: {metrics}")


if __name__ == "__main__":
    main()
