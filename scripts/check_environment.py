"""
ASTRA — Environment Validation Script

Reports: Python, PyTorch, CUDA, GPU, CPU, RAM, all dependencies.
Usage:  python scripts/check_environment.py
"""
from __future__ import annotations
import importlib
import json
import os
import platform
import struct
import sys
from pathlib import Path

REQUIRED = {
    "torch": "PyTorch (core neural network runtime)",
    "torchvision": "TorchVision (image transforms, pretrained models)",
    "ultralytics": "Ultralytics YOLO (object detection)",
    "cv2": "OpenCV (image/video I/O)",
    "numpy": "NumPy (array operations)",
    "yaml": "PyYAML (config loading)",
    "PIL": "Pillow (image utilities)",
}

OPTIONAL = {
    "timm": "timm (ConvNeXt and other vision backbones)",
    "rtmlib": "rtmlib (RTMPose inference)",
    "onnxruntime": "ONNX Runtime (CPU/GPU model export)",
    "mamba_ssm": "Mamba-2 SSM (temporal model, GPU only)",
    "causal_conv1d": "causal-conv1d (Mamba dependency)",
    "psutil": "psutil (memory monitoring)",
    "tqdm": "tqdm (progress bars)",
    "matplotlib": "matplotlib (visualisation)",
    "scipy": "scipy (statistics)",
}


def _try_import(name: str):
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", getattr(mod, "VERSION", "installed"))
        return True, str(ver)
    except ImportError:
        return False, None


def check_environment() -> dict:
    info = {}

    # ---- Python ----
    info["python"] = {
        "version": platform.python_version(),
        "executable": sys.executable,
        "arch": f"{struct.calcsize('P') * 8}-bit",
        "platform": platform.platform(),
    }

    # ---- CPU ----
    info["cpu"] = {
        "processor": platform.processor() or "unknown",
        "cores_logical": os.cpu_count(),
    }

    # ---- RAM ----
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram"] = {
            "total_gb": round(vm.total / 1024**3, 1),
            "available_gb": round(vm.available / 1024**3, 1),
            "percent_used": vm.percent,
        }
    except ImportError:
        info["ram"] = {"note": "psutil not installed — cannot measure RAM"}

    # ---- PyTorch / CUDA / GPU ----
    has_torch, torch_ver = _try_import("torch")
    info["pytorch"] = {"installed": has_torch, "version": torch_ver}

    if has_torch:
        import torch
        info["pytorch"]["cuda_available"] = torch.cuda.is_available()
        info["pytorch"]["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
        info["pytorch"]["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
        info["pytorch"]["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if torch.cuda.is_available():
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "memory_gb": round(torch.cuda.get_device_properties(0).total_mem / 1024**3, 1),
                "compute_capability": f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}",
            }
        else:
            info["gpu"] = {"status": "no_cuda_gpu"}
    else:
        info["pytorch"]["cuda_available"] = False
        info["gpu"] = {"status": "pytorch_not_installed"}

    # ---- Required dependencies ----
    info["required_deps"] = {}
    missing_required = []
    for pkg, desc in REQUIRED.items():
        ok, ver = _try_import(pkg)
        info["required_deps"][pkg] = {"installed": ok, "version": ver, "description": desc}
        if not ok:
            missing_required.append(pkg)

    # ---- Optional dependencies ----
    info["optional_deps"] = {}
    missing_optional = []
    for pkg, desc in OPTIONAL.items():
        ok, ver = _try_import(pkg)
        info["optional_deps"][pkg] = {"installed": ok, "version": ver, "description": desc}
        if not ok:
            missing_optional.append(pkg)

    info["missing_required"] = missing_required
    info["missing_optional"] = missing_optional

    return info


def print_report(info: dict):
    print("=" * 64)
    print("  ASTRA Environment Report")
    print("=" * 64)

    py = info["python"]
    print(f"\n  Python:         {py['version']} ({py['arch']})")
    print(f"  Executable:     {py['executable']}")
    print(f"  Platform:       {py['platform']}")

    cpu = info["cpu"]
    print(f"\n  CPU:            {cpu['processor']}")
    print(f"  Logical cores:  {cpu['cores_logical']}")

    ram = info.get("ram", {})
    if "total_gb" in ram:
        print(f"\n  RAM:            {ram['total_gb']} GB total, {ram['available_gb']} GB free ({ram['percent_used']}% used)")
    else:
        print(f"\n  RAM:            {ram.get('note', 'unknown')}")

    pt = info["pytorch"]
    print(f"\n  PyTorch:        {'v' + pt['version'] if pt['installed'] else '❌ NOT INSTALLED'}")
    if pt["installed"]:
        print(f"  CUDA available: {'✅ ' + str(pt.get('cuda_version', '')) if pt['cuda_available'] else '❌ No'}")
        print(f"  cuDNN:          {pt.get('cudnn_version', 'N/A')}")

    gpu = info.get("gpu", {})
    if "name" in gpu:
        print(f"\n  GPU:            {gpu['name']}")
        print(f"  VRAM:           {gpu['memory_gb']} GB")
        print(f"  Compute:        {gpu['compute_capability']}")
    else:
        print(f"\n  GPU:            {gpu.get('status', 'none')}")

    print("\n  Required Dependencies:")
    for pkg, d in info["required_deps"].items():
        status = f"✅ v{d['version']}" if d["installed"] else "❌ MISSING"
        print(f"    {pkg:20s} {status:30s}  {d['description']}")

    print("\n  Optional Dependencies:")
    for pkg, d in info["optional_deps"].items():
        status = f"✅ v{d['version']}" if d["installed"] else "— not installed"
        print(f"    {pkg:20s} {status:30s}  {d['description']}")

    print("\n" + "-" * 64)
    mr = info["missing_required"]
    if mr:
        print(f"\n  ⚠️  MISSING REQUIRED ({len(mr)}):")
        pip_map = {"torch": "torch torchvision", "torchvision": "", "cv2": "opencv-python",
                   "yaml": "pyyaml", "PIL": "Pillow"}
        for pkg in mr:
            pip_name = pip_map.get(pkg, pkg)
            if pip_name:
                print(f"      pip install {pip_name}")
    else:
        print("\n  ✅ All required dependencies installed.")

    mo = info["missing_optional"]
    if mo:
        print(f"\n  Optional missing ({len(mo)}):")
        pip_map = {"timm": "timm", "rtmlib": "rtmlib", "onnxruntime": "onnxruntime",
                   "mamba_ssm": "mamba-ssm causal-conv1d", "causal_conv1d": "",
                   "psutil": "psutil", "tqdm": "tqdm", "matplotlib": "matplotlib",
                   "scipy": "scipy"}
        for pkg in mo:
            pip_name = pip_map.get(pkg, pkg)
            if pip_name:
                print(f"      pip install {pip_name}")

    print("\n" + "=" * 64)


def main():
    info = check_environment()
    print_report(info)

    # Save JSON
    out_dir = Path(__file__).resolve().parent.parent / "eval_results"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "environment.json"
    with open(json_path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"\n  JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
