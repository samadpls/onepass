"""Merge a LoRA adapter into its base, convert to GGUF and quantize.

Conversion uses llama.cpp's convert_hf_to_gguf.py (with its gguf-py and
conversion packages beside it); quantization uses llama-cpp-python's binding."""
import ctypes
import os
import subprocess
import sys
from pathlib import Path

FTYPES = {"q8_0": 7, "q4_k_m": 15, "q5_k_m": 17, "q6_k": 18}  # llama.cpp LLAMA_FTYPE_MOSTLY_*


def merge(base, adapter, out_dir):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(base).save_pretrained(out_dir)
    return Path(out_dir)


def to_gguf(hf_dir, outfile, tools_dir, outtype="f16"):
    tools = Path(tools_dir)
    env = dict(os.environ, PYTHONPATH=f"{tools}{os.pathsep}{tools / 'gguf-py'}")
    subprocess.run([sys.executable, str(tools / "convert_hf_to_gguf.py"), str(hf_dir), "--outfile", str(outfile),
                    "--outtype", outtype], check=True, env=env)
    return Path(outfile)


def quantize(src, dst, kind):
    import llama_cpp

    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = FTYPES[kind]
    if llama_cpp.llama_model_quantize(str(src).encode(), str(dst).encode(), ctypes.byref(params)) != 0:
        raise RuntimeError(f"quantizing {src} to {kind} failed")
    return Path(dst)
