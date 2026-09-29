#!/usr/bin/env python3
"""Launch InfiniteTalk's generate_infinitetalk.py with runtime patches.

Runs inside .venv-infinitetalk with the InfiniteTalk checkout as its working
directory, never imported by make_avatar.py (same subprocess isolation rule as
every other model repo here - see CLAUDE.md). All arguments pass straight
through to generate_infinitetalk.py; this file only fixes up the process
before handing over.

Why a wrapper instead of editing the vendored repo: InfiniteTalk/ is
gitignored and re-cloned by setup_infinitetalk.sh, so anything changed inside
it silently disappears on a fresh setup. Keeping the patches here keeps them
under version control and leaves the vendored checkout pristine.

1. Low-commit quantized loading. Upstream loads its fp8/int8 checkpoints via
   optimum-quanto's requantize(), which first materialises a full-size
   placeholder for every parameter of the model (18.9B params x fp32 = ~75GB)
   before load_state_dict() overwrites them. Linux overcommit shrugs that
   off; Windows charges the whole allocation against the commit limit up
   front, so on a 32GB machine the process dies with exit code 3221225477
   (0xC0000005) in move_tensor before a single weight is read. The
   replacement below keeps the model on the meta device and loads the
   checkpoint with load_state_dict(assign=True), so the only large allocation
   is the checkpoint itself (~19.5GB for the DiT, ~6.7GB for T5). Verified by
   building the meta model and calling both loaders back to back: the
   upstream one faults, this one finishes with zero tensors left on meta.

2. inspect.ArgSpec. wan/multitalk.py does `from inspect import ArgSpec` at
   import time but never uses it, and Python 3.11 removed that name, so the
   import fails outright on the interpreter setup_infinitetalk.sh selects.
   Reinstating the name as the namedtuple it always was is enough.

3. wav2vec2 attention backend. InfiniteTalk's audio encoder subclass
   (src/audio_analysis/wav2vec2.py) sets `config.output_attentions = True`
   inside forward(). transformers 5 loads models with SDPA attention by
   default and rejects that assignment outright ("not supported when using
   the attn_implementation set to sdpa"), which kills the run right after
   the video models have loaded. Loading the encoder with the eager
   attention implementation - what transformers 4.x, which upstream pins,
   used - is the documented remedy and changes nothing numerically.

4. wav2vec2 per-layer hidden states. The same subclass calls the HF encoder
   directly with output_hidden_states=True and reads every layer's output
   back from `.hidden_states` - that is the audio feature InfiniteTalk was
   trained on (12 layers x 768). transformers 5's encoder no longer collects
   them for a direct call (that moved to a decorator on the top-level
   forward, which the subclass replaces), so it returns None and the run
   dies in get_embedding. Forward hooks on the 12 encoder layers capture
   exactly the tensors the old code got: this checkpoint has
   do_stable_layer_norm=false, so hidden_states[1:] was simply each layer's
   output, with no final norm applied afterwards.

5. fp32 attention on Blackwell GPUs. The audio cross-attention
   (wan/modules/attention.py) calls xformers' memory_efficient_attention on
   float32 tensors. xformers' only fp32 kernel is the cutlass one, which it
   refuses to run on compute capability > 9.0 ("too new"), and its
   flash-attention paths only take fp16/bf16 - so on an RTX 50-series card
   every diffusion step dies with NotImplementedError. PyTorch's own
   scaled_dot_product_attention handles fp32 on these GPUs and computes the
   same thing (checked against an explicit softmax: max abs diff 4e-6), so
   fp32 calls are routed there; fp16/bf16 calls still go to xformers.

6. T5 and CLIP are loaded on demand and dropped after use. Upstream builds
   both at pipeline construction and keeps them resident on the CPU for the
   whole run (6.7GB quantized T5 + 2.2GB fp16 CLIP) although each is used
   for a few seconds per clip, while the 18GB fp8 DiT is streamed from CPU
   RAM on every diffusion step. On a 32GB machine that combination pages.
   The pipeline's own access pattern is `x.model.to(device)` -> use ->
   `x.model.cpu()` (only when offload_model is on), so a holder that
   materialises the real object on the first of those and discards it on
   the last drops in without touching the pipeline code. Reloading costs a
   couple of seconds per clip. CLIP is additionally built on the meta
   device and filled with load_state_dict(assign=True): upstream instantiates
   it in fp32 on the GPU (9GB) before casting to fp16, which is a transient
   no 12GB card can afford once latents are resident. Verified identical to
   the upstream construction (same state dict, visual() output diff 0.0).

7. GPU allocator cap. Recent NVIDIA Windows drivers default to "sysmem
   fallback": when VRAM runs out, CUDA allocations silently land in system
   RAM instead of failing, and anything placed there runs at PCIe speed.
   PyTorch's caching allocator never sees an OOM, so it never compacts: on
   this 12GB card one 480p forward pass allocates at most 8.8GB but the
   cache grew to 12.2GB reserved, the overflow spilled, and a diffusion
   step took 7 minutes instead of ~30s. Capping PyTorch's share of the GPU
   just below what is actually free makes the allocator free cached blocks
   and retry, as it would on Linux, and a genuine shortfall now surfaces
   as a real OOM error instead of a silent 15x slowdown. (PyTorch's
   expandable_segments allocator mode would help further but is not
   supported on Windows.) Users can also set the driver's "CUDA - Sysmem
   Fallback Policy" to "Prefer No Sysmem Fallback" in the NVIDIA Control
   Panel; the cap makes that unnecessary.

8. Reference attention map in smaller head chunks. Every block's
   self-attention also computes an explicit visual-to-reference attention
   map (wan/utils/multitalk_utils.get_attn_map_with_target) for the
   speaker mask, 20 of the 40 heads at a time: at 480p that is a
   (20, 33264, 1584) fp32 tensor, 3.9GB, plus its softmax and masked copy,
   which was the single allocation that overflowed a 12GB card once the
   cap above stopped it spilling. The map is a mean over heads, so
   processing 5 heads at a time gives the same numbers (equal-size chunks)
   with a quarter of the transient.

9. bf16 scales for the fp8 weights. The quantized checkpoints store their
   per-channel scales in fp32, and quanto's matmul casts the activations
   to the scale dtype and dequantizes the weights into it before calling
   torch.matmul - so every linear layer of the DiT (and T5) ran as an fp32
   GEMM on CUDA cores. Profiled: 75 of the 111 GPU-seconds of one forward
   pass were cutlass/magma sgemm kernels. Casting the scales to bf16, the
   dtype the rest of the model already computes in, puts those GEMMs on
   tensor cores. The rounding this adds (2^-8 relative, on the scale) is
   far below the fp8 weights' own 2^-4 quantization error.

10. VAE decode headroom and fallback. Decoding one 81-frame 480p clip
    needs ~8.2GB of GPU memory by itself (measured; fp16/bf16 autocast do
    not reduce it), on top of what the pipeline still holds and whatever
    fragmentation the diffusion loop left behind. The decode therefore
    runs with the allocator cap lifted to everything but a small reserve,
    and if it still hits OOM it is retried on the CPU - slow, but the clip
    is not lost after an hour of sampling.

11. LoRAs on the quantized model. Upstream applies --lora_dir only when
    --quant is off (`if lora_dir is not None and quant is None`) and
    silently ignores it otherwise - but the fp8 model is the only one that
    fits this machine, and the step-distillation LoRAs the InfiniteTalk
    README recommends (lightx2v: 4 steps at text CFG 1 / audio CFG 2) are
    the difference between 120 and 8 DiT passes per chunk. Merging the
    delta into fp8 weights would round most of it away (e4m3 steps are
    ~6% of the weight; a LoRA delta is typically smaller), so each LoRA
    linear instead gets a bf16 low-rank side branch, out + s*B(A(x)), via a
    forward hook - the same function the upstream merge computes, at rank
    32 cost. The factors live in pinned host memory and are copied up per
    call, keeping ~300MB off a GPU the VAE decode needs. Bias and norm deltas (diff_b / diff) are plain tensors and are
    added in place, as upstream does. Attached on the first generate call,
    after the VRAM-management wrappers have replaced the original modules.

12. File-backed DiT weights. Patch 1 brought the quantized load down to
    the checkpoint itself, but safetensors' load_file still copies it into
    ~19.5GB of private memory, and every byte of that is charged against
    Windows' commit limit (RAM + pagefile). With a browser, Steam and WSL
    open, this 32GB machine had 38.2 of 47.9GB committed and the load died
    with the same 0xC0000005 as in patch 1. The checkpoint is instead
    memory-mapped read-only and the tensors are views into the mapping:
    file-backed pages cost no commit, the OS keeps them cached while RAM
    allows and re-reads them from disk otherwise. Nothing writes to them -
    the fp8 data is only ever copied to the GPU, and every fp32 tensor in
    the file (scales, biases, norms) is replaced by a bf16 copy before use
    (patch 9 and upstream's to_param_dtype_fp32only).
"""
import gc
import inspect
import os
import sys
from collections import namedtuple

if not hasattr(inspect, "ArgSpec"):
    inspect.ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

# `python path/to/infinitetalk_run.py` puts THIS file's directory on sys.path,
# not the working directory - InfiniteTalk's own modules live in the latter.
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from optimum.quanto.quantize import _quantize_submodule  # noqa: E402
import optimum.quanto  # noqa: E402


def requantize_low_commit(model, state_dict, quantization_map, device=None):
    """Drop-in for optimum.quanto.requantize that never materialises
    placeholders for weights the state dict is about to replace."""
    if device is None:
        device = next(model.parameters()).device
        if device.type == "meta":
            device = torch.device("cpu")

    for name, module in model.named_modules():
        qconfig = quantization_map.get(name)
        if qconfig is None:
            continue
        weights = None if qconfig["weights"] == "none" else qconfig["weights"]
        activations = None if qconfig["activations"] == "none" else qconfig["activations"]
        _quantize_submodule(model, name, module, weights=weights, activations=activations)

    # assign=True hands the checkpoint tensors to the model directly instead
    # of copying them into (pre-allocated) parameters. optimum-quanto's
    # QModuleMixin honours the same flag for its flattened quantized weights.
    model.load_state_dict(state_dict, strict=False, assign=True)

    # Anything still on meta had no entry in the checkpoint. Upstream would
    # have left those as uninitialised torch.empty() memory too; do the same
    # so behaviour matches, but say so - a long list here means the
    # checkpoint and the model config disagree.
    leftovers = []
    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            if param.device.type == "meta":
                leftovers.append(name)
                setattr(module, name, torch.nn.Parameter(
                    torch.empty_like(param, device="cpu"), requires_grad=param.requires_grad))
        for name, buf in list(module.named_buffers(recurse=False)):
            if buf.device.type == "meta":
                leftovers.append(name)
                setattr(module, name, torch.empty_like(buf, device="cpu"))
    if leftovers:
        print(f"infinitetalk_run: {len(leftovers)} tensors missing from the quantized "
              f"checkpoint were left uninitialised (first few: {leftovers[:5]})",
              file=sys.stderr, flush=True)

    cast_quantized_scales(model, torch.bfloat16)
    model.to(device)


def cast_quantized_scales(model, dtype) -> None:
    """Give every quanto weight a scale in `dtype` so its matmuls run in that
    dtype (see patch 9); a no-op for weights that already match."""
    from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

    converted = 0
    for module in model.modules():
        weight = getattr(module, "weight", None)
        qweight = weight.data if isinstance(weight, torch.nn.Parameter) else weight
        if isinstance(qweight, WeightQBytesTensor) and qweight._scale.dtype != dtype:
            recast = WeightQBytesTensor(
                qweight.qtype, qweight.axis, qweight.size(), qweight.stride(),
                qweight._data, qweight._scale.to(dtype), qweight.activation_qtype,
                requires_grad=False)
            module.weight = torch.nn.Parameter(recast, requires_grad=False)
            converted += 1
    if converted:
        print(f"infinitetalk_run: cast {converted} quantized weight scales to {dtype}", flush=True)


def custom_init_eager(device, wav2vec):
    """generate_infinitetalk.custom_init with attn_implementation='eager'."""
    from transformers import Wav2Vec2FeatureExtractor
    from src.audio_analysis.wav2vec2 import Wav2Vec2Model

    audio_encoder = Wav2Vec2Model.from_pretrained(
        wav2vec, local_files_only=True, attn_implementation="eager").to(device)
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        wav2vec, local_files_only=True)
    return wav2vec_feature_extractor, audio_encoder


def get_embedding_hooked(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device="cpu"):
    """generate_infinitetalk.get_embedding, collecting the per-layer hidden
    states with forward hooks when the encoder no longer returns them."""
    import numpy as np
    from einops import rearrange

    video_length = len(speech_array) / sr * 25  # InfiniteTalk renders at 25 fps
    audio_feature = np.squeeze(wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values)
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device).unsqueeze(0)

    captured = []

    def keep(_module, _inputs, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    handles = [layer.register_forward_hook(keep) for layer in audio_encoder.encoder.layers]
    try:
        with torch.no_grad():
            embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()

    hidden_states = embeddings.hidden_states[1:] if embeddings.hidden_states is not None else captured
    if len(hidden_states) != len(audio_encoder.encoder.layers):
        raise RuntimeError(
            f"wav2vec2 produced {len(hidden_states)} hidden states for "
            f"{len(audio_encoder.encoder.layers)} encoder layers")
    audio_emb = torch.stack(hidden_states, dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")
    return audio_emb.cpu().detach()


def install_fp32_attention_fallback() -> None:
    import torch.nn.functional as F
    import xformers.ops

    xformers_attention = xformers.ops.memory_efficient_attention

    def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, scale=None, *, op=None, **kwargs):
        if query.dtype in (torch.float16, torch.bfloat16):
            return xformers_attention(query, key, value, attn_bias=attn_bias, p=p, scale=scale, op=op, **kwargs)
        # xformers layout is (batch, seq, heads, dim); SDPA wants heads before seq.
        if attn_bias is None or isinstance(attn_bias, torch.Tensor):
            mask = attn_bias
        else:  # e.g. BlockDiagonalMask from the sequence-parallel path
            mask = attn_bias.materialize((query.shape[1], key.shape[1]), dtype=query.dtype, device=query.device)
        out = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            attn_mask=mask, dropout_p=p, scale=scale)
        return out.transpose(1, 2)

    xformers.ops.memory_efficient_attention = memory_efficient_attention


class _ModelHandle:
    """What `holder.model` returns: .to() materialises, .cpu() evicts."""

    def __init__(self, holder):
        self._holder = holder

    def to(self, *args, **kwargs):
        self._holder._real().model.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        self._holder._real().model.cuda(*args, **kwargs)
        return self

    def cpu(self):
        self._holder._drop()
        return self

    def __getattr__(self, name):
        return getattr(self._holder._real().model, name)


class EvictingModel:
    """Lazy stand-in for a T5EncoderModel / CLIPModel instance."""

    def __init__(self, factory, label):
        self._factory = factory
        self._label = label
        self._obj = None

    def _real(self):
        if self._obj is None:
            print(f"infinitetalk_run: loading {self._label}", flush=True)
            self._obj = self._factory()
        return self._obj

    def _drop(self):
        if self._obj is not None:
            print(f"infinitetalk_run: releasing {self._label}", flush=True)
            self._obj = None
            gc.collect()
            torch.cuda.empty_cache()

    @property
    def model(self):
        return _ModelHandle(self)

    def __call__(self, *args, **kwargs):
        return self._real()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real(), name)


def build_clip_lean(dtype, device, checkpoint_path, tokenizer_path):
    """wan.modules.clip.CLIPModel, built via meta device + assign so the fp32
    checkpoint is the only transient. Left on the CPU; the pipeline moves it."""
    from wan.modules.clip import CLIPModel, clip_xlm_roberta_vit_h_14
    from wan.modules.tokenizers import HuggingfaceTokenizer

    clip = CLIPModel.__new__(CLIPModel)
    clip.dtype = dtype
    clip.device = device
    clip.checkpoint_path = checkpoint_path
    clip.tokenizer_path = tokenizer_path
    clip.model, clip.transforms = clip_xlm_roberta_vit_h_14(
        pretrained=False, return_transforms=True, return_tokenizer=False,
        dtype=dtype, device=torch.device("meta"))
    # The upstream checkpoint is fp32 (4.4GB). Converting it costs a RAM
    # spike right when the 18GB DiT is already resident, so the converted
    # weights are cached once next to it and memory-mapped afterwards:
    # safetensors' load_file maps the file, so later loads add no private
    # memory at all and the pages can be dropped by the OS at any time.
    from safetensors.torch import load_file, save_file

    cache_path = f"{checkpoint_path}.{str(dtype).split('.')[-1]}.safetensors"
    if not os.path.exists(cache_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu", mmap=True)
        state_dict = {name: (tensor.to(dtype) if tensor.is_floating_point() else tensor.clone()).contiguous()
                      for name, tensor in state_dict.items()}
        save_file(state_dict, cache_path + ".tmp")
        os.replace(cache_path + ".tmp", cache_path)
        print(f"infinitetalk_run: cached {dtype} CLIP weights at {cache_path}", flush=True)
        del state_dict
    clip.model.load_state_dict(load_file(cache_path), assign=True)
    clip.model = clip.model.eval().requires_grad_(False)
    clip.tokenizer = HuggingfaceTokenizer(
        name=tokenizer_path, seq_len=clip.model.max_text_len - 2, clean="whitespace")
    return clip


def install_evicting_encoders() -> None:
    import wan.multitalk
    from wan.modules.t5 import T5EncoderModel

    wan.multitalk.T5EncoderModel = lambda **kwargs: EvictingModel(lambda: T5EncoderModel(**kwargs), "T5")
    wan.multitalk.CLIPModel = lambda **kwargs: EvictingModel(lambda: build_clip_lean(**kwargs), "CLIP")


GPU_CAP_FRACTION = None  # set by install_gpu_memory_cap, restored after a VAE decode


def install_gpu_memory_cap(margin_gb: float = 0.75) -> None:
    """Keep PyTorch's CUDA allocations inside the VRAM that is actually free,
    so the caching allocator compacts instead of spilling to system RAM."""
    global GPU_CAP_FRACTION
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    cap = max(free - margin_gb * 2**30, total * 0.5)
    GPU_CAP_FRACTION = min(cap / total, 1.0)
    torch.cuda.set_per_process_memory_fraction(GPU_CAP_FRACTION)
    print(f"infinitetalk_run: capping PyTorch GPU allocations at {cap/2**30:.1f} GB "
          f"({free/2**30:.1f} GB free of {total/2**30:.1f} GB at start)", flush=True)


def install_chunked_ref_attn_map(heads_per_chunk: int = 5) -> None:
    import functools
    import wan.modules.multitalk_model as mm

    original = mm.get_attn_map_with_target

    @functools.wraps(original)
    def chunked(visual_q, ref_k, shape, ref_target_masks=None, split_num=2, enable_sp=False):
        heads = visual_q.shape[2]
        if heads % heads_per_chunk == 0:
            split_num = heads // heads_per_chunk
        return original(visual_q, ref_k, shape, ref_target_masks=ref_target_masks,
                        split_num=split_num, enable_sp=enable_sp)

    mm.get_attn_map_with_target = chunked


def install_vae_decode_fallback(reserve_gb: float = 0.4) -> None:
    from wan.modules.vae import WanVAE

    original_decode = WanVAE.decode

    def decode(self, zs):
        if not torch.cuda.is_available():
            return original_decode(self, zs)
        _, total = torch.cuda.mem_get_info()
        torch.cuda.empty_cache()
        print(f"infinitetalk_run: VAE decode starting with "
              f"{torch.cuda.memory_allocated() / 2**30:.2f} GB allocated", flush=True)
        torch.cuda.set_per_process_memory_fraction(min((total - reserve_gb * 2**30) / total, 1.0))
        try:
            return original_decode(self, zs)
        except torch.OutOfMemoryError:
            print("infinitetalk_run: VAE decode ran out of GPU memory; decoding on the CPU "
                  "instead (several minutes)", flush=True)
            torch.cuda.empty_cache()
            gpu = self.device
            self.model.to("cpu")
            self.mean, self.std = self.mean.cpu(), self.std.cpu()
            self.scale = [self.mean, 1.0 / self.std]
            try:
                return original_decode(self, [z.cpu() for z in zs])
            finally:
                self.model.to(gpu)
                self.mean, self.std = self.mean.to(gpu), self.std.to(gpu)
                self.scale = [self.mean, 1.0 / self.std]
        finally:
            if GPU_CAP_FRACTION is not None:
                torch.cuda.set_per_process_memory_fraction(GPU_CAP_FRACTION)

    WanVAE.decode = decode


def _resolve_lora_target(model, dotted):
    """model.<dotted>, stepping through the `.module` indirection that
    AutoWrappedModule adds around norms and convs."""
    current = model
    for part in dotted.split("."):
        if part.isdigit():
            current = current[int(part)]
        elif hasattr(current, part):
            current = getattr(current, part)
        elif hasattr(current, "module") and hasattr(current.module, part):
            current = getattr(current.module, part)
        else:
            return None
    return current


def attach_lora_side_branches(model, lora_paths, lora_scales, device) -> None:
    from safetensors import safe_open
    import torch.nn.functional as F

    prefix = "diffusion_model."
    hooked = adjusted = 0
    missing = []
    for path, scale in zip(lora_paths, lora_scales):
        with safe_open(path, framework="pt") as f:
            keys = set(f.keys())
            for key in sorted(keys):
                if not key.startswith(prefix):
                    continue
                name = key[len(prefix):]
                if name.endswith(".lora_down.weight"):
                    up_key = key.replace("lora_down.weight", "lora_up.weight")
                    if up_key not in keys:
                        continue
                    module = _resolve_lora_target(model, name[: -len(".lora_down.weight")])
                    if module is None:
                        missing.append(name)
                        continue
                    # Kept in pinned host memory and copied up per call: ~300MB
                    # resident on the GPU was enough to push the VAE decode of a
                    # second streaming chunk into OOM, and the copies cost
                    # ~0.1% of a forward pass.
                    down = f.get_tensor(key).to(torch.bfloat16).pin_memory()
                    up = (f.get_tensor(up_key) * scale).to(torch.bfloat16).pin_memory()

                    def branch(_module, inputs, output, down=down, up=up):
                        x = inputs[0]
                        d = down.to(x.device, non_blocking=True)
                        u = up.to(x.device, non_blocking=True)
                        return output + F.linear(F.linear(x.to(d.dtype), d), u).to(output.dtype)

                    module.register_forward_hook(branch)
                    hooked += 1
                elif name.endswith(".diff_b") or name.endswith(".diff"):
                    attr = "bias" if name.endswith(".diff_b") else "weight"
                    owner = _resolve_lora_target(model, name.rsplit(".", 1)[0])
                    param = getattr(owner, attr, None) if owner is not None else None
                    if param is None and owner is not None and hasattr(owner, "module"):
                        param = getattr(owner.module, attr, None)
                    if param is None:
                        missing.append(name)
                        continue
                    with torch.no_grad():
                        param.add_((f.get_tensor(key).float() * scale).to(param.device, param.dtype))
                    adjusted += 1
    print(f"infinitetalk_run: LoRA attached to quantized model: {hooked} low-rank branches, "
          f"{adjusted} bias/norm deltas", flush=True)
    if missing:
        print(f"infinitetalk_run: {len(missing)} LoRA entries had no matching module "
              f"(first few: {missing[:5]})", file=sys.stderr, flush=True)


def install_quantized_lora() -> None:
    import wan.multitalk

    pipeline = wan.multitalk.InfiniteTalkPipeline
    original_init = pipeline.__init__
    original_generate = pipeline.generate_infinitetalk

    def __init__(self, *args, lora_dir=None, lora_scales=None, quant=None, **kwargs):
        original_init(self, *args, lora_dir=lora_dir, lora_scales=lora_scales, quant=quant, **kwargs)
        self._pending_lora = (lora_dir, lora_scales) if (lora_dir and quant is not None) else None

    def generate_infinitetalk(self, *args, **kwargs):
        pending = getattr(self, "_pending_lora", None)
        if pending:
            self._pending_lora = None
            attach_lora_side_branches(self.model, pending[0], pending[1], self.device)
        return original_generate(self, *args, **kwargs)

    pipeline.__init__ = __init__
    pipeline.generate_infinitetalk = generate_infinitetalk


_SAFETENSORS_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool,
}


def load_file_mapped(filename, device="cpu"):
    """safetensors.torch.load_file, but returning read-only views into a
    memory mapping of the file instead of private copies (see patch 12)."""
    import json
    import mmap
    import struct
    import warnings

    if str(device) != "cpu":
        from safetensors.torch import load_file
        return load_file(filename, device=device)
    with open(filename, "rb") as f:
        mapping = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    header_len = struct.unpack("<Q", mapping[:8])[0]
    header = json.loads(mapping[8:8 + header_len])
    base = 8 + header_len
    tensors = {}
    with warnings.catch_warnings():
        # "The given buffer is not writable": intended, nothing writes to these.
        warnings.simplefilter("ignore", UserWarning)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = _SAFETENSORS_DTYPES[info["dtype"]]
            start, end = info["data_offsets"]
            numel = 1
            for dim in info["shape"]:
                numel *= dim
            if numel == 0:
                tensors[name] = torch.empty(info["shape"], dtype=dtype)
                continue
            tensors[name] = torch.frombuffer(
                mapping, dtype=dtype, count=numel, offset=base + start).view(info["shape"])
    return tensors


def main() -> None:
    install_gpu_memory_cap()
    install_quantized_lora()
    install_fp32_attention_fallback()
    install_evicting_encoders()
    install_chunked_ref_attn_map()
    install_vae_decode_fallback()
    optimum.quanto.requantize = requantize_low_commit
    # Both call sites bound the name at import time (`from optimum.quanto
    # import ... requantize`), so patch their module globals as well.
    import wan.multitalk
    import wan.modules.t5
    wan.multitalk.requantize = requantize_low_commit
    wan.multitalk.load_file = load_file_mapped
    wan.modules.t5.requantize = requantize_low_commit

    import generate_infinitetalk
    generate_infinitetalk.custom_init = custom_init_eager
    generate_infinitetalk.get_embedding = get_embedding_hooked
    generate_infinitetalk.generate(generate_infinitetalk._parse_args())


if __name__ == "__main__":
    main()
