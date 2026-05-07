# Flash Attention vs Manual/SDPA Notes

## Why results can differ even with a fixed seed

They can differ because the current code paths are not numerically identical.

### 1. Different execution paths are used

- Normal inference path:
  - `model/modeling_llada.py`
  - `_scaled_dot_product_attention()`
  - Uses either:
    - `flash_attn_func(...)`, or
    - `F.scaled_dot_product_attention(...)`

- Attention-map saving path:
  - `model/modeling_llada.py`
  - `_manual_attention()`
  - Computes manually:
    - `scores = q @ k^T / sqrt(d)`
    - `attn_probs = softmax(scores)`
    - `attn_output = attn_probs @ v`

So when attention saving is enabled, the model is not using Flash Attention or PyTorch SDPA for that captured conditional forward. It is using a separate manual attention implementation.

### 2. The dtype path is different

Current manual attention implementation:

```python
scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
attn_probs = torch.softmax(scores, dim=-1)
attn_output = torch.matmul(attn_probs.to(dtype=v.dtype), v)
```

This means:

- `q` and `k` are promoted to `float32`
- softmax is also done in `float32`
- the final probabilities are cast back to `v.dtype` before multiplying with `v`

Flash Attention and SDPA may use different internal accumulation, casting, and fused-kernel behavior. That alone is enough to create different logits and therefore different sampled outputs.

### 3. Capture mode changes cache behavior

In `generators/image_generation_generator.py`, capture steps force the conditional path to run with full recomputation:

```python
cond_compute_mask = None if should_capture else ...
```

This was done to keep attention-map indexing aligned across the full sequence, but it also means the captured path does not exactly match the cached partial-compute path used otherwise.

### 4. Fixed seed does not guarantee identical outputs across different kernels

A fixed seed only fixes the random number stream.

It does not guarantee identical outputs when:

- the kernel is different
- the reduction order is different
- the precision/accumulation path is different
- the softmax is computed differently

Even very small logit differences can change sampling outcomes.

## What the current flags actually do

- `--no-save-attention-maps`
  - disables attention-map saving only
  - does **not** disable Flash Attention
  - model uses the original attention path

- `--save-attention-maps`
  - enables attention capture
  - captured conditional forward uses `_manual_attention()`
  - so this effectively bypasses Flash for those captured conditional passes

## Important conclusion

The current comparison is not a clean:

- Flash Attention vs PyTorch SDPA

It is closer to:

- Flash/SDPA normal path vs manually implemented attention path

## Better comparison design

For a cleaner comparison:

1. Add a separate `--disable-flash-attention` flag.
2. Keep the model forward path using `F.scaled_dot_product_attention(...)` when Flash is disabled.
3. Compute attention maps separately for saving, without changing the main output path.

That would separate:

- output-generation behavior
- attention visualization behavior

and make backend comparisons much more defensible.
