# The export path

`polima compile --checkpoint <path> --build-dir <dir>` turns a LeRobot
checkpoint into the inputs the compiler needs:

    onnx/<graph>.onnx           six fixed-shape graphs
    calibration/<graph>.npz     real activations, one per graph
    act_fixture.npz             the reference the smoke test compares against
    direct_inputs/*.f32         raw tensors the board reads
    normalization_stats.npz     host-owned mean/std
    input_contract.json         the build tree's self-description

It replaces `ACT/scripts/export_act_modalix.py` (309 lines), split so that only
the policy-specific half needs torch.

## Where the split is

`polima/export/` is generic; `polima/policies/<name>/graphs.py` is the policy's
own torch code, named from `CompilePlan` as dotted strings:

    export_entry          write onnx/ and calibration/
    fixture_entry         write the reference tensors
    verify_entry          replay the chain under onnxruntime
    normalization_entry   pull mean/std out of the checkpoint

Strings rather than imports, because `polima.policies.act` has to stay
importable in the compiler venv and on the board, where torch does not exist.
Only the export stage resolves them, and it runs directly in PoLiMa's
self-contained CPU-only `.venv`; CUDA is not needed for checkpoint loading,
ONNX export, or ONNX verification.

## Verify before compile, not after

The ONNX chain is replayed under onnxruntime and compared against PyTorch
*before* anything is quantized. If the ONNX matches and the deployed bundle does
not, the fault is in quantization or on the board; if the ONNX does not match,
compiling it would only bake the error in. Without that split, a single wrong
number at the end of a six-graph pipeline is guesswork.

`polima compile` refuses to continue on a failing verify, and says explicitly
that nothing has been quantized yet.

## Two reference files that are not interchangeable

`direct_inputs/expected_normalized_actions.f32` comes from **PyTorch**.
The per-graph `<graph>_output.f32` files come from **onnxruntime**. They differ
by ~1.4e-06, which is small enough to look like noise and large enough to fail a
strict comparison. Confusing them makes a working bundle look broken. Both are
kept, and `polima run --fixture` uses the PyTorch one.

## Reproduction proof

Re-exporting the checkpoint that produced the deployed bundle regenerates
everything byte for byte:

| artifact | result |
|---|---|
| 6 × `onnx/*.onnx` | identical |
| 6 × `calibration/*.npz` | identical |
| `act_fixture.npz`, `normalization_stats.npz` | identical |
| 16 × `direct_inputs/*.f32` | identical |
| verify report | `max_abs` 1.430511474609375e-06, `mean_abs` 2.2067067106945615e-07 — matches the recorded legacy values exactly |

Chained with the compile proof in [compile.md](compile.md), a checkpoint goes to
the same content-addressed bundle id as the one running on the board,
`act-rcwb_f_t-100000-a3573dae`, with every fixture byte-identical too. Export
takes ~32 s, a cold compile ~9.5 min.

## Per-graph references

`verify_chain` also dumps each graph's input and output to `direct_inputs/`.
These are what turn "the actions are wrong" into "graph 3 is wrong" — the board
can replay one ELF against its recorded input and compare, instead of bisecting a
six-graph chain by hand.

The legacy build trees carried these files, but produced them from a separate
devkit validation run rather than from the export, so a freshly built tree
silently lacked them. Every value is already computed during verification, so
writing them costs nothing.

### One trap this caught

The first port differed on exactly one graph. `EncoderStemPacked`'s state
projection had been inlined into the `torch.cat` call that consumes it:

```python
hidden = torch.cat([latent.unsqueeze(1), self.state(state).unsqueeze(1), ...])
```

`torch.onnx` traces in evaluation order, so that emits the state `Gemm` *after*
the latent's `Unsqueeze` rather than before it. Same 67 nodes, same 21
initializers, identical values, equivalent graph — different bytes, and
therefore a different ELF. Keeping the projection as its own statement restores
byte identity.

This is why the proof is byte identity rather than a numerical tolerance: a
tolerance check would have passed and quietly given up reproducibility.

## Determinism

The export seeds `random`, `numpy` and `torch` at 123. Calibration draws real
observations, so an unseeded export would produce different calibration data
every run — which would make the ELFs unreproducible and the compile stage's
content-keyed resume meaningless.

Samples are spread evenly across the dataset with `np.linspace` rather than
taken from the front. Episodes are recorded in order, so the first N frames are
all the same moment of the same episode and would cover a fraction of the
activation range.
