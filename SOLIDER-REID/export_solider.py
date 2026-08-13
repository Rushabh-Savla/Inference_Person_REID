"""
export_solider.py -- SOLIDER-REID swin_base (MSMT17) -> ONNX for the harness.

Run it from INSIDE the SOLIDER-REID checkout, because make_model.py does
`from loss.metric_learning import ...` and `from .backbones...`, so the repo
root has to be the import root:

    cd SOLIDER-REID
    python export_solider.py \
        --ckpt ~/swin_base_msmt17.pth \
        --config configs/msmt17/swin_base.yml \
        --out ~/solider_swin_base_msmt17.onnx

Then, from the pipeline repo:

    python tests/calibration/compare_backbones_on_run.py 20260806_095141 \
        --onnx ~/solider_swin_base_msmt17.onnx --onnx-size 384x128

No --onnx-mean / --onnx-std needed: the harness already defaults to ImageNet
(0.485,0.456,0.406 / 0.229,0.224,0.225), which is what this model wants. The
0.226 uniform std was a ReIdentificationNet-specific thing.

WHAT THIS TAPS. build_transformer.forward in eval mode returns
(feat, featmaps) where `feat` is post-bottleneck (BN) when TEST.NECK_FEAT ==
'after'. That is the direct analogue of the post-bnneck tap the shipping
FastReID uses, so the two are being compared at the same point in their
networks. The classifier is discarded -- its 1041 outputs are MSMT17 training
identities and mean nothing on these cameras.

WHY L2-NORMALIZE IS BAKED IN. Cosine similarity needs unit vectors, and the
model does not normalize. Normalization is IDEMPOTENT, so doing it here is safe
whether or not the harness also does it -- normalize(normalize(x)) ==
normalize(x). This removes the need to verify embed_onnx's behaviour.
"""
import argparse
import os
import sys

import torch
import torch.nn as nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="swin_base_msmt17.pth")
    ap.add_argument("--config", default="configs/msmt17/swin_base.yml")
    ap.add_argument("--out", required=True, help="output .onnx path")
    ap.add_argument("--num-class", type=int, default=1041,
                    help="MSMT17 training identities; must match "
                         "classifier.weight's first dim or the load fails")
    ap.add_argument("--semantic-weight", type=float, default=0.2,
                    help="runtest.sh passes MODEL.SEMANTIC_WEIGHT 0.2; it is "
                         "NOT in the yml and NOT in the checkpoint")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--trace-batch", type=int, default=1,
                    help="batch to trace at. SOLIDER's window_reverse does "
                         "B = int(windows.shape[0] / ...), which FREEZES the "
                         "batch into the graph, so the export is only valid at "
                         "this size. 1 means the harness must run --batch 1, "
                         "which is correct at every chunk size.")
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    from config import cfg                                   # noqa: E402
    from model import make_model                             # noqa: E402

    cfg.merge_from_file(args.config)
    # make_model line 192-193: `pretrained=model_path` then
    # `if model_path != '': self.base.init_weights(model_path)`. That is the
    # LUPerson SSL backbone, which we do not have and do not need -- the
    # fine-tuned checkpoint already contains those weights.
    cfg.merge_from_list([
        "MODEL.PRETRAIN_PATH", "",
        "MODEL.PRETRAIN_CHOICE", "self",
        "TEST.NECK_FEAT", "after",        # post-bnneck, matches FastReID's tap
    ])
    cfg.freeze()

    print(f"[export] transformer type : {cfg.MODEL.TRANSFORMER_TYPE}")
    print(f"[export] input size       : {cfg.INPUT.SIZE_TEST}")
    print(f"[export] semantic_weight  : {args.semantic_weight}")

    model = make_model(cfg, num_class=args.num_class,
                       camera_num=0, view_num=0,
                       semantic_weight=args.semantic_weight)

    sd = torch.load(os.path.expanduser(args.ckpt), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # STRICT on purpose. A silent partial load leaves randomly-initialised
    # layers, which embed happily and produce numbers that look like a model
    # performing badly rather than a model that never loaded.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[export] MISSING   ({len(missing)}): {missing[:8]}")
        print(f"[export] UNEXPECTED({len(unexpected)}): {unexpected[:8]}")
        raise SystemExit(
            "[export] checkpoint does not match the built model. Do NOT "
            "continue -- fix the config/num-class first. A forced load gives "
            "garbage embeddings that look like a bad model.")
    print("[export] checkpoint loaded, all keys matched")
    model.eval()

    # SOLIDER's Swin forward builds its semantic_weight tensor with a hardcoded
    # .cuda(), so the model has to live on the GPU or the two disagree. Exporting
    # from CUDA is fine -- ONNX graphs carry no device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit(
            "[export] no CUDA available. swin_transformer.py hardcodes .cuda() "
            "for semantic_weight, so a CPU export cannot work without patching "
            "that line.")
    model = model.to(device)

    class Embed(nn.Module):
        """Take the post-BN feature only, and L2-normalize it."""
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            out = self.m(x)
            feat = out[0] if isinstance(out, (tuple, list)) else out
            return torch.nn.functional.normalize(feat, p=2, dim=1)

    wrapper = Embed(model).eval().to(device)

    H, W = cfg.INPUT.SIZE_TEST
    dummy = torch.randn(args.trace_batch, 3, H, W, device=device)
    with torch.no_grad():
        ref = wrapper(dummy)
    print(f"[export] torch output: {tuple(ref.shape)}  "
          f"norm={ref.norm(dim=1).mean().item():.4f}")
    if ref.shape[1] != 1024:
        raise SystemExit(
            f"[export] expected 1024-d, got {ref.shape[1]}. 1041 means the "
            f"classifier is being tapped instead of the bottleneck.")

    out = os.path.expanduser(args.out)
    torch.onnx.export(
        wrapper, dummy, out,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
        # LEGACY TorchScript exporter. The dynamo path warns that dynamic_axes
        # is not supported there, and its opset 18->17 downconversion failed on
        # Pad, producing a graph 2.4e-01 away from the model.
        dynamo=False)
    print(f"[export] wrote {out}")

    # Verify the graph reproduces torch. A Swin export can silently differ --
    # `roll` and masked attention are the usual suspects -- and a wrong graph
    # would be measured as a bad model rather than a bad export.
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        # DIFFERENT batch than the trace, deliberately. SOLIDER builds its
        # semantic_weight tensor from x.shape[0]; if that got baked in as a
        # constant, the graph silently only works at batch 8 -- and the harness
        # runs at 32.
        # Check BOTH the traced batch and a different one. If 8 is fine and
        # 3 is not, the graph baked in a batch-dependent constant (SOLIDER
        # builds semantic_weight from x.shape[0]). If both are wrong, the graph
        # itself is bad and batching is a red herring.
        N = args.trace_batch
        worst = 0.0
        for n in (N,):
            probe = torch.randn(n, 3, H, W, device=device)
            with torch.no_grad():
                pref = wrapper(probe)
            got = sess.run(None, {"input": probe.cpu().numpy()})[0]
            if got.shape != tuple(pref.shape):
                raise SystemExit(
                    f"[export] batch BAKED IN: asked {n}, graph gave "
                    f"{got.shape}.")
            d = float(abs(got - pref.cpu().numpy()).max())
            worst = max(worst, d)
            print(f"[export] onnx vs torch max abs diff (batch {n}): {d:.2e}")
        if worst > 1e-3:
            raise SystemExit(
                "[export] GRAPHS DISAGREE at the traced batch -- do not use "
                "this file. The harness would measure a broken export and "
                "report it as a bad model. Try --opset 18, then --opset 14.")

        # Is the batch dimension actually free? Probe one size up. Not fatal --
        # a fixed-batch graph is perfectly usable as long as the harness feeds
        # that exact size -- but it must be SAID, because a silently wrong
        # batch produces plausible numbers rather than an error.
        free = True
        try:
            probe = torch.randn(N + 1, 3, H, W, device=device)
            with torch.no_grad():
                pref = wrapper(probe)
            got = sess.run(None, {"input": probe.cpu().numpy()})[0]
            free = (got.shape == tuple(pref.shape)
                    and float(abs(got - pref.cpu().numpy()).max()) < 1e-3)
        except Exception:
            free = False

        print(f"[export] verified at batch {N}: max abs diff {worst:.2e}")
        if free:
            print("[export] batch dimension is FREE -- any --batch works")
        else:
            print(f"[export] batch is FROZEN at {N} by swin_transformer.py:867 "
                  f"(B = int(windows.shape[0] / ...)).")
            print(f"[export] YOU MUST RUN THE HARNESS WITH  --batch {N}")
    except ImportError:
        print("[export] onnxruntime not installed -- skipped verification. "
              "Install it before trusting the result.")


if __name__ == "__main__":
    main()