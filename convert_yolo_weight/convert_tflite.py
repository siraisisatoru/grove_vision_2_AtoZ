"""Export a YOLO .pt checkpoint to a Vela-compiled INT8 TFLite for the Himax WE2.

TWO fully isolated pipelines, selected by --yolo26:

  YOLO11 (default)              YOLO26 (--yolo26)
  ----------------              -----------------
  reg_max == 16 (DFL)           reg_max == 1 (DFL disabled, nn.Identity)
  kris-himax ultralytics 8.3.x  official ultralytics >= 8.4
  onnx2tf backend               LiteRT backend
  export int8=True,             export quantize='int8', end2end=False
         no_post=True           + _force_int8_boundary()
  -> best_saved_model/             -> best_int8.tflite (next to the .pt)
     best_full_integer_quant

BOTH split the detection head into 6 raw per-stride tensors, because both WE2
decoders parse the same channels-first contract (Yolo11::isValid() /
Yolo26::isValid()):

    box[i] : (1, 4*reg_max, grid, grid)   64 for YOLO11, 4 for YOLO26
    cls[i] : (1, nc,        grid, grid)
    interleaved: box0, cls0, box1, cls1, box2, cls2

Anything else loads on the device, binds to no decoder, and silently detects
nothing. verify_export() enforces this before Vela runs.

The two paths share only split_detection_head() and the Vela invocation. Do not
reintroduce cross-branch conditionals: the point of the split is that a change
to one architecture cannot silently alter the other.

Which Docker image supplies which ultralytics is flow.sh's job (--yolo26 routes
to grove_vision_convert_yolo26). This script asserts it got what it expected and
fails loudly otherwise, rather than mis-converting in silence.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import ultralytics
ultralytics.checks()
from ultralytics import YOLO


def _parse_names(yaml_path: Path) -> list[str] | None:
    """Extract class names from a data.yaml without resolving image paths."""
    if not yaml_path.exists():
        return None
    text = yaml_path.read_text()
    if "names:" not in text:
        return None
    after = text.split("names:", 1)[1]
    line0 = after.splitlines()[0]
    names: list[str] = []
    if "[" in line0:
        inner = line0[line0.find("[") + 1: line0.find("]")]
        names = [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
    else:
        for line in after.splitlines():
            s = line.strip()
            if not s:
                continue
            if s[0].isdigit() and ":" in s:
                names.append(s.split(":", 1)[1].strip())
            elif s.startswith("-"):
                names.append(s[1:].strip().strip("'\""))
            else:
                break
    return names or None


def build_calibration_yaml(args, output_dir: Path) -> str:
    """Build a data.yaml whose paths are valid in THIS environment (e.g. inside
    Docker, where a host-written data.yaml carries unreachable absolute paths).
    Returns the --data argument to hand to model.export()."""
    if not args.data_dir:
        return args.data

    data_dir = Path(args.data_dir).resolve()
    names = _parse_names(data_dir / "data.yaml") or ["object"]

    val_src = data_dir / "valid" / "images"
    cal_dir = val_src  # default: use the full val set

    if args.cal_images and val_src.exists():
        all_imgs = sorted(p for p in val_src.iterdir() if p.is_file())
        if len(all_imgs) > args.cal_images:
            # Evenly-spaced subset for representative class coverage
            step = max(1, len(all_imgs) // args.cal_images)
            selected = all_imgs[::step][: args.cal_images]
            cal_dir = output_dir / "_cal_subset"
            cal_dir.mkdir(exist_ok=True)
            for f in cal_dir.iterdir():
                f.unlink(missing_ok=True)
            for img in selected:
                shutil.copy2(str(img), str(cal_dir / img.name))
            print(f"Calibration subset: {len(selected)}/{len(all_imgs)} images "
                  f"(--cal-images {args.cal_images})")

    calib_yaml = output_dir / "_calib_data.yaml"
    calib_yaml.write_text(
        f"path: {data_dir}\n"
        f"train: {data_dir / 'train' / 'images'}\n"
        f"val:   {cal_dir}\n"
        f"test:  {data_dir / 'test'  / 'images'}\n"
        f"nc: {len(names)}\n"
        f"names: [{', '.join(names)}]\n"
    )
    print(f"Calibration data.yaml generated at {calib_yaml}  classes={names}")
    return str(calib_yaml)


# ---------------------------------------------------------------------------
# Post-export verification
# ---------------------------------------------------------------------------

def _load_interpreter(path: Path):
    """Return a TFLite Interpreter for `path`, or None if no runtime is available.

    Only ever called on the PRE-Vela file. A Vela-compiled model contains the
    ethos-u custom op and cannot be allocated by a stock interpreter — but Vela
    preserves the I/O tensor types, so checking beforehand is equivalent and
    actually works.
    """
    try:
        from ai_edge_litert.interpreter import Interpreter  # official 8.4+ image
        return Interpreter(model_path=str(path))
    except Exception:
        pass
    try:
        import tensorflow as tf  # onnx2tf image
        return tf.lite.Interpreter(model_path=str(path))
    except Exception:
        return None


def verify_export(tflite_path: Path, *, is_yolo26: bool, imgsz: int) -> None:
    """Check the exported TFLite matches what the WE2 firmware expects.

    Every problem checked here was first found the hard way on hardware, by
    reading the firmware's own tensor dump over the serial console:

        Detector input: type=10 ... shape.dims=[1,3,192,192] quant scale_x1000=0
        outputs_count=6

    type=10 is MA_TENSOR_TYPE_F32. A quantize='int8' LiteRT export still hands
    back FLOAT32 at the graph boundary — only the interior is quantized. The
    firmware then converts every frame on the CPU, costing ~40 ms/frame
    (measured perf [pre=12, invoke=84, post=6] ms against Vela's predicted 46 ms
    for the NPU portion) and forcing a float-expansion path in
    ma_model_detector.cpp that a true full-integer model would not need. The
    onnx2tf/YOLO11 path does not have this problem: its
    *_full_integer_quant.tflite is int8 end to end.
    """
    interp = _load_interpreter(tflite_path)
    if interp is None:
        print("WARNING: no TFLite runtime available in this image — skipping export "
              "verification. The Vela step will still run.")
        return

    try:
        interp.allocate_tensors()
    except Exception as e:
        print(f"WARNING: could not allocate tensors for verification ({e}); "
              f"reporting details only.")

    inputs = interp.get_input_details()
    outputs = interp.get_output_details()

    print("\n--- exported model boundary ---")
    for d in inputs:
        print(f"  input  {d['name']}: dtype={d['dtype'].__name__} "
              f"shape={list(d['shape'])} quant={d['quantization']}")
    for d in outputs:
        print(f"  output {d['name']}: dtype={d['dtype'].__name__} "
              f"shape={list(d['shape'])} quant={d['quantization']}")
    print("--- end boundary ---\n")

    problems: list[str] = []
    import numpy as np

    # Mirror the firmware's isValid() exactly — Yolo11::isValid() and
    # Yolo26::isValid() in sscma_micro/sscma/core/model/. A model that fails
    # these binds to NO decoder, loads cleanly, and silently never detects.
    #
    #   6 outputs, interleaved box0,cls0,box1,cls1,box2,cls2
    #   box[i] : (1, box_ch, grid, grid)   box_ch = 64 (YOLO11 DFL) or 4 (YOLO26)
    #   cls[i] : (1, nc,     grid, grid)
    #   grid   = w>>3, w>>4, w>>5
    box_ch = 4 if is_yolo26 else 64
    arch = "YOLO26" if is_yolo26 else "YOLO11"

    if len(outputs) != 6:
        problems.append(
            f"expected 6 output tensors for {arch}, got {len(outputs)}. The "
            f"detection-head split did not take effect; the firmware decoder "
            f"will reject this model and it will detect nothing.")
    else:
        for i in range(0, 6, 2):
            grid = imgsz >> ((i // 2) + 3)
            b = list(outputs[i]["shape"])
            c = list(outputs[i + 1]["shape"])
            if len(b) != 4 or b[1] != box_ch or b[2] != grid or b[3] != grid:
                problems.append(
                    f"output[{i}] (box, stride {1 << ((i // 2) + 3)}) is {b}, "
                    f"expected [1,{box_ch},{grid},{grid}]. {arch} needs "
                    f"channels-first box tensors with {box_ch} channels.")
            if len(c) != 4 or c[2] != grid or c[3] != grid:
                problems.append(
                    f"output[{i + 1}] (cls, stride {1 << ((i // 2) + 3)}) is {c}, "
                    f"expected [1,nc,{grid},{grid}].")

    # Full-integer boundary. A quantize='int8' LiteRT export leaves the boundary
    # FLOAT32 unless _force_int8_boundary() has neutralised Ultralytics'
    # NO_QUANTIZE override — see that function.
    bad_io = [d for d in (inputs + outputs) if d["dtype"] != np.int8]
    if bad_io:
        kinds = sorted({d["dtype"].__name__ for d in bad_io})
        problems.append(
            f"{len(bad_io)} of {len(inputs) + len(outputs)} boundary tensors are "
            f"{'/'.join(kinds)}, not int8. This is NOT a full-integer model: the "
            f"firmware must quantize/dequantize on the CPU every frame (~40 ms) "
            f"and the Ethos-U cannot own the whole graph.")

    # Input geometry. onnx2tf emits NHWC (1,H,W,3); LiteRT/torch emits NCHW
    # (1,3,H,W). The firmware accepts either for the INPUT (it swaps h/c), but
    # the size must match what gets deployed.
    ishape = list(inputs[0]["shape"]) if inputs else []
    if len(ishape) == 4:
        nchw = ishape[1] == 3
        h, w = (ishape[2], ishape[3]) if nchw else (ishape[1], ishape[2])
        print(f"Input layout: {'NCHW' if nchw else 'NHWC'}  {h}x{w}")
        if (h, w) != (imgsz, imgsz):
            problems.append(f"input is {h}x{w} but --imgsz={imgsz}")

    if problems:
        print("=" * 72)
        print("EXPORT VERIFICATION FAILED")
        for p in problems:
            print(f"  * {p}")
        print("=" * 72)
        if os.environ.get("ALLOW_BAD_EXPORT") == "1":
            print("ALLOW_BAD_EXPORT=1 set — continuing to Vela anyway.")
        else:
            raise SystemExit(
                "Refusing to Vela-compile a model that does not match the WE2 "
                "firmware's expectations. Set ALLOW_BAD_EXPORT=1 to override "
                "(the resulting on-device behaviour is known-degraded)."
            )
    else:
        print("Export verification passed.")


# ---------------------------------------------------------------------------
# Shared: detection-head split
# ---------------------------------------------------------------------------

def split_detection_head(model):
    """Make the detection head emit 6 separate raw per-stride tensors.

    Both WE2 decoders parse the SAME shape contract (see Yolo11::isValid() and
    Yolo26::isValid() in sscma_micro/sscma/core/model/):

        box[i] : (1, 4*reg_max, grid, grid)   64 for YOLO11 (DFL), 4 for YOLO26
        cls[i] : (1, nc,        grid, grid)   raw logits
        interleaved: box0, cls0, box1, cls1, box2, cls2

    Note both require CHANNELS-FIRST outputs (dims[1] is the channel count).
    That is why neither model can be routed through an NHWC converter.

    Ultralytics instead concatenates box/cls across the 3 scales (dim=-1) in
    Detect.forward_head(), then concatenates box+cls together (dim=1) in
    Detect._inference(), producing ONE (4+nc, anchors) tensor. That matches
    neither firmware parse path: the model loads fine and then never emits a
    box. This was the actual cause of "YOLO11 runs but detects nothing".

    Returns the number of output tensors produced (6).
    """
    head = model.model.model[-1]
    if not (hasattr(head, "cv2") and hasattr(head, "cv3") and hasattr(head, "nl")):
        raise SystemExit(
            "Detection head lacks cv2/cv3/nl — cannot split into the 6 raw tensors "
            "the WE2 firmware requires. Exporting anyway would produce a model that "
            "loads but never detects."
        )

    import types
    _cv2, _cv3, _nl = head.cv2, head.cv3, head.nl

    def _split_head(self, x):
        """Interleaved box/cls per stride; no torch.cat -> no CONCATENATION op."""
        out = []
        for i in range(_nl):
            out.append(_cv2[i](x[i]))  # (B, 4*reg_max, H, W)
            out.append(_cv3[i](x[i]))  # (B, nc,        H, W) — class logits
        return tuple(out)

    head.forward = types.MethodType(_split_head, head)
    n_out = _nl * 2
    print(f"Patched detection head -> {n_out} separate box/cls outputs "
          f"(interleaved: box0,cls0,box1,cls1,box2,cls2)")
    return n_out


# ---------------------------------------------------------------------------
# Shared LiteRT export mechanics
# ---------------------------------------------------------------------------

def _force_int8_boundary():
    """Stop Ultralytics from pinning the LiteRT graph boundary to FLOAT32.

    export_litert() applies static int8 quantization and then deliberately
    excludes the graph INPUT and OUTPUT ops from it:

        # Keep FP32 graph input/output ... matches the historical onnx2tf
        # "fp32 in/out" contract that downstream consumers expect
        for op in (qtyping.TFLOperationName.INPUT, qtyping.TFLOperationName.OUTPUT):
            qt.update_quantization_recipe(regex=".*", operation_name=op,
                                          algorithm_key=recipe.AlgorithmName.NO_QUANTIZE)

    That contract is wrong for the WE2. It leaves all 7 boundary tensors
    float32 (measured on-device: `Detector input: type=10`, quant scale 0), so
    the Ethos-U cannot own the whole graph and the CPU must quantize every
    frame — ~84 ms/inference against Vela's predicted 46 ms, and Vela reported
    540 KiB of SRAM instead of 320 KiB.

    Neutralise exactly those two calls and nothing else: the interior recipe,
    calibration and export all proceed untouched, but INPUT/OUTPUT stay in the
    int8 recipe and the exported model becomes genuinely full-integer.
    """
    try:
        from ai_edge_quantizer import qtyping, quantizer, recipe
    except ImportError:
        # export_litert() installs this on demand; pull it in early so we can patch.
        from ultralytics.utils.checks import check_requirements
        check_requirements("ai-edge-quantizer>=0.6.0")
        from ai_edge_quantizer import qtyping, quantizer, recipe

    if getattr(quantizer.Quantizer, "_we2_int8_boundary_patched", False):
        return

    _orig = quantizer.Quantizer.update_quantization_recipe
    _skip_ops = {qtyping.TFLOperationName.INPUT, qtyping.TFLOperationName.OUTPUT}
    _no_quant = recipe.AlgorithmName.NO_QUANTIZE

    def _patched(self, *a, **kw):
        if kw.get("operation_name") in _skip_ops and kw.get("algorithm_key") == _no_quant:
            print(f"  [we2] keeping {kw['operation_name']} in the int8 recipe "
                  f"(skipped Ultralytics' NO_QUANTIZE boundary override)")
            return
        return _orig(self, *a, **kw)

    quantizer.Quantizer.update_quantization_recipe = _patched
    quantizer.Quantizer._we2_int8_boundary_patched = True
    print("Patched ai_edge_quantizer to emit an int8 graph boundary (WE2 requires full-integer)")


def _export_litert(model, model_path: Path, args, data_arg: str,
                   output_dir: Path, *, arch: str) -> Path:
    """Head-split + full-integer LiteRT export. Used by BOTH architectures.

    Why not onnx2tf for YOLO11 (which is what the 8.3.x fork used)? Because
    onnx2tf breaks the firmware's contract in two independent ways, both caught
    by verify_export() against a real export:

      1. It emits NHWC. Box tensors came out (1,24,24,64); Yolo11::isValid()
         requires channels-first (1,64,24,24).
      2. It scrambles output ORDER. The 6 heads came back as
         box_s8, cls_s32, cls_s8, box_s16, cls_s16, box_s32 — not the
         interleaved box0,cls0,box1,cls1,box2,cls2 the firmware indexes
         positionally via getOutput(0..5).

    litert-torch traces the patched module directly, so it preserves both the
    NCHW layout and the tuple order we return from split_detection_head().
    Verified on hardware for YOTLO26; YOLO11 uses the identical mechanism and
    differs only in box channel count (64 vs 4), which follows from reg_max.
    """
    import ultralytics as _ul
    ul_ver = tuple(int(x) for x in _ul.__version__.split(".")[:2])
    if ul_ver < (8, 4):
        raise SystemExit(
            f"This pipeline requires official ultralytics >= 8.4 (found "
            f"{_ul.__version__}). The kris-himax 8.3.x fork exports via onnx2tf, "
            f"which produces NHWC outputs in scrambled order — rejected by both "
            f"WE2 decoders. Use the grove_vision_convert_yolo26 image.")

    split_detection_head(model)

    # The LiteRT export wraps the model in _NormalizeCoords, which assumes a
    # single concatenated (B, 4+nc, A) tensor and slices det[:, :4] / det[:, 4:]
    # to normalize box coords into [0,1] before quantization (mixed pixel-space
    # boxes and [0,1] scores under one int8 scale would collapse the scores).
    # Our 6-tensor output is already homogeneous per tensor — each is all-box or
    # all-cls — so that mixed-range problem does not arise. Pass it through
    # rather than teaching the wrapper to slice 6 tensors.
    from ultralytics.utils.export import litert as _litert_mod
    _orig_norm_forward = _litert_mod._NormalizeCoords.forward

    def _patched_norm_forward(self, x):
        y = self.model(x)
        if isinstance(y, tuple) and len(y) == 6:
            return y  # raw 6-tensor head — nothing to normalize
        return _orig_norm_forward(self, x)

    _litert_mod._NormalizeCoords.forward = _patched_norm_forward
    print("Patched LiteRT _NormalizeCoords to pass through the 6-tensor output unchanged")

    _force_int8_boundary()

    # quantize='int8' replaces 8.3's int8=True. end2end=False forces the classic
    # NMS-based head rather than a NMS-free end-to-end head; TFLite export
    # disables end2end implicitly anyway, but stating it keeps the exported op
    # set predictable instead of relying on that fallback.
    model.export(format=args.format, imgsz=args.imgsz, data=data_arg,
                 quantize="int8", end2end=False)
    print("export success")

    # LiteRT writes a single quantized .tflite next to the .pt — no
    # saved_model/ directory, unlike the onnx2tf path.
    exported = model_path.parent / f"{model_path.stem}_int8.tflite"
    if not exported.exists():
        candidates = sorted(model_path.parent.glob(f"{model_path.stem}*.tflite"))
        if not candidates:
            raise FileNotFoundError(f"Could not find exported .tflite next to {model_path}")
        exported = candidates[0]

    dest = output_dir / f"{arch}_int8.tflite"
    shutil.copy2(str(exported), str(dest))
    print(f"Copied export -> {dest}")
    return dest


# ---------------------------------------------------------------------------
# YOLO26 pipeline
# ---------------------------------------------------------------------------

def export_yolo26(model, model_path: Path, args, data_arg: str, output_dir: Path) -> Path:
    """reg_max==1: DFL disabled, so cv2[i] emits 4 channels of direct LTRB."""
    return _export_litert(model, model_path, args, data_arg, output_dir, arch="yolo26")


# ---------------------------------------------------------------------------
# YOLO11 pipeline
# ---------------------------------------------------------------------------

def export_yolo11(model, model_path: Path, args, data_arg: str, output_dir: Path) -> Path:
    """reg_max==16: cv2[i] emits 64 channels of raw DFL logits, which the
    firmware decodes on-device via compute_dfl() in ma_model_yolo11.cpp."""
    return _export_litert(model, model_path, args, data_arg, output_dir, arch="yolo11")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11n.pt",
                    help="Path to .pt weights file (default: yolo11n.pt)")
    ap.add_argument("--imgsz", type=int, default=192,
                    help="Model input image size (default: 192)")
    ap.add_argument("--data", default="coco128.yaml",
                    help="Calibration dataset yaml for int8 export (default: coco128.yaml)")
    ap.add_argument("--data-dir", default=None, metavar="DIR",
                    help="Dataset root directory. Generates a fresh data.yaml with paths "
                         "relative to DIR, overriding --data. Use this inside Docker where "
                         "a host-written data.yaml would have unreachable absolute paths.")
    ap.add_argument("--cal-images", type=int, default=100, metavar="N",
                    help="Max validation images for INT8 calibration when --data-dir is used "
                         "(default 100). Larger values improve quantization quality but "
                         "increase conversion time significantly.")
    ap.add_argument("--format", default="tflite",
                    help="Export format passed to model.export() (default: tflite). "
                         "Ultralytics 8.4.83+ redirects 'tflite' to 'litert' automatically.")
    ap.add_argument("--yolo26", action="store_true", default=False,
                    help="Select the YOLO26 pipeline: asserts reg_max==1, splits the "
                         "detection head into 6 raw box/cls tensors, and exports via "
                         "LiteRT. Without this flag the YOLO11 pipeline runs instead "
                         "(reg_max==16, DFL baked in, onnx2tf). The two paths are "
                         "fully isolated.")
    ap.add_argument("--skip-verify", action="store_true", default=False,
                    help="Skip post-export verification of the model boundary. Not "
                         "recommended: verification catches float32-boundary and "
                         "wrong-output-count exports that load on-device but "
                         "misbehave at runtime.")
    ap.add_argument("--output-dir", default=None, metavar="DIR",
                    help="Directory to write export/Vela artifacts to (default: "
                         "./output next to this script). flow.sh points this at the "
                         "use-case's dataset directory so converted weights live "
                         "alongside the training run that produced them.")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).resolve() if args.output_dir else script_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run from script_dir so ultralytics export artifacts land here, not the caller's CWD.
    os.chdir(script_dir)

    data_arg = build_calibration_yaml(args, output_dir)

    model_path = Path(args.model).resolve()
    model = YOLO(str(model_path))

    # Detect the architecture from the checkpoint itself — reg_max==1 means DFL
    # was disabled (nn.Identity()), which only YOLO26's head does. YOLO11/v8
    # always have reg_max==16. Independent of the .pt's filename.
    head = model.model.model[-1]
    reg_max = getattr(head, "reg_max", None)
    is_yolo26_arch = reg_max == 1
    yaml_file = (getattr(model.model, "yaml_file", None)
                 or getattr(model.model, "yaml", {}).get("yaml_file", "?"))
    print(f"Detected architecture: yaml={yaml_file} head={type(head).__name__} "
          f"reg_max={reg_max} -> {'YOLO26' if is_yolo26_arch else 'YOLO11/v8 (DFL)'}")

    # The flag and the weights must agree. Either mismatch silently produces a
    # model that loads on the device and then never detects anything.
    if args.yolo26 and not is_yolo26_arch:
        raise SystemExit(
            f"--yolo26 was passed but {model_path.name} is NOT a YOLO26 model "
            f"(reg_max={reg_max}, expected 1). Wrong weights file, or converting "
            f"through the wrong Docker image/pipeline.")
    if not args.yolo26 and is_yolo26_arch:
        raise SystemExit(
            f"{model_path.name} IS a YOLO26 model (reg_max=1) but --yolo26 was not "
            f"passed. The YOLO11 pipeline would bake in a DFL decode this "
            f"architecture does not have, producing a model that never detects. "
            f"Pass --yolo26 (flow.sh does this via its own --yolo26).")

    print(f"\n=== Pipeline: {'YOLO26' if args.yolo26 else 'YOLO11'} "
          f"(LiteRT, full-integer, 6 raw heads) ===\n")
    if args.yolo26:
        tflite_path = export_yolo26(model, model_path, args, data_arg, output_dir)
    else:
        tflite_path = export_yolo11(model, model_path, args, data_arg, output_dir)

    if not args.skip_verify:
        verify_export(tflite_path, is_yolo26=args.yolo26, imgsz=args.imgsz)

    print(f"Input to Vela : {tflite_path}")

    subprocess.run(
        [
            "vela",
            "--accelerator-config", "ethos-u55-64",
            "--config", str(script_dir / "himax_vela.ini"),
            "--system-config", "My_Sys_Cfg",
            "--memory-mode", "My_Mem_Mode_Parent",
            "--output-dir", str(output_dir),
            str(tflite_path),
        ],
        check=True,
    )

    vela_out = output_dir / (tflite_path.stem + "_vela.tflite")
    print(f"\nDone! Deploy this file to WE2: {vela_out}")


if __name__ == "__main__":
    main()
