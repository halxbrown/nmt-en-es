"""
Verify that the local install can run the trained translator.

    python setup_local.py

Checks each requirement in turn and, for anything missing, prints exactly what
to do. Ends with a real translation so a pass means the system actually works,
not merely that files are present.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

OK, BAD, WARN = "  [OK]  ", "  [--]  ", "  [!!]  "
problems: list[str] = []


def line(status: str, msg: str) -> None:
    print(status + msg)


def main() -> None:
    print("=" * 66)
    print("  Local setup check — EN→ES Transformer")
    print("=" * 66)

    # ---------------------------------------------------------- packages
    print("\nPackages")
    for mod, hint in (("torch", "pip install torch"),
                      ("sentencepiece", "pip install sentencepiece"),
                      ("numpy", "pip install numpy")):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "?")
            line(OK, f"{mod:<16} {v}")
        except ImportError:
            line(BAD, f"{mod:<16} missing")
            problems.append(f"Install {mod}:  {hint}")

    if any("torch" in p for p in problems):
        print("\nCannot continue without torch.")
        for p in problems:
            print("  * " + p)
        return

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    line(OK, f"{'device':<16} {dev}")
    if dev == "cpu":
        print("         (CPU is fine — expect ~1-3 s/sentence with beam search)")

    # ---------------------------------------------------------- project files
    print("\nProject files")
    for rel in ("config.py", "src/model.py", "src/decode.py", "src/tokenizer.py",
                "src/dataset.py", "translate.py"):
        p = ROOT / rel
        line(OK if p.exists() else BAD, rel)
        if not p.exists():
            problems.append(f"Missing {rel} — re-download it from the repo.")

    # ---------------------------------------------------------- tokenizer
    print("\nTokenizer")
    from config import ARTIFACT_DIR, CFG
    spm = ARTIFACT_DIR / f"{CFG.tok.model_prefix}.model"
    if spm.exists():
        line(OK, f"{spm.name}  ({spm.stat().st_size / 1024:.0f} KB)")
    else:
        line(BAD, f"{spm.name} not found")
        problems.append(
            f"Missing {spm}. It is tracked in the repo — try `git checkout "
            f"artifacts/{CFG.tok.model_prefix}.model`, or rebuild with "
            f"`python run_milestone1.py --force`.")

    # ---------------------------------------------------------- checkpoints
    print("\nModel checkpoints")
    from translate import find_checkpoint
    found = {}
    for preset in ("base", "small"):
        c = find_checkpoint(preset)
        if c:
            found[preset] = c
            line(OK, f"{preset:<7} {c.relative_to(ROOT) if ROOT in c.parents else c}"
                     f"  ({c.stat().st_size / (1024 * 1024):.0f} MB)")
        else:
            line(BAD, f"{preset:<7} not found")

    if not found:
        problems.append(
            "No checkpoint found. In Colab, after training:\n"
            "      !python export_model.py --preset base --checkpoint-dir \"{CKPT_DIR}\"\n"
            "    then download base_inference.pt from Drive into  models/")

    if problems:
        print("\n" + "=" * 66)
        print("  Not ready yet")
        print("=" * 66)
        for p in problems:
            print("  * " + p)
        return

    # ---------------------------------------------------------- live test
    print("\nLive translation test")
    preset = "base" if "base" in found else "small"
    try:
        from translate import DEFAULT_ALPHA, load, translate
        model, tok, mcfg, n = load(preset, found[preset], torch.device(dev))
        line(OK, f"loaded {preset}: {n:,} parameters, vocab {tok.vocab_size:,}")

        probes = ["Good morning.",
                  "Where is the train station?",
                  "I don't know what happened."]
        out = translate(model, tok, probes, torch.device(dev), "beam",
                        DEFAULT_ALPHA[preset], 5)
        print()
        for e, s in zip(probes, out):
            print(f"    EN  {e}")
            print(f"    ES  {s}\n")
    except SystemExit as e:
        line(BAD, str(e))
        return
    except Exception as e:                                     # noqa: BLE001
        line(BAD, f"translation failed: {type(e).__name__}: {e}")
        return

    print("=" * 66)
    print("  Ready.  Run:  python translate.py")
    print("=" * 66)


if __name__ == "__main__":
    main()
