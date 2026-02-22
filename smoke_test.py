"""Smoke test — verifies all modules import correctly under the new package structure."""

import sys


def main() -> None:
    errors = []

    # --- PathGuard core modules ---
    try:
        from pathguard import config  # noqa: F401
        from pathguard import audio_alerts  # noqa: F401
        from pathguard import corridor  # noqa: F401
        from pathguard import depth  # noqa: F401
        from pathguard import detect  # noqa: F401
        from pathguard import events  # noqa: F401
        from pathguard import fallback  # noqa: F401
        from pathguard import realtime  # noqa: F401
        from pathguard import segment  # noqa: F401
        print("  ✅ pathguard package — all modules imported")
    except Exception as exc:
        errors.append(f"pathguard: {exc}")
        print(f"  ❌ pathguard package — {exc}")

    # --- Integration modules ---
    try:
        from integration import dynamic_prompts  # noqa: F401
        from integration import enriched_telemetry  # noqa: F401
        print("  ✅ integration package — all modules imported")
    except Exception as exc:
        errors.append(f"integration: {exc}")
        print(f"  ❌ integration package — {exc}")

    # --- Test dynamic prompt parsing ---
    try:
        from integration.dynamic_prompts import parse_dino_prompt_text
        result = parse_dino_prompt_text("crane . worker . tractor . safety vest .")
        assert result == ["crane", "worker", "tractor", "safety vest"], f"Unexpected: {result}"
        print("  ✅ dynamic prompt parsing — verified")
    except Exception as exc:
        errors.append(f"dynamic_prompts test: {exc}")
        print(f"  ❌ dynamic prompt parsing — {exc}")

    # --- Test config load_dynamic_prompts ---
    try:
        from pathguard.config import load_dynamic_prompts
        # With None filepath, should return static prompts
        static = load_dynamic_prompts(None)
        assert len(static) > 0, "Static prompts should not be empty"
        print(f"  ✅ load_dynamic_prompts — {len(static)} static prompts loaded")
    except Exception as exc:
        errors.append(f"load_dynamic_prompts: {exc}")
        print(f"  ❌ load_dynamic_prompts — {exc}")

    # --- Narrator (optional — only works on macOS with Cactus engine) ---
    try:
        from narrator import cactus_vl  # noqa: F401
        print("  ✅ narrator package — imported (Cactus engine available)")
    except Exception as exc:
        print(f"  ⚠️  narrator package — skipped ({exc})")
        # Don't count as error — Cactus only works on Apple Silicon

    # --- Video read test (optional) ---
    try:
        import cv2
        from pathguard.config import DEFAULT_VIDEO_PATH
        cap = cv2.VideoCapture(DEFAULT_VIDEO_PATH)
        ok, _ = cap.read()
        cap.release()
        if ok:
            print(f"  ✅ video read — first frame from {DEFAULT_VIDEO_PATH}")
        else:
            print(f"  ⚠️  video read — could not read {DEFAULT_VIDEO_PATH} (file may not exist)")
    except Exception as exc:
        print(f"  ⚠️  video read — skipped ({exc})")

    print()
    if errors:
        print(f"SMOKE TEST FAILED — {len(errors)} error(s)")
        for err in errors:
            print(f"  • {err}")
        sys.exit(1)
    else:
        print("SMOKE TEST OK ✅")


if __name__ == "__main__":
    main()
