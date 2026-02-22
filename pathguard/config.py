from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def discover_video_paths() -> List[str]:
    local_roots = [Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]
    roots = []
    for base in local_roots:
        roots.append(base / "data_ironsite_hackathon")
        roots.append(base / "video")
        if base.exists() and base.is_dir():
            for child in base.iterdir():
                if child.is_dir() and child.name.lower().replace(" ", "").startswith("data_ironsite_hackatho"):
                    roots.append(child)
    roots.append(Path("/mnt/data"))
    discovered: List[str] = []
    seen = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for path in sorted(root.glob("*.mp4")):
            value = str(path)
            key = value.lower()
            if key not in seen:
                seen.add(key)
                discovered.append(value)
    return discovered


DISCOVERED_VIDEO_PATHS = discover_video_paths()
DEFAULT_VIDEO_PATH = DISCOVERED_VIDEO_PATHS[0] if DISCOVERED_VIDEO_PATHS else "/mnt/data/12_downtime_prep_mp.mp4"
ALT_VIDEO_PATH = DISCOVERED_VIDEO_PATHS[1] if len(DISCOVERED_VIDEO_PATHS) > 1 else "/mnt/data/14_production_mp.mp4"

PROMPTS = [
    "hose",
    "cable",
    "extension cord",
    "power cord",
    "wire",
    "rebar",
    "rope",
    "chain",
    "strap",
    "netting",
    "tool",
    "hand tool",
    "power tool",
    "drill",
    "grinder",
    "saw",
    "hammer",
    "wrench",
    "shovel",
    "spade",
    "bucket",
    "wheelbarrow",
    "box",
    "crate",
    "pallet",
    "wood plank",
    "timber",
    "plywood",
    "board",
    "sheet metal",
    "debris",
    "rubble",
    "trash bag",
    "plastic sheet",
    "tarp",
    "cone",
    "traffic cone",
    "barrier",
    "caution tape",
    "fence panel",
    "ladder",
    "step ladder",
    "scaffold",
    "scaffolding",
    "stool",
    "platform",
    "pipe",
    "pvc pipe",
    "metal pipe",
    "duct",
    "beam",
    "column",
    "mixer",
    "generator",
    "compressor",
    "cart",
    "carton",
    "bag of cement",
    "brick",
    "material",
]


@dataclass
class CorridorParams:
    bottom_width_frac: float = 0.90
    top_width_frac: float = 0.34
    height_frac: float = 0.64
    center_x_frac: float = 0.50


@dataclass
class RuntimeParams:
    dino_interval: int = 10
    depth_interval: int = 10
    occ_thresh: float = 0.015
    persistence_frames: int = 2
    blur_thresh: float = 20.0
    output_width: int = 960
    simulate_realtime: bool = True
    use_dino: bool = True
    use_sam2: bool = False
    use_depth: bool = False
    show_depth_overlay: bool = False
    depth_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    prompts: List[str] = field(default_factory=lambda: list(PROMPTS))
    dynamic_prompt_path: Optional[str] = None


def load_dynamic_prompts(filepath: Optional[str] = None) -> List[str]:
    """Load and merge dynamic prompts from a Cactus-generated DINO prompt file.

    If the file exists and contains valid prompts, merges them with the static
    PROMPTS list (dynamic prompts get priority). Falls back to static prompts
    if the file is missing or empty.

    Args:
        filepath: Path to a *_dino_prompt.txt file, or None for static-only.

    Returns:
        Merged prompt list (dynamic first, then static backfill).
    """
    if not filepath:
        return list(PROMPTS)

    try:
        from integration.dynamic_prompts import load_dino_prompt_file, merge_prompts
        dynamic = load_dino_prompt_file(filepath)
        if dynamic:
            return merge_prompts(PROMPTS, dynamic)
    except Exception:
        pass

    return list(PROMPTS)

