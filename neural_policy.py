"""Neural policy/value components for Battlesnake.

The module is deliberately independent from the HTTP server.  Production code may
import it lazily; when PyTorch or a checkpoint is unavailable, the server keeps
using the deterministic tactical engine rather than risking an invalid move.

The encoder retains only information supplied by the Battlesnake request.  It
therefore works in both standard games and Blackout-style partial-observation
games without inventing unseen opponents or food.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DIRECTIONS = ("up", "down", "left", "right")
NUM_CHANNELS = 12
DEFAULT_BOARD_SIZE = 25
CHECKPOINT_FORMAT = 1

try:  # Keep the production heuristic usable on minimal deployments.
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on deployment extras
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def torch_required() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "Neural mode requires PyTorch. Install requirements-training.txt "
            "on the training machine; the production server will safely use "
            "the tactical fallback until a compatible checkpoint is present."
        )


def _xy(point: dict[str, Any] | None) -> tuple[int, int] | None:
    if not point or "x" not in point or "y" not in point:
        return None
    return int(point["x"]), int(point["y"])


def encode_state(
    data: dict[str, Any],
    board_size: int = DEFAULT_BOARD_SIZE,
) -> "Tensor":
    """Encode one Battlesnake request as a fixed-size spatial tensor.

    Planes distinguish our body, hostile bodies and hostile head size classes,
    food, hazards, valid board cells and compact scalar context.  Coordinates
    are placed directly at ``[y, x]``; boards larger than ``board_size`` are
    rejected instead of silently truncating information.
    """
    torch_required()
    board = data.get("board", {})
    width, height = int(board.get("width", 11)), int(board.get("height", 11))
    if width > board_size or height > board_size:
        raise ValueError(f"Board {width}x{height} exceeds encoder limit {board_size}x{board_size}")

    planes = torch.zeros((NUM_CHANNELS, board_size, board_size), dtype=torch.float32)
    planes[7, :height, :width] = 1.0  # valid cells

    you = data.get("you", {})
    our_length = max(1, int(you.get("length", len(you.get("body", [])) or 1)))
    our_head = _xy(you.get("head"))
    if our_head and 0 <= our_head[0] < width and 0 <= our_head[1] < height:
        planes[0, our_head[1], our_head[0]] = 1.0

    for segment in you.get("body", []):
        pos = _xy(segment)
        if pos and 0 <= pos[0] < width and 0 <= pos[1] < height:
            planes[1, pos[1], pos[0]] = 1.0

    our_id = you.get("id")
    for snake in board.get("snakes", []):
        if snake.get("id") == our_id:
            continue
        enemy_length = int(snake.get("length", len(snake.get("body", [])) or 1))
        enemy_head = _xy(snake.get("head"))
        if enemy_head and 0 <= enemy_head[0] < width and 0 <= enemy_head[1] < height:
            plane = 2 if enemy_length < our_length else 3
            planes[plane, enemy_head[1], enemy_head[0]] = 1.0
        for segment in snake.get("body", []):
            pos = _xy(segment)
            if pos and 0 <= pos[0] < width and 0 <= pos[1] < height:
                planes[4, pos[1], pos[0]] = 1.0

    for food in board.get("food", []):
        pos = _xy(food)
        if pos and 0 <= pos[0] < width and 0 <= pos[1] < height:
            planes[5, pos[1], pos[0]] = 1.0
    for hazard in board.get("hazards", []):
        pos = _xy(hazard)
        if pos and 0 <= pos[0] < width and 0 <= pos[1] < height:
            planes[6, pos[1], pos[0]] = 1.0

    planes[8, :height, :width] = min(1.0, max(0.0, float(you.get("health", 100)) / 100.0))
    planes[9, :height, :width] = min(1.0, our_length / float(max(1, width * height)))
    planes[10, :height, :width] = min(1.0, float(data.get("turn", 0)) / 200.0)

    # A short-range geometric cue speeds early learning while the convolutions
    # still receive the exact object locations in the preceding planes.
    if our_head:
        max_distance = max(1, width + height - 2)
        for y in range(height):
            for x in range(width):
                planes[11, y, x] = 1.0 - (abs(x - our_head[0]) + abs(y - our_head[1])) / max_distance
    return planes


if TORCH_AVAILABLE:
    class ResidualBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.norm1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.norm2 = nn.BatchNorm2d(channels)

        def forward(self, x: Tensor) -> Tensor:
            residual = x
            x = F.silu(self.norm1(self.conv1(x)))
            x = self.norm2(self.conv2(x))
            return F.silu(x + residual)


    class PolicyValueNet(nn.Module):
        """Small residual CNN with a 4-action policy head and scalar value head.

        It is intentionally compact so a CPU forward pass is practical within
        a live-move budget.  Training can use the same definition on CUDA.
        """
        def __init__(self, channels: int = 48, blocks: int = 3) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(NUM_CHANNELS, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(),
            )
            self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.policy_head = nn.Sequential(
                nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, len(DIRECTIONS))
            )
            self.value_head = nn.Sequential(
                nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, 1), nn.Tanh()
            )

        def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
            x = self.blocks(self.stem(x))
            x = self.pool(x).flatten(1)
            return self.policy_head(x), self.value_head(x).squeeze(-1)
else:
    class PolicyValueNet:  # pragma: no cover - raises only without optional dependency
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            torch_required()


@dataclass(frozen=True)
class CheckpointMeta:
    format: int
    channels: int
    blocks: int
    board_size: int


def save_checkpoint(
    path: str | Path,
    model: "PolicyValueNet",
    *,
    optimizer_state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    blocks: int = 3,
    board_size: int = DEFAULT_BOARD_SIZE,
) -> None:
    torch_required()
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "channels": NUM_CHANNELS,
        "blocks": blocks,
        "board_size": board_size,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer_state,
        "extra": extra or {},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> tuple["PolicyValueNet", CheckpointMeta, dict[str, Any]]:
    """Load only the documented checkpoint format and return an eval model."""
    torch_required()
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported neural checkpoint format")
    if payload.get("channels") != NUM_CHANNELS:
        raise ValueError("Checkpoint encoder channels do not match this server")
    blocks = int(payload.get("blocks", 3))
    model = PolicyValueNet(blocks=blocks).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    meta = CheckpointMeta(
        format=int(payload["format"]), channels=int(payload["channels"]),
        blocks=blocks, board_size=int(payload.get("board_size", DEFAULT_BOARD_SIZE)),
    )
    return model, meta, dict(payload.get("extra") or {})


def legal_action_mask(legal_directions: set[str], *, device: str = "cpu") -> "Tensor":
    torch_required()
    return torch.tensor([direction in legal_directions for direction in DIRECTIONS], dtype=torch.bool, device=device)


def masked_distribution(logits: "Tensor", legal_directions: set[str]) -> "Tensor":
    """Mask invalid actions before sampling or argmax, never afterwards."""
    torch_required()
    mask = legal_action_mask(legal_directions, device=str(logits.device))
    if not bool(mask.any()):
        raise ValueError("No legal actions supplied to policy")
    return logits.masked_fill(~mask, -1e9)


def predict_logits(
    model: "PolicyValueNet", data: dict[str, Any], *, device: str = "cpu", board_size: int = DEFAULT_BOARD_SIZE,
) -> tuple[dict[str, float], float]:
    torch_required()
    with torch.inference_mode():
        observation = encode_state(data, board_size=board_size).unsqueeze(0).to(device)
        logits, value = model(observation)
    return ({direction: float(logits[0, idx].cpu()) for idx, direction in enumerate(DIRECTIONS)}, float(value[0].cpu()))
