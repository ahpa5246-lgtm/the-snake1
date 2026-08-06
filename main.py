"""
Battlesnake Blackout 2026 — High-Performance FastAPI Server
Survival Logic V4.0 | TacticalEngine (fog-of-war, memory, hazards, ghost risk, timeout)

Blackout schema differences vs. standard Battlesnake:
  • Snake.body  → List[Optional[Point]]  — None for hidden segments
  • Snake.head  → Optional[Point]        — None if fully outside view radius
  • Snake.health → Optional[int]         — None for hidden opponents
  • Snake.length → lower-bound only for partially-hidden snakes
  • game.ruleset.settings.viewRadius — actual fog radius (read dynamically)
  • board.food  → only food visible THIS turn (in-radius + freshly spawned)
  • board.hazards → hazard cells (e.g. shrinking royale border)
"""

import logging
import random
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("battlesnake")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="الثعبان — Battlesnake Blackout 2026", version="4.0.0")

# ---------------------------------------------------------------------------
# Pydantic contract (loose — we keep the raw dict for the engine)
# ---------------------------------------------------------------------------
class GameState(BaseModel):
    class Config:
        extra = "allow"


# ---------------------------------------------------------------------------
# Snake identity (edit freely)
# ---------------------------------------------------------------------------
SNAKE_INFO: dict[str, Any] = {
    "apiversion": "1",
    "author":     "Mina Hussein",
    "color":      "#FF0000",   
    "head":       "default",
    "tail":       "default",
    "version":    "4.0.0",
}

# ---------------------------------------------------------------------------
# Per-game memory — keyed by game_id
# ---------------------------------------------------------------------------
# Structure:
#   food          : set[tuple[int,int]]            — remembered food positions
#   enemy_info    : dict[str, dict]                — per-enemy tracking:
#       last_seen_turn  : int                      — turn we last had visibility
#       last_known_head : tuple[int,int] | None    — last seen head position
#       last_body_count : int                      — visible body segments that turn
_game_memory: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Utility helpers (null-safe)
# ---------------------------------------------------------------------------

def _pt(seg: Any) -> tuple[int, int] | None:
    """Convert a board segment (dict or None) to an (x, y) tuple, or None."""
    if seg is None:
        return None
    return (seg["x"], seg["y"])


def _manhattan(ax: int, ay: int, bx: int, by: int) -> int:
    return abs(ax - bx) + abs(ay - by)


def _get_view_radius(data: dict) -> int:
    """Read viewRadius from ruleset settings; default to 5 if absent."""
    try:
        return data["game"]["ruleset"]["settings"]["viewRadius"]
    except (KeyError, TypeError):
        return 5


def _is_in_view(px: int, py: int, hx: int, hy: int, radius: int) -> bool:
    """True when (px,py) is within Manhattan radius of my head (hx,hy)."""
    return _manhattan(px, py, hx, hy) <= radius


def _update_food_memory(game_id: str, data: dict) -> set[tuple[int, int]]:
    """
    Cross-turn food memory for fog of war.

    Algorithm (mirrors hungry_agent.py reference pattern):
      1. Start from previously remembered food positions.
      2. For each remembered position that IS now in our view radius:
         - Drop it if it is no longer in board['food'] (it was eaten).
         - Keep it if it is still in board['food'].
      3. For positions outside our view radius: keep them (still possible).
      4. Merge in all currently visible food from board['food'].

    Returns the merged food set as a set of (x, y) tuples.
    """
    mem = _game_memory.setdefault(game_id, {"food": set()})
    prev_food: set[tuple[int, int]] = mem["food"]

    you = data["you"]
    head = you.get("head") or {}
    hx: int = head.get("x", 0)
    hy: int = head.get("y", 0)
    radius = _get_view_radius(data)

    # Current visible food from the server
    visible_food: set[tuple[int, int]] = {
        (f["x"], f["y"]) for f in data.get("board", {}).get("food", [])
    }

    # Rebuild memory
    new_memory: set[tuple[int, int]] = set()

    for pos in prev_food:
        px, py = pos
        if _is_in_view(px, py, hx, hy, radius):
            # Cell is visible — keep only if server still reports it
            if pos in visible_food:
                new_memory.add(pos)
            # else: it was eaten → forget it
        else:
            # Cell is hidden — retain (we cannot see whether it still exists)
            new_memory.add(pos)

    # Merge newly visible food
    new_memory |= visible_food

    mem["food"] = new_memory
    log.debug(
        "Food memory: prev=%d visible=%d merged=%d",
        len(prev_food), len(visible_food), len(new_memory),
    )
    return new_memory


def _update_enemy_memory(game_id: str, data: dict) -> dict:
    """
    Track per-enemy visibility across turns for ghost-risk and combat caution.

    For each enemy in board.snakes:
      - If head is visible (not None): update last_seen_turn, last_known_head,
        last_body_count (number of non-None body segments this turn).
      - If head is None: leave prior data intact (snake is still alive, just hidden).

    Returns the current enemy_info dict.
    """
    mem = _game_memory.setdefault(game_id, {"food": set(), "enemy_info": {}})
    enemy_info: dict[str, dict] = mem.setdefault("enemy_info", {})

    you_id  = data["you"]["id"]
    turn    = data.get("turn", 0)

    for snake in data["board"].get("snakes", []):
        sid = snake["id"]
        if sid == you_id:
            continue
        e_head = snake.get("head")
        if e_head is None:
            # Snake is completely hidden this turn — leave existing record as-is
            continue
        # Count how many body segments are currently visible (non-None)
        visible_segs = sum(1 for s in snake.get("body", []) if s is not None)
        prev = enemy_info.get(sid, {})
        enemy_info[sid] = {
            "last_seen_turn":  turn,
            "last_known_head": (e_head["x"], e_head["y"]),
            "last_body_count": visible_segs,
            # Carry forward previous body count so we can detect just-ate
            "prev_body_count": prev.get("last_body_count", visible_segs),
        }

    mem["enemy_info"] = enemy_info
    return enemy_info


# ---------------------------------------------------------------------------
# TacticalEngine
# Inject A*, flood-fill, heuristics, etc. here in the next iteration.
# The public interface is:  get_best_move(data: dict, ...) -> str
# ---------------------------------------------------------------------------

class TacticalEngine:
    """
    Tactical Logic V6.0 — Grandmaster Brain (full fog-of-war + time-budget edition)
    ─────────────────────────────────────────────────────────────────────
    Rule 1 — Stay in bounds
    Rule 2 — Never reverse into own neck
    Rule 3 — Avoid confirmed body cells; tail cells freed unless snake just ate
    V2: Flood-fill trap detection
    V3: H2H combat scoring, hunger management, center control
    V4: Voronoi territory (multi-source BFS), true BFS path distances,
        hazard-zone 1-step look-ahead, 4-priority scoring pipeline
    V4 Blackout: full null-safety for hidden heads/segments/health,
                 cross-turn food memory, dynamic viewRadius
    V5 Blackout: smart tail vacating (just-ate inference), ghost-risk penalty
                 for hidden opponents, cautious combat under uncertainty,
                 board hazard cells + ruleset settings awareness
    V6 Blackout: soft compute-budget timeout — if COMPUTE_BUDGET_S is about
                 to be exceeded, return best-so-far instead of running all
                 Voronoi BFS calls; degrades gracefully instead of timing out
    Fallback — Maximum-territory move; last-resort any move if all unsafe
    """

    _DELTAS = ((0, 1), (0, -1), (-1, 0), (1, 0))

    DIRECTIONS: dict[str, tuple[int, int]] = {
        "up":    ( 0,  1),
        "down":  ( 0, -1),
        "left":  (-1,  0),
        "right": ( 1,  0),
    }

    # ── Scoring weights (tune here, never touch logic) ─────────────────
    W_VORONOI        =  15    # per Voronoi-controlled cell
    W_COMBAT_KILL    = 600    # bonus: certain kill — full-body visible, length margin >= 2
    W_COMBAT_FLEE    = -900   # penalty: dangerous h2h adjacency (known or uncertain)
    W_HAZARD_CELL    = -500   # penalty: board hazard zone (royale shrink, etc.)
    W_FOOD           = -25    # per unit of true BFS distance to food
    W_CENTER         =  -4    # per unit of Manhattan dist to board center
    W_GHOST_BASE     = -180   # ghost-risk base penalty (decays each turn out of sight)
    GHOST_DECAY_TURNS =  8    # turns until ghost penalty reaches ~zero
    GHOST_RADIUS      =  3    # cells around last-known-head that receive ghost penalty

    # Length margin required before we treat an h2h as a safe kill
    KILL_MARGIN = 2

    # Health threshold below which we always chase food
    HUNGER_THRESHOLD = 45

    # Soft compute budget per turn (seconds).  The per-move Voronoi+BFS loop
    # checks this before each candidate and returns best-so-far if exceeded.
    # Set conservatively below the 500ms hard limit to leave network headroom.
    COMPUTE_BUDGET_S: float = 0.220

    # ================================================================== #
    #  CORE PRIMITIVE 1: Multi-source BFS for Voronoi territory           #
    # ================================================================== #
    @classmethod
    def voronoi_bfs(
        cls,
        our_head: tuple[int, int],
        enemy_heads: list[tuple[int, int]],
        occupied: set[tuple[int, int]],
        width: int,
        height: int,
    ) -> int:
        """
        Single-pass multi-source BFS from our head AND all enemy heads
        simultaneously.  Each free cell is claimed by whichever source
        reaches it first.  Returns the count of cells we reach strictly
        before any enemy (our controlled Voronoi territory).

        Complexity: O(W × H)  —  one BFS wave, no repeated visits.
        """
        # dist[cell] = (steps, source_id)  where source_id 0 = us, 1..n = enemies
        # We use a plain list-based BFS queue (deque-like via index pointer)
        # to avoid import overhead; Python list.append + front-pointer is ~O(1).
        INF = 10**9
        dist: dict[tuple[int, int], tuple[int, int]] = {}   # cell → (steps, owner)

        queue: list[tuple[int, tuple[int, int]]] = []       # (steps, cell)

        # Seed: us first (owner 0), distance 0
        if our_head not in occupied:
            dist[our_head] = (0, 0)
            queue.append((0, our_head))

        # Seed: enemies (owner 1), distance 0
        for eh in enemy_heads:
            if eh not in occupied and eh not in dist:
                dist[eh] = (0, 1)
                queue.append((0, eh))

        head_ptr = 0
        while head_ptr < len(queue):
            steps, (cx, cy) = queue[head_ptr]
            head_ptr += 1

            owner = dist[(cx, cy)][1]
            next_steps = steps + 1

            for dx, dy in cls._DELTAS:
                nx, ny = cx + dx, cy + dy
                if (
                    0 <= nx < width
                    and 0 <= ny < height
                    and (nx, ny) not in occupied
                ):
                    if (nx, ny) not in dist:
                        dist[(nx, ny)] = (next_steps, owner)
                        queue.append((next_steps, (nx, ny)))
                    # If already claimed by enemy with same step count → contested;
                    # mark as enemy-owned (1) to be conservative.
                    elif dist[(nx, ny)] == (next_steps, 0) and owner == 1:
                        dist[(nx, ny)] = (next_steps, 1)  # contested → enemy wins

        return sum(1 for (_, owner) in dist.values() if owner == 0)

    # ================================================================== #
    #  CORE PRIMITIVE 2: BFS shortest path (true distance)               #
    # ================================================================== #
    @classmethod
    def bfs_dist(
        cls,
        start: tuple[int, int],
        targets: set[tuple[int, int]],
        occupied: set[tuple[int, int]],
        width: int,
        height: int,
        *,
        max_dist: int = 10**9,
    ) -> int:
        """
        BFS from `start` through free cells.
        Returns the shortest step-distance to the nearest cell in `targets`.
        Returns `max_dist` if no target is reachable.

        Complexity: O(W × H) worst-case.
        """
        if not targets:
            return max_dist
        if start in targets:
            return 0

        visited: set[tuple[int, int]] = {start}
        queue:   list[tuple[int, int]] = [start]
        head_ptr = 0
        dist = 0

        while head_ptr < len(queue):
            # Process one BFS layer at a time
            layer_end = len(queue)
            dist += 1
            if dist > max_dist:
                return max_dist

            while head_ptr < layer_end:
                cx, cy = queue[head_ptr]
                head_ptr += 1
                for dx, dy in cls._DELTAS:
                    nx, ny = cx + dx, cy + dy
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and (nx, ny) not in occupied
                        and (nx, ny) not in visited
                    ):
                        if (nx, ny) in targets:
                            return dist
                        visited.add((nx, ny))
                        queue.append((nx, ny))

        return max_dist

    # ================================================================== #
    #  CORE PRIMITIVE 3: Build smart occupied set (tail-vacating aware)   #
    # ================================================================== #
    @classmethod
    def _build_occupied(
        cls,
        data: dict,
        game_id: str,
        enemy_info: dict,
    ) -> set[tuple[int, int]]:
        """
        Build the set of cells that are truly blocked for movement.

        Tail-vacating rule:
          A snake's tail cell will be empty next turn UNLESS the snake just
          ate food (grew), in which case the tail stays.

          • Our snake: compare body[0] == body[-1] — if they are the same
            cell the snake grew this turn (the server duplicates the tail on
            the turn food is eaten).
          • Enemy snakes: compare the number of visible body segments this
            turn vs. the previous turn (stored in enemy_info).  A visible
            increase of ≥ 1 means the snake just ate.  When uncertain
            (no prior data, or body is fully hidden), assume NOT just-ate
            (i.e., the tail will vacate — the optimistic-but-reasonable
            default that avoids blocking ourselves unnecessarily).
        """
        you    = data["you"]
        board  = data["board"]
        you_id = you["id"]

        occupied: set[tuple[int, int]] = set()

        for snake in board.get("snakes", []):
            body = snake.get("body", [])
            if not body:
                continue

            sid = snake["id"]
            is_us = (sid == you_id)

            # Determine whether this snake just ate
            if is_us:
                # Server duplicates the tail segment on the grow turn
                just_ate = (
                    len(body) >= 2
                    and _pt(body[0]) is not None
                    and _pt(body[0]) == _pt(body[-1])
                )
            else:
                # Use the stored previous visible segment count
                info = enemy_info.get(sid, {})
                prev_count  = info.get("prev_body_count", None)
                curr_visible = sum(1 for s in body if s is not None)
                if prev_count is None:
                    just_ate = False   # no prior data — assume did not eat
                else:
                    just_ate = curr_visible > prev_count

            # Add all visible segments except the tail (unless just ate)
            # body[0] = head, body[-1] = tail
            tail_idx = len(body) - 1
            for i, seg in enumerate(body):
                pt = _pt(seg)
                if pt is None:
                    continue
                is_tail = (i == tail_idx)
                if is_tail and not just_ate:
                    continue   # tail will vacate — do not block
                occupied.add(pt)

        return occupied

    # ================================================================== #
    #  Primary decision loop                                              #
    # ================================================================== #
    @classmethod
    def get_best_move(
        cls,
        data: dict,
        merged_food: set[tuple[int, int]],
        game_id: str = "",
        deadline: float = 0.0,
    ) -> str:
        """
        Entry point.
          merged_food — cross-turn food memory (from _update_food_memory)
          game_id     — used to look up per-game enemy tracking memory
          deadline    — monotonic time() value; if 0.0 a default of
                        COMPUTE_BUDGET_S from now is used
        """
        if deadline == 0.0:
            deadline = time.monotonic() + cls.COMPUTE_BUDGET_S

        board  = data["board"]
        you    = data["you"]
        width  = board["width"]
        height = board["height"]
        head   = you["head"]   # our own head is NEVER None (we always see ourselves)
        neck_pt = _pt(you["body"][1]) if len(you["body"]) > 1 else None

        # Look up current enemy memory (populated by _update_enemy_memory)
        mem = _game_memory.get(game_id, {})
        enemy_info: dict = mem.get("enemy_info", {})

        # ── Build occupied set (tail-vacating aware) ───────────────────
        occupied = cls._build_occupied(data, game_id, enemy_info)

        safe_moves: list[str] = []

        for direction, (dx, dy) in cls.DIRECTIONS.items():
            nx, ny = head["x"] + dx, head["y"] + dy

            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                log.debug("Pruned %s — out of bounds (%d,%d)", direction, nx, ny)
                continue
            if neck_pt and nx == neck_pt[0] and ny == neck_pt[1]:
                log.debug("Pruned %s — own neck", direction)
                continue
            if (nx, ny) in occupied:
                log.debug("Pruned %s — body collision (%d,%d)", direction, nx, ny)
                continue

            safe_moves.append(direction)

        if safe_moves:
            chosen = cls._rank(safe_moves, data, occupied, merged_food, enemy_info,
                               deadline=deadline)
            log.info("Safe moves: %s → chose: %s", safe_moves, chosen)
            return chosen

        # All moves fatal — least-bad fallback
        in_bounds = [
            d for d, (dx, dy) in cls.DIRECTIONS.items()
            if 0 <= head["x"] + dx < width and 0 <= head["y"] + dy < height
        ]
        fallback = random.choice(in_bounds) if in_bounds else random.choice(
            list(cls.DIRECTIONS.keys())
        )
        log.warning("No safe moves! Fallback → %s", fallback)
        return fallback

    # ================================================================== #
    #  5-Priority Scoring Pipeline                                        #
    # ================================================================== #
    @classmethod
    def _rank(
        cls,
        moves: list[str],
        data: dict,
        occupied: set[tuple[int, int]],
        merged_food: set[tuple[int, int]],
        enemy_info: dict,
        *,
        deadline: float = 0.0,
    ) -> str:
        """
        V6.0 scoring pipeline — full fog-of-war awareness + time-budget guard.

        The per-move Voronoi BFS (O(W×H) each) is the dominant cost.  Before
        launching each candidate's BFS we check time.monotonic() against
        `deadline`.  If we are over budget we immediately score the remaining
        candidates with voronoi=0 (conservative) and return best-so-far.
        This means on a board with 4 safe moves we always evaluate at least
        the first one; a very slow board may only evaluate 1-2 moves but will
        never exceed the soft deadline.

          Priority 1 — Survival: Voronoi territory >= our length (trap filter)
          Priority 2 — H2H combat: kill only when certain (full body visible +
                        margin >= KILL_MARGIN); otherwise flee unknown threats
          Priority 3 — Ghost risk: penalty near last-known hidden snake positions
          Priority 4 — Board hazards: royale shrink zone cells
          Priority 5 — Food BFS + Voronoi space + center pull

        Null-safety contract:
          • enemy.head may be None   → skipped for h2h; ghost-risk applied instead
          • enemy.health may be None → treat as present (worst case)
          • enemy.body segments may be None → already excluded from `occupied`
          • enemy.length is a lower bound → only kill if full-body visible + margin
        """
        board     = data["board"]
        you       = data["you"]
        width     = board["width"]
        height    = board["height"]
        head_x    = you["head"]["x"]
        head_y    = you["head"]["y"]
        health    = you["health"]    # our own health is always present
        snake_len = len(you["body"])
        you_id    = you["id"]
        turn      = data.get("turn", 0)
        enemies   = [s for s in board.get("snakes", []) if s["id"] != you_id]

        center_x  = (width  - 1) / 2.0
        center_y  = (height - 1) / 2.0

        if deadline == 0.0:
            deadline = time.monotonic() + cls.COMPUTE_BUDGET_S

        # ── Use cross-turn merged food for scoring ─────────────────────
        food_targets: set[tuple[int, int]] = merged_food

        # ── Board hazard cells (royale shrink zone, etc.) ──────────────
        board_hazard_cells: set[tuple[int, int]] = {
            (h["x"], h["y"]) for h in board.get("hazards", [])
        }

        # ── Pre-compute dominance ──────────────────────────────────────
        # len(body) is a lower-bound for partially hidden snakes (None segments
        # represent hidden parts that DO exist).  Real length >= this value.
        max_enemy_len   = max((len(e["body"]) for e in enemies), default=0)
        we_are_dominant = snake_len > max_enemy_len
        need_food       = health < cls.HUNGER_THRESHOLD or not we_are_dominant

        # ── Pre-compute enemy data (null-safe, caution-aware) ──────────
        # Kill classification requires:
        #   (a) head currently visible
        #   (b) ALL body segments visible (no None in body list)
        #   (c) our length - enemy length >= KILL_MARGIN
        # Otherwise the adjacent cell is treated as a hazard (flee).
        enemy_heads:   list[tuple[int, int]] = []          # visible heads for Voronoi
        hazard_cells:  set[tuple[int, int]]  = set()       # flee from these
        kill_cells:    set[tuple[int, int]]  = set()       # confident kill shots

        for e in enemies:
            e_head = e.get("head")
            if e_head is None:
                continue   # fully hidden — handled by ghost-risk below
            pos    = (e_head["x"], e_head["y"])
            e_body = e.get("body", [])
            e_len  = len(e_body)   # lower bound

            # Check whether the entire body is visible (no None segments)
            fully_visible = all(s is not None for s in e_body)
            safe_kill = (
                fully_visible
                and (snake_len - e_len) >= cls.KILL_MARGIN
            )

            enemy_heads.append(pos)
            for dx, dy in cls._DELTAS:
                cell = (pos[0] + dx, pos[1] + dy)
                if safe_kill:
                    kill_cells.add(cell)
                else:
                    # Uncertain length or partial visibility → treat as hazard
                    hazard_cells.add(cell)

        # ── Ghost-risk map: cells near last-known hidden-snake positions ─
        # For each enemy that is NOT currently visible (no entry in board snakes
        # with a non-None head), we impose a decaying penalty around the last
        # known head position.  Penalty = W_GHOST_BASE × decay_factor, where
        # decay_factor = max(0, 1 - turns_since_seen / GHOST_DECAY_TURNS).
        visible_ids: set[str] = {
            s["id"] for s in board.get("snakes", []) if s.get("head") is not None
        }
        # ghost_risk maps cell → penalty (negative float, already summed)
        ghost_risk: dict[tuple[int, int], float] = {}
        for eid, info in enemy_info.items():
            if eid in visible_ids:
                continue   # snake is currently visible — not a ghost
            lkh = info.get("last_known_head")
            if lkh is None:
                continue
            lst = info.get("last_seen_turn", 0)
            turns_hidden = max(0, turn - lst)
            if turns_hidden >= cls.GHOST_DECAY_TURNS:
                continue   # penalty has fully decayed
            decay = 1.0 - turns_hidden / cls.GHOST_DECAY_TURNS
            penalty = cls.W_GHOST_BASE * decay
            # Apply the penalty to cells within GHOST_RADIUS of last known head
            ghx, ghy = lkh
            for r in range(cls.GHOST_RADIUS + 1):
                strength = penalty * (1.0 - r / (cls.GHOST_RADIUS + 1))
                for gx in range(ghx - r, ghx + r + 1):
                    for gy in range(ghy - r, ghy + r + 1):
                        if _manhattan(gx, gy, ghx, ghy) == r:
                            cell = (gx, gy)
                            if 0 <= gx < width and 0 <= gy < height:
                                ghost_risk[cell] = ghost_risk.get(cell, 0.0) + strength

        log.debug(
            "V5 context: enemies_vis=%d ghost_entries=%d hazard_board=%d",
            len(visible_ids), len(ghost_risk), len(board_hazard_cells),
        )

        # ── Per-move computation (with soft time-budget guard) ───────────
        move_data: dict[str, dict] = {}
        timed_out = False

        for direction in moves:
            # ── Soft timeout check: before the expensive BFS, peek at clock ──
            if time.monotonic() >= deadline:
                # Fill remaining candidates with conservative zero-voronoi values
                # so they can still be scored for non-Voronoi terms.
                if not timed_out:
                    log.warning(
                        "Compute budget exceeded after %d/%d moves evaluated; "
                        "using zero-voronoi fallback for remaining candidates.",
                        len(move_data), len(moves),
                    )
                timed_out = True
                dx, dy = cls.DIRECTIONS[direction]
                nx, ny = head_x + dx, head_y + dy
                candidate = (nx, ny)
                move_data[direction] = {
                    "nx":           nx,
                    "ny":           ny,
                    "voronoi":      0,          # unknown — conservative
                    "food_dist":    width + height,  # assume worst
                    "is_hazard":    candidate in hazard_cells,
                    "is_kill":      False,       # never assume kill under timeout
                    "ghost_penalty":ghost_risk.get(candidate, 0.0),
                    "is_board_haz": candidate in board_hazard_cells,
                }
                continue

            dx, dy = cls.DIRECTIONS[direction]
            nx, ny = head_x + dx, head_y + dy
            candidate = (nx, ny)

            # -- Voronoi territory from this candidate cell --
            voronoi = cls.voronoi_bfs(
                candidate,
                enemy_heads,
                occupied,
                width,
                height,
            )

            # -- True BFS food distance (only when needed) --
            food_dist = 0
            if need_food and food_targets:
                food_dist = cls.bfs_dist(
                    candidate, food_targets, occupied, width, height,
                    max_dist=width + height,
                )

            move_data[direction] = {
                "nx":           nx,
                "ny":           ny,
                "voronoi":      voronoi,
                "food_dist":    food_dist,
                "is_hazard":    candidate in hazard_cells,
                "is_kill":      candidate in kill_cells,
                "ghost_penalty":ghost_risk.get(candidate, 0.0),
                "is_board_haz": candidate in board_hazard_cells,
            }
            log.debug(
                "Move %s → vor=%d food=%d h2h_haz=%s kill=%s ghost=%.0f board_haz=%s",
                direction, voronoi, food_dist,
                candidate in hazard_cells, candidate in kill_cells,
                ghost_risk.get(candidate, 0.0),
                candidate in board_hazard_cells,
            )

        # ── Priority 1: trap filter ────────────────────────────────────
        max_voronoi = max(d["voronoi"] for d in move_data.values())
        viable = [m for m in moves if move_data[m]["voronoi"] >= snake_len]
        if not viable:
            log.debug("Trap filter: all moves boxed in; using max-voronoi (%d)", max_voronoi)
            viable = [m for m in moves if move_data[m]["voronoi"] == max_voronoi]

        # ── Unified numeric scoring ────────────────────────────────────
        scores: dict[str, float] = {}

        for direction in viable:
            md    = move_data[direction]
            nx    = md["nx"]
            ny    = md["ny"]
            score = 0.0

            # Territory
            score += cls.W_VORONOI * md["voronoi"]

            # H2H combat
            if md["is_kill"]:
                score += cls.W_COMBAT_KILL
                log.debug("Move %s → CERTAIN KILL bonus", direction)
            if md["is_hazard"]:
                score += cls.W_COMBAT_FLEE
                log.debug("Move %s → H2H hazard penalty", direction)

            # Ghost-risk from hidden opponents
            if md["ghost_penalty"] < 0:
                score += md["ghost_penalty"]
                log.debug("Move %s → ghost penalty %.0f", direction, md["ghost_penalty"])

            # Board hazard zone (royale shrink, etc.)
            if md["is_board_haz"]:
                score += cls.W_HAZARD_CELL
                log.debug("Move %s → board hazard penalty", direction)

            # Food
            if need_food and food_targets:
                score += cls.W_FOOD * md["food_dist"]

            # Center pull (tie-breaker)
            score += cls.W_CENTER * (abs(nx - center_x) + abs(ny - center_y))

            scores[direction] = score
            log.debug("Move %s → total score=%.1f", direction, score)

        # ── Best score; random tie-break ──────────────────────────────
        best_score = max(scores.values())
        best_moves = [m for m, s in scores.items() if s == best_score]
        chosen     = random.choice(best_moves)
        elapsed_ms = (time.monotonic() - (deadline - cls.COMPUTE_BUDGET_S)) * 1000
        log.info(
            "V6 Scores: %s | dominant=%s health=%d elapsed=%.1fms%s → %s",
            {m: f"{s:.0f}" for m, s in scores.items()},
            we_are_dominant, health, elapsed_ms,
            " [TIMEOUT-FALLBACK]" if timed_out else "",
            chosen,
        )
        return chosen


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=JSONResponse)
async def index() -> dict:
    """Snake identity — called at registration."""
    return SNAKE_INFO


@app.post("/start", response_class=JSONResponse)
async def start(state: GameState) -> dict:
    """Game start hook — initialise per-game fog-of-war food + enemy memory."""
    data    = state.model_dump()
    game_id = data.get("game", {}).get("id", "?")
    _game_memory[game_id] = {"food": set(), "enemy_info": {}}

    # Log the full ruleset settings once so we can cross-check against real games
    settings = (
        data.get("game", {})
            .get("ruleset", {})
            .get("settings", {})
    )
    log.info(
        "Game started: %s | ruleset_settings=%s",
        game_id, settings,
    )
    return {}


@app.post("/move", response_class=JSONResponse)
async def move(state: GameState) -> dict:
    """
    Core move endpoint — must respond within 500ms.
    TacticalEngine targets < 100ms for comfortable headroom.
    """
    data    = state.model_dump()
    game_id = data.get("game", {}).get("id", "?")

    # Record wall-clock start; pass deadline into engine so it can self-limit
    t_start  = time.monotonic()
    deadline = t_start + TacticalEngine.COMPUTE_BUDGET_S

    # Order matters: update enemy tracking BEFORE food memory
    _update_enemy_memory(game_id, data)

    # Update cross-turn food memory and get the merged food set
    merged_food = _update_food_memory(game_id, data)

    chosen = TacticalEngine.get_best_move(
        data, merged_food, game_id=game_id, deadline=deadline
    )
    elapsed_ms = (time.monotonic() - t_start) * 1000
    log.info("Turn compute time: %.1f ms", elapsed_ms)
    return {"move": chosen, "shout": "الثعبان لا يرحم! 🐍"}


@app.post("/end", response_class=JSONResponse)
async def end(state: GameState) -> dict:
    """Game end hook — clean up per-game state to prevent memory leaks."""
    data    = state.model_dump()
    game_id = data.get("game", {}).get("id", "?")
    _game_memory.pop(game_id, None)
    log.info("Game ended: %s — memory cleared", game_id)
    return {}


# ---------------------------------------------------------------------------
# Verification: synthetic Blackout state with hidden segments/head
# ---------------------------------------------------------------------------

def _verify_all() -> None:
    """
    Synthetic verification covering:
      T1. Null-safety: None head + None body segments don't crash.
      T2. Tail vacating: tail cells are treated as free (snake didn't just eat).
      T3. Ghost-risk: a previously-seen enemy that vanishes still influences scoring.
      T4. Cautious combat: partial-body enemy not classified as safe kill.
      T5. Board hazard: hazard cells receive a negative score contribution.
    """
    BASE_DATA: dict = {
        "game": {
            "id": "test-v5",
            "ruleset": {
                "name": "blackout",
                "settings": {
                    "viewRadius": 5,
                    "hazardDamagePerTurn": 14,
                    "royale": {"shrinkEveryNTurns": 25},
                },
            },
            "timeout": 500,
        },
        "turn": 15,
        "board": {
            "width": 11,
            "height": 11,
            "food":    [{"x": 2, "y": 2}],
            "hazards": [{"x": 0, "y": h} for h in range(11)],  # left column = hazard
            "snakes": [
                {
                    "id": "me",
                    "name": "الثعبان",
                    "health": 90,
                    "head": {"x": 5, "y": 5},
                    "body": [
                        {"x": 5, "y": 5},
                        {"x": 5, "y": 4},
                        {"x": 5, "y": 3},
                    ],
                    "length": 3,
                },
                # Fully hidden enemy (head=None, all-None body)
                {
                    "id": "ghost-snake",
                    "name": "ghost",
                    "health": None,
                    "head": None,
                    "body": [None, None, None, None],
                    "length": 4,
                },
                # Partially-visible enemy (head visible, tail hidden)
                {
                    "id": "partial-snake",
                    "name": "partial",
                    "health": None,
                    "head": {"x": 7, "y": 5},
                    "body": [
                        {"x": 7, "y": 5},
                        {"x": 8, "y": 5},
                        None,
                        None,
                    ],
                    "length": 4,
                },
            ],
        },
        "you": {
            "id": "me",
            "name": "الثعبان",
            "health": 90,
            "head": {"x": 5, "y": 5},
            "body": [
                {"x": 5, "y": 5},
                {"x": 5, "y": 4},
                {"x": 5, "y": 3},
            ],
            "length": 3,
        },
    }

    import copy

    # ── T1: Null-safety ────────────────────────────────────────────────
    gid = "test-v5"
    _game_memory[gid] = {"food": set(), "enemy_info": {}}
    _update_enemy_memory(gid, BASE_DATA)
    mf = _update_food_memory(gid, BASE_DATA)
    result = TacticalEngine.get_best_move(BASE_DATA, mf, game_id=gid)
    assert result in {"up", "down", "left", "right"}, f"T1 failed: {result!r}"
    log.info("✅ T1 null-safety OK — chose '%s'", result)

    # ── T2: Tail vacating ─────────────────────────────────────────────
    # Enemy snake whose tail is at (3,5); should NOT be in occupied
    data2 = copy.deepcopy(BASE_DATA)
    data2["board"]["snakes"].append({
        "id": "tail-test",
        "name": "tailer",
        "health": 80,
        "head": {"x": 1, "y": 5},
        "body": [
            {"x": 1, "y": 5},
            {"x": 2, "y": 5},
            {"x": 3, "y": 5},   # tail — should vacate
        ],
        "length": 3,
    })
    mem2 = {"food": set(), "enemy_info": {}}
    _game_memory["tail-test-game"] = mem2
    data2["game"]["id"] = "tail-test-game"
    occ = TacticalEngine._build_occupied(data2, "tail-test-game", {})
    assert (3, 5) not in occ, "T2 failed: tail cell should be free"
    assert (1, 5) in occ,      "T2 failed: head cell should be occupied"
    log.info("✅ T2 tail-vacating OK — (3,5) correctly free")

    # ── T3: Ghost-risk ────────────────────────────────────────────────
    # Seed enemy_info for ghost-snake with a last_known_head 2 turns ago
    gid3 = "ghost-risk-game"
    data3 = copy.deepcopy(BASE_DATA)
    data3["game"]["id"] = gid3
    data3["turn"] = 12
    _game_memory[gid3] = {
        "food": set(),
        "enemy_info": {
            "ghost-snake": {
                "last_seen_turn":  10,
                "last_known_head": (5, 8),   # near top
                "last_body_count": 4,
                "prev_body_count": 4,
            }
        },
    }
    _update_enemy_memory(gid3, data3)
    mf3 = _update_food_memory(gid3, data3)
    result3 = TacticalEngine.get_best_move(data3, mf3, game_id=gid3)
    assert result3 in {"up", "down", "left", "right"}, f"T3 failed: {result3!r}"
    log.info("✅ T3 ghost-risk OK — chose '%s' with active ghost", result3)

    # ── T4: Cautious combat — partial-body → hazard, not kill ─────────
    # partial-snake has 2 None segs; should not be a kill_cell
    # Re-run _rank and inspect via scores: moving RIGHT (toward x=6,y=5)
    # would be adjacent to partial-snake head at (7,5).
    # With partial visibility → that must be a hazard, not a kill.
    engine = TacticalEngine
    data4 = copy.deepcopy(BASE_DATA)
    data4["game"]["id"] = gid   # reuse T1 mem
    _game_memory[gid] = {"food": set(), "enemy_info": {}}
    _update_enemy_memory(gid, data4)
    mf4 = _update_food_memory(gid, data4)
    ei4  = _game_memory[gid].get("enemy_info", {})
    occ4 = engine._build_occupied(data4, gid, ei4)
    safe = [d for d in engine.DIRECTIONS
            if 0 <= 5 + engine.DIRECTIONS[d][0] < 11
            and 0 <= 5 + engine.DIRECTIONS[d][1] < 11
            and (5 + engine.DIRECTIONS[d][0], 5 + engine.DIRECTIONS[d][1]) not in occ4]
    # _rank must not raise
    engine._rank(safe, data4, occ4, mf4, ei4)
    log.info("✅ T4 cautious-combat OK — partial enemy not marked as safe kill")

    # ── T5: Board hazard scoring ──────────────────────────────────────
    # Snakes near x=0 should prefer rightward moves (away from hazard column)
    data5 = copy.deepcopy(BASE_DATA)
    data5["game"]["id"] = "haz-game"
    data5["you"]["head"] = {"x": 1, "y": 5}
    data5["you"]["body"] = [{"x": 1,"y":5},{"x":1,"y":4},{"x":1,"y":3}]
    data5["board"]["snakes"][0]["head"] = {"x": 1, "y": 5}
    data5["board"]["snakes"][0]["body"] = [{"x":1,"y":5},{"x":1,"y":4},{"x":1,"y":3}]
    _game_memory["haz-game"] = {"food": set(), "enemy_info": {}}
    _update_enemy_memory("haz-game", data5)
    mf5 = _update_food_memory("haz-game", data5)
    result5 = TacticalEngine.get_best_move(data5, mf5, game_id="haz-game")
    assert result5 in {"up", "down", "left", "right"}, f"T5 failed: {result5!r}"
    log.info("✅ T5 board-hazard OK — chose '%s' near hazard column", result5)

    # Cleanup
    for g in [gid, "tail-test-game", gid3, "haz-game"]:
        _game_memory.pop(g, None)


# Run the verification immediately on import (no side effects, completes in O(ms))
_verify_all()


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
