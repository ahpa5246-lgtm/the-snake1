"""
run_games.py — local batch test runner for الثعبان vs. baseline agents.

Requirements
------------
This runner works in TWO modes:

Mode A: With hisss installed (pip install hisss)
  - Uses hisss.BattleSnakeGame as the simulator (fast C++).
  - Install: pip install hisss

Mode B: Standalone (no hisss)
  - Falls back to a minimal pure-Python Blackout simulator.
  - Zero dependencies beyond main.py's own requirements.

Usage
-----
  python run_games.py                       # 20 games, standalone
  python run_games.py --games 100           # 100 games
  python run_games.py --hazards             # enable hazard strip (left edge)
  python run_games.py --hazard-dmg 14       # hazard damage per turn (default 14)
  python run_games.py --no-hisss            # force standalone even if hisss available
  python run_games.py --verbose             # print every turn
  python run_games.py --seed 42             # reproducible RNG
  python run_games.py --latency-warn 250    # warn if move > Nms (default 250)

Fixes applied (v2):
  1. RandomAgent now excludes occupied body cells (not just neck/bounds).
  2. Added SafeFoodSeekingAgent as a smarter baseline replacing one Random.
  3. --hazards CLI flag adds a left-edge hazard strip with real per-turn damage.
  4. Per-move latency stats (max, p95, over-threshold count) in summary.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Local Battlesnake Blackout batch test")
parser.add_argument("--games",        type=int,   default=20,   help="Number of games to run")
parser.add_argument("--no-hisss",     action="store_true",      help="Force standalone simulator")
parser.add_argument("--verbose",      action="store_true",      help="Print every turn")
parser.add_argument("--seed",         type=int,   default=None, help="RNG seed for reproducibility")
parser.add_argument("--hazards",      action="store_true",      help="Enable left-edge hazard strip")
parser.add_argument("--hazard-dmg",   type=int,   default=14,   help="Hazard damage per turn (default 14)")
parser.add_argument("--latency-warn", type=int,   default=250,  help="Warn if move > Nms (default 250)")
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)

# ---------------------------------------------------------------------------
# Import our agent + latency collector
# ---------------------------------------------------------------------------
import agent_adapter
from agent_adapter import ThuebanAgent, move_latency_ms

# ---------------------------------------------------------------------------
# Try to import hisss; fall back to standalone simulator
# ---------------------------------------------------------------------------
USE_HISSS = False
if not args.no_hisss:
    try:
        import hisss
        USE_HISSS = True
        print("[runner] hisss found — using C++ simulator")
    except ImportError:
        print("[runner] hisss not found — using standalone Python simulator")
else:
    print("[runner] --no-hisss set — using standalone Python simulator")

# ---------------------------------------------------------------------------
# Simulator constants
# ---------------------------------------------------------------------------
WIDTH       = 11
HEIGHT      = 11
VIEW_RADIUS = 5
FOOD_MIN    = 5
FOOD_SPAWN_PROB = 0.15
HUNGER_PER_TURN = 1
START_HEALTH    = 100

# Hazard strip: left column (x == 0) when --hazards is set
HAZARD_CELLS: list[tuple[int, int]] = (
    [(0, y) for y in range(HEIGHT)] if args.hazards else []
)
HAZARD_DMG = args.hazard_dmg if args.hazards else 0


def _mk_ruleset() -> dict:
    return {
        "name": "blackout",
        "settings": {
            "viewRadius": VIEW_RADIUS,
            "hazardDamagePerTurn": HAZARD_DMG,
            "royale": {"shrinkEveryNTurns": 0},
        },
    }


# ===========================================================================
# Snake dataclass
# ===========================================================================
@dataclass
class Snake:
    id:     str
    name:   str
    body:   list[tuple[int, int]]
    health: int  = START_HEALTH
    alive:  bool = True

    @property
    def head(self) -> tuple[int, int]:
        return self.body[0]

    @property
    def length(self) -> int:
        return len(self.body)


# ===========================================================================
# State builder
# ===========================================================================
def _mk_state(
    game_id: str,
    turn:    int,
    snakes:  list[Snake],
    food:    list[tuple[int, int]],
    you_idx: int,
) -> dict:
    """
    Build a raw Blackout game-state dict for snake at index `you_idx`.
    Applies fog of war: segments beyond VIEW_RADIUS from my head become None.
    Hazard cells from the module-level HAZARD_CELLS list are always included.
    """
    me = snakes[you_idx]
    hx, hy = me.head

    def fogged_body(s: Snake) -> list:
        result = []
        for seg in s.body:
            sx, sy = seg
            if abs(sx - hx) + abs(sy - hy) <= VIEW_RADIUS:
                result.append({"x": sx, "y": sy})
            else:
                result.append(None)
        return result

    def fogged_head(s: Snake) -> Optional[dict]:
        sx, sy = s.head
        if abs(sx - hx) + abs(sy - hy) <= VIEW_RADIUS:
            return {"x": sx, "y": sy}
        return None

    snake_list = []
    for s in snakes:
        if not s.alive:
            continue
        fh = ({"x": s.head[0], "y": s.head[1]}
              if s.id == me.id else fogged_head(s))
        fb = fogged_body(s)
        snake_list.append({
            "id":     s.id,
            "name":   s.name,
            "health": s.health if (s.id == me.id or fh is not None) else None,
            "head":   fh,
            "body":   fb,
            "length": len(fb),
        })

    visible_food = [
        {"x": fx, "y": fy} for fx, fy in food
        if abs(fx - hx) + abs(fy - hy) <= VIEW_RADIUS
    ]

    # Always return all hazard cells (they are always visible in royale rules)
    hazard_list = [{"x": cx, "y": cy} for cx, cy in HAZARD_CELLS]

    you_dict = next(s for s in snake_list if s["id"] == me.id)
    you_dict["head"]   = {"x": me.head[0], "y": me.head[1]}
    you_dict["health"] = me.health

    return {
        "game":  {"id": game_id, "ruleset": _mk_ruleset(), "timeout": 500},
        "turn":  turn,
        "board": {
            "width":   WIDTH,
            "height":  HEIGHT,
            "food":    visible_food,
            "hazards": hazard_list,
            "snakes":  snake_list,
        },
        "you": you_dict,
    }


# ===========================================================================
# Helpers
# ===========================================================================
def _random_empty(occupied: set) -> Optional[tuple[int, int]]:
    for _ in range(400):
        x, y = random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1)
        if (x, y) not in occupied:
            return (x, y)
    return None


def _build_occupied_sim(snakes: list[Snake]) -> set[tuple[int, int]]:
    """
    Tail-vacating-aware occupied set for the simulator itself (mirrors logic
    in main.py's _build_occupied).  The tail is excluded unless the snake's
    body[0] == body[-1] (i.e. it just ate — server duplicate tail signal).
    """
    occ: set[tuple[int, int]] = set()
    for s in snakes:
        if not s.alive:
            continue
        just_ate = len(s.body) >= 2 and s.body[0] == s.body[-1]
        tail_idx = len(s.body) - 1
        for i, seg in enumerate(s.body):
            if i == tail_idx and not just_ate:
                continue   # tail will vacate
            occ.add(seg)
    return occ


# ===========================================================================
# Pydantic helper (cached class)
# ===========================================================================
from pydantic import BaseModel as _BM

class _GS(_BM):
    class Config:
        extra = "allow"

def _wrap(state_dict: dict):
    return _GS.model_validate(state_dict)


# ===========================================================================
# Standalone game runner
# ===========================================================================
def _run_standalone_game(
    agents:  list,
    game_id: str,
    verbose: bool,
) -> dict:
    """
    Run one complete Blackout game.
    Returns {winner, turns, deaths: {agent_idx: turn_died},
             hazard_deaths, hazard_damage_taken}.
    """
    start_positions = [(1, 1), (9, 9), (1, 9), (9, 1)][:len(agents)]
    snakes = [
        Snake(id=f"snake-{i}", name=agents[i].get_name(), body=[pos, pos, pos])
        for i, pos in enumerate(start_positions)
    ]

    food: list[tuple[int, int]] = []
    init_occ = {s.body[0] for s in snakes}
    for _ in range(FOOD_MIN):
        p = _random_empty(init_occ | set(food))
        if p:
            food.append(p)

    # start()
    for i, agent in enumerate(agents):
        try:
            agent.start(_wrap(_mk_state(game_id, 0, snakes, food, i)))
        except Exception:
            pass

    DIRS = {"up": (0,1), "down": (0,-1), "left": (-1,0), "right": (1,0)}

    deaths: dict[int, int] = {}
    hazard_deaths = 0
    hazard_dmg_total = 0
    turn = 0

    while True:
        turn += 1
        alive = [i for i, s in enumerate(snakes) if s.alive]
        if len(alive) <= 1 or turn > 500:
            break

        # Spawn food
        if len(food) < FOOD_MIN or random.random() < FOOD_SPAWN_PROB:
            all_occ = {seg for s in snakes if s.alive for seg in s.body} | set(food)
            p = _random_empty(all_occ)
            if p:
                food.append(p)

        # Collect moves
        moves: dict[int, str] = {}
        for i in alive:
            state_dict = _mk_state(game_id, turn, snakes, food, i)
            try:
                mv = agents[i].move(_wrap(state_dict))
                if hasattr(mv, "move"):
                    d = str(mv.move).lower()
                    for k in DIRS:
                        if k in d:
                            d = k
                            break
                    moves[i] = d
                else:
                    moves[i] = str(mv)
            except Exception:
                moves[i] = random.choice(list(DIRS.keys()))

        # Compute new heads
        new_heads: dict[int, tuple[int, int]] = {
            i: (snakes[i].head[0] + DIRS[moves[i]][0],
                snakes[i].head[1] + DIRS[moves[i]][1])
            for i in alive
        }

        # Out-of-bounds
        for i in alive:
            nx, ny = new_heads[i]
            if not (0 <= nx < WIDTH and 0 <= ny < HEIGHT):
                snakes[i].alive = False
                deaths[i] = turn
                if verbose:
                    print(f"    [{snakes[i].name}] died turn {turn}: out-of-bounds")

        alive = [i for i, s in enumerate(snakes) if s.alive]

        # Starvation (health checked after hazard damage below)
        # Body collision (tail-vacating aware)
        body_cells = _build_occupied_sim([snakes[i] for i in alive])
        for i in alive:
            if new_heads[i] in body_cells:
                snakes[i].alive = False
                deaths[i] = turn
                if verbose:
                    print(f"    [{snakes[i].name}] died turn {turn}: body collision")

        alive = [i for i, s in enumerate(snakes) if s.alive]

        # H2H
        head_pos: dict[tuple[int,int], list[int]] = defaultdict(list)
        for i in alive:
            head_pos[new_heads[i]].append(i)
        for pos, claimants in head_pos.items():
            if len(claimants) < 2:
                continue
            max_len = max(snakes[i].length for i in claimants)
            for i in claimants:
                if snakes[i].length < max_len:
                    snakes[i].alive = False
                    deaths[i] = turn
                    if verbose:
                        print(f"    [{snakes[i].name}] died turn {turn}: h2h (shorter)")
                else:
                    peers = [j for j in claimants if j != i and snakes[j].length == max_len]
                    if peers:
                        snakes[i].alive = False
                        deaths[i] = turn
                        if verbose:
                            print(f"    [{snakes[i].name}] died turn {turn}: h2h (equal)")

        alive = [i for i, s in enumerate(snakes) if s.alive]

        # Advance bodies; eat food
        for i in alive:
            ate = new_heads[i] in food
            snakes[i].body.insert(0, new_heads[i])
            if not ate:
                snakes[i].body.pop()
            else:
                snakes[i].health = START_HEALTH
                food.remove(new_heads[i])

        # Hunger + hazard damage
        hazard_set = set(HAZARD_CELLS)
        for i in alive:
            snakes[i].health -= HUNGER_PER_TURN
            if snakes[i].head in hazard_set:
                snakes[i].health -= HAZARD_DMG
                hazard_dmg_total += HAZARD_DMG
            if snakes[i].health <= 0:
                snakes[i].alive = False
                deaths[i] = turn
                on_hazard = snakes[i].head in hazard_set
                if on_hazard:
                    hazard_deaths += 1
                reason = "hazard+starved" if on_hazard else "starved"
                if verbose:
                    print(f"    [{snakes[i].name}] died turn {turn}: {reason}")

        if verbose:
            alive_now = [snakes[i].name for i in alive if snakes[i].alive]
            print(f"  Turn {turn:3d}: alive={alive_now}")

    alive = [i for i, s in enumerate(snakes) if s.alive]
    winner = alive[0] if len(alive) == 1 else None

    for i, agent in enumerate(agents):
        try:
            agent.end(_wrap(_mk_state(game_id, turn, snakes, food, i)))
        except Exception:
            pass

    return {
        "winner":       winner,
        "turns":        turn,
        "deaths":       deaths,
        "hazard_deaths":    hazard_deaths,
        "hazard_dmg_total": hazard_dmg_total,
    }


# ===========================================================================
# FIXED RandomAgent — excludes occupied body cells, not just neck/bounds
# ===========================================================================
class RandomAgent:
    """
    Picks a uniformly-random move that is:
      • in bounds
      • not the neck (immediate reversal)
      • not occupied by any currently-visible snake body segment
        (tail excluded since it will vacate — mirrors main.py logic)

    Falls back to any in-bounds move if all legal moves are blocked
    (genuinely boxed in), and finally to 'up' if even that fails.
    """

    def get_name(self) -> str:  return "Random"
    def get_color(self) -> str: return "#32CD32"
    def get_author(self) -> str: return "Chaos Itself"
    def start(self, gs) -> None: pass
    def end(self, gs)   -> None: pass

    def move(self, gs):
        data   = gs.model_dump() if hasattr(gs, "model_dump") else gs
        you    = data["you"]
        hx, hy = you["head"]["x"], you["head"]["y"]
        w      = data["board"]["width"]
        h      = data["board"]["height"]
        DIRS   = {"up": (0,1), "down": (0,-1), "left": (-1,0), "right": (1,0)}

        # Build occupied set from visible body segments (tail-vacating)
        occupied: set[tuple[int, int]] = set()
        for snake in data["board"].get("snakes", []):
            body = snake.get("body", [])
            if not body:
                continue
            tail_idx = len(body) - 1
            # Detect just-ate: server duplicates tail segment when snake grew
            just_ate = (
                len(body) >= 2
                and body[0] is not None
                and body[0] == body[-1]
            )
            for i, seg in enumerate(body):
                if seg is None:
                    continue
                if i == tail_idx and not just_ate:
                    continue   # tail will vacate
                occupied.add((seg["x"], seg["y"]))

        neck: Optional[tuple[int,int]] = None
        if len(you["body"]) > 1 and you["body"][1] is not None:
            n = you["body"][1]
            neck = (n["x"], n["y"])

        safe = []
        for d, (dx, dy) in DIRS.items():
            nx, ny = hx + dx, hy + dy
            if (0 <= nx < w and 0 <= ny < h
                    and (nx, ny) != neck
                    and (nx, ny) not in occupied):
                safe.append(d)

        chosen = random.choice(safe) if safe else random.choice([
            d for d, (dx, dy) in DIRS.items()
            if 0 <= hx + dx < w and 0 <= hy + dy < h
        ] or ["up"])

        try:
            from battlesnake_types import MoveAction, Direction
            dm = {"up": Direction.UP, "down": Direction.DOWN,
                  "left": Direction.LEFT, "right": Direction.RIGHT}
            return MoveAction(move=dm[chosen])
        except ImportError:
            return chosen


# ===========================================================================
# SafeFoodSeekingAgent — smarter baseline
# Never moves into an occupied cell; BFS-greedy toward nearest visible food
# when health < HUNGER_THRESHOLD; otherwise takes best-space safe move.
# ===========================================================================
class SafeFoodSeekingAgent:
    """
    A non-suicidal goal-directed baseline:
      • Builds the same tail-vacating occupied set as RandomAgent.
      • When health < HUNGER_THRESHOLD: BFS to nearest visible food; picks
        the direction of the first BFS step.
      • Otherwise: picks the safe move that maximises a simple flood-fill
        count (greedy space maximisation, no Voronoi).
      • Falls back to any safe move, then any in-bounds move.
    """

    HUNGER_THRESHOLD = 50

    def get_name(self) -> str:  return "SafeFood"
    def get_color(self) -> str: return "#FFA500"
    def get_author(self) -> str: return "Baseline"
    def start(self, gs) -> None: pass
    def end(self, gs)   -> None: pass

    @staticmethod
    def _occupied(data: dict) -> set[tuple[int,int]]:
        occ: set = set()
        for snake in data["board"].get("snakes", []):
            body = snake.get("body", [])
            if not body:
                continue
            just_ate = (
                len(body) >= 2
                and body[0] is not None
                and body[0] == body[-1]
            )
            tail_idx = len(body) - 1
            for i, seg in enumerate(body):
                if seg is None:
                    continue
                if i == tail_idx and not just_ate:
                    continue
                occ.add((seg["x"], seg["y"]))
        return occ

    @staticmethod
    def _flood_fill(start: tuple[int,int], occ: set, w: int, h: int) -> int:
        visited = {start}
        queue   = [start]
        ptr     = 0
        while ptr < len(queue):
            cx, cy = queue[ptr]; ptr += 1
            for dx, dy in ((0,1),(0,-1),(-1,0),(1,0)):
                nx, ny = cx+dx, cy+dy
                if 0 <= nx < w and 0 <= ny < h and (nx,ny) not in occ and (nx,ny) not in visited:
                    visited.add((nx,ny))
                    queue.append((nx,ny))
        return len(visited)

    @staticmethod
    def _bfs_first_step(
        start: tuple[int,int],
        targets: set[tuple[int,int]],
        occ: set,
        w: int,
        h: int,
    ) -> Optional[str]:
        """Returns the direction of the first BFS step toward nearest target, or None."""
        DIRS = {"up":(0,1),"down":(0,-1),"left":(-1,0),"right":(1,0)}
        # Seed with each direct neighbour, tagging which direction we came from
        visited = {start}
        queue: list[tuple[tuple[int,int], str]] = []   # (cell, first_dir)
        for d, (dx, dy) in DIRS.items():
            nx, ny = start[0]+dx, start[1]+dy
            if 0 <= nx < w and 0 <= ny < h and (nx,ny) not in occ:
                cell = (nx, ny)
                if cell in targets:
                    return d
                visited.add(cell)
                queue.append((cell, d))
        ptr = 0
        while ptr < len(queue):
            (cx, cy), first_dir = queue[ptr]; ptr += 1
            for dx, dy in ((0,1),(0,-1),(-1,0),(1,0)):
                nx, ny = cx+dx, cy+dy
                if 0 <= nx < w and 0 <= ny < h and (nx,ny) not in occ and (nx,ny) not in visited:
                    cell = (nx, ny)
                    if cell in targets:
                        return first_dir
                    visited.add(cell)
                    queue.append((cell, first_dir))
        return None

    def move(self, gs):
        data   = gs.model_dump() if hasattr(gs, "model_dump") else gs
        you    = data["you"]
        hx, hy = you["head"]["x"], you["head"]["y"]
        health = you["health"] or 100
        w      = data["board"]["width"]
        h      = data["board"]["height"]
        DIRS   = {"up":(0,1),"down":(0,-1),"left":(-1,0),"right":(1,0)}

        occ  = self._occupied(data)
        neck: Optional[tuple[int,int]] = None
        if len(you["body"]) > 1 and you["body"][1] is not None:
            n = you["body"][1]; neck = (n["x"], n["y"])

        safe = [
            d for d, (dx, dy) in DIRS.items()
            if (0 <= hx+dx < w and 0 <= hy+dy < h
                and (hx+dx, hy+dy) != neck
                and (hx+dx, hy+dy) not in occ)
        ]

        chosen = None

        # Hungry? BFS toward food
        if health < self.HUNGER_THRESHOLD and safe:
            food_set = {
                (f["x"], f["y"])
                for f in data["board"].get("food", [])
                if f is not None
            }
            if food_set:
                first = self._bfs_first_step((hx, hy), food_set, occ, w, h)
                if first and first in safe:
                    chosen = first

        # Fallback: maximise flood-fill space
        if chosen is None and safe:
            scores = {
                d: self._flood_fill((hx + dx, hy + dy), occ, w, h)
                for d, (dx, dy) in DIRS.items()
                if d in safe
            }
            chosen = max(scores, key=lambda d: scores[d])

        # Last resort: any in-bounds
        if chosen is None:
            ib = [d for d, (dx,dy) in DIRS.items()
                  if 0 <= hx+dx < w and 0 <= hy+dy < h]
            chosen = random.choice(ib) if ib else "up"

        try:
            from battlesnake_types import MoveAction, Direction
            dm = {"up": Direction.UP, "down": Direction.DOWN,
                  "left": Direction.LEFT, "right": Direction.RIGHT}
            return MoveAction(move=dm[chosen])
        except ImportError:
            return chosen


# ===========================================================================
# Main loop
# ===========================================================================
def main() -> None:
    num_games    = args.games
    verbose      = args.verbose
    latency_warn = args.latency_warn

    # Agent pool: الثعبان | SafeFood | Random | Random
    our_agent = ThuebanAgent()
    agents    = [our_agent, SafeFoodSeekingAgent(), RandomAgent(), RandomAgent()]

    print(f"\n{'='*60}")
    print(f"Batch test: {num_games} games  |  "
          f"simulator: {'hisss' if USE_HISSS else 'standalone'}")
    print(f"Hazards: {'ON (left edge, dmg=' + str(HAZARD_DMG) + '/turn)' if args.hazards else 'OFF'}")
    print(f"Agents:  {[a.get_name() for a in agents]}")
    print(f"{'='*60}\n")

    results       = []
    agent_wins    = defaultdict(int)
    agent_deaths  = defaultdict(int)
    game_timing_ms: list[float] = []
    total_haz_deaths  = 0
    total_haz_dmg     = 0

    for g in range(num_games):
        game_id = f"local-game-{g:04d}"
        t0 = time.monotonic()

        if USE_HISSS:
            # ----------------------------------------------------------------
            # Mode A: hisss
            # ----------------------------------------------------------------
            try:
                DIRECTION_TO_HISSS = {
                    "up":    hisss.UP,
                    "down":  hisss.DOWN,
                    "left":  hisss.LEFT,
                    "right": hisss.RIGHT,
                }
                from battlesnake_types import GameState
                game_cfg = hisss.restricted_standard_config()
                game_cfg.all_actions_legal = True
                env = hisss.BattleSnakeGame(game_cfg)

                for idx, agent in enumerate(agents):
                    cur_str   = hisss.to_battlesnake_json(env, idx)
                    cur_state = GameState.model_validate_json(cur_str)
                    agent.start(cur_state)

                done = False
                turns_played = 0
                while not done:
                    turns_played += 1
                    hisss_actions = []
                    alive_indices = env.alive_indices()
                    for idx in range(len(agents)):
                        if idx in alive_indices:
                            cur_str   = hisss.to_battlesnake_json(env, idx)
                            cur_state = GameState.model_validate_json(cur_str)
                            mv = agents[idx].move(cur_state)
                            ds = (str(mv.move).lower().split(".")[-1]
                                  if hasattr(mv, "move") else str(mv))
                            for k in DIRECTION_TO_HISSS:
                                if k in ds:
                                    ds = k; break
                            hisss_actions.append(DIRECTION_TO_HISSS[ds])
                        else:
                            hisss_actions.append(hisss.UP)
                    _, done, _ = env.step(actions=tuple(hisss_actions))

                alive = env.alive_indices()
                winner_idx = alive[0] if len(alive) == 1 else None

                for idx, agent in enumerate(agents):
                    cur_str   = hisss.to_battlesnake_json(env, 0)
                    cur_state = GameState.model_validate_json(cur_str)
                    agent.end(cur_state)

                result = {
                    "winner": winner_idx,
                    "turns":  turns_played,
                    "deaths": {},
                    "hazard_deaths": 0,
                    "hazard_dmg_total": 0,
                }
            except Exception as exc:
                print(f"  [Game {g}] hisss ERROR: {exc}")
                traceback.print_exc()
                result = {
                    "winner": None, "turns": 0, "deaths": {}, "error": str(exc),
                    "hazard_deaths": 0, "hazard_dmg_total": 0,
                }
        else:
            # ----------------------------------------------------------------
            # Mode B: standalone
            # ----------------------------------------------------------------
            try:
                result = _run_standalone_game(agents, game_id, verbose)
            except Exception as exc:
                print(f"  [Game {g}] standalone ERROR: {exc}")
                traceback.print_exc()
                result = {
                    "winner": None, "turns": 0, "deaths": {}, "error": str(exc),
                    "hazard_deaths": 0, "hazard_dmg_total": 0,
                }

        elapsed = (time.monotonic() - t0) * 1000
        game_timing_ms.append(elapsed)
        results.append(result)

        w = result["winner"]
        if w is not None:
            agent_wins[agents[w].get_name()] += 1
        for d_idx in result["deaths"]:
            agent_deaths[agents[d_idx].get_name()] += 1

        total_haz_deaths += result.get("hazard_deaths", 0)
        total_haz_dmg    += result.get("hazard_dmg_total", 0)

        winner_name = agents[w].get_name() if w is not None else "draw/timeout"
        status = " [ERR]" if "error" in result else ""
        print(
            f"  Game {g+1:3d}/{num_games}: turns={result['turns']:4d}  "
            f"winner={winner_name:<20s}{status}  ({elapsed:.0f} ms total)"
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for a in agents:
        name   = a.get_name()
        wins   = agent_wins.get(name, 0)
        deaths = agent_deaths.get(name, 0)
        print(f"  {name:<20s}  wins={wins:3d}/{num_games}  "
              f"({wins/num_games*100:.1f}%)  deaths={deaths}")

    draws     = sum(1 for r in results if r["winner"] is None and "error" not in r)
    errors    = sum(1 for r in results if "error" in r)
    avg_game  = sum(game_timing_ms) / len(game_timing_ms) if game_timing_ms else 0
    max_game  = max(game_timing_ms) if game_timing_ms else 0
    avg_turns = sum(r["turns"] for r in results) / len(results) if results else 0

    print(f"\n  Draws/timeouts   : {draws}")
    print(f"  Errors           : {errors}")
    print(f"  Avg game time    : {avg_game:.0f} ms  (max {max_game:.0f} ms)")
    print(f"  Avg game length  : {avg_turns:.1f} turns")

    # Per-move latency stats (from agent_adapter.move_latency_ms)
    lats = agent_adapter.move_latency_ms
    if lats:
        lats_sorted = sorted(lats)
        max_lat  = lats_sorted[-1]
        p95_idx  = max(0, int(len(lats_sorted) * 0.95) - 1)
        p95_lat  = lats_sorted[p95_idx]
        over_thr = sum(1 for t in lats if t > latency_warn)
        print(f"\n  --- Per-move latency (الثعبان, {len(lats)} moves) ---")
        print(f"  Max single move  : {max_lat:.1f} ms")
        print(f"  p95 single move  : {p95_lat:.1f} ms")
        print(f"  Moves > {latency_warn} ms    : {over_thr} / {len(lats)}")

    if args.hazards:
        print(f"\n  --- Hazard stats ---")
        print(f"  Hazard-zone deaths (any agent)  : {total_haz_deaths}")
        print(f"  Total hazard damage dealt       : {total_haz_dmg} hp")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
