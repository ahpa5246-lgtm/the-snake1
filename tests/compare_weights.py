import argparse, json, os, random, sys, time
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
STARTER = os.path.join(os.path.dirname(REPO), "bs-blackout-starter")
if os.path.isdir(STARTER):
    sys.path.insert(0, STARTER)

_argv = sys.argv
sys.argv = [_argv[0]]
import run_games as rg
sys.argv = _argv

import hisss
import main as engine
from agent_adapter import ThuebanAgent

HAS_HUNGRY = False
try:
    from hungry_agent import HungryAgent
    from battlesnake_types import GameState as StarterGameState
    HAS_HUNGRY = True
except ImportError:
    pass

DIR_MAP = {"up": hisss.UP, "down": hisss.DOWN, "left": hisss.LEFT, "right": hisss.RIGHT}


def to_direction(mv) -> str:
    if hasattr(mv, "move"):
        mv = mv.move
    if hasattr(mv, "value"):
        return str(mv.value).lower()
    return str(mv).lower()


class Candidate:
    def __init__(self, weights, tag):
        self.weights = {engine.GamePhase(p): w for p, w in weights.items()}
        self.tag = tag
        self.agent = ThuebanAgent()

    def get_name(self):
        return f"cand-{self.tag}"

    def _tag(self, d):
        d = dict(d)
        d["game"] = dict(d.get("game", {}))
        d["game"]["id"] = f"{self.tag}:{d['game'].get('id', 'g')}"
        return d

    def start(self, d):
        self.agent.start(rg._wrap(self._tag(d)))

    def move(self, d):
        engine.PHASE_WEIGHTS.update(self.weights)
        return self.agent.move(rg._wrap(self._tag(d)))

    def end(self, d):
        self.agent.end(rg._wrap(self._tag(d)))


class HungryWrap:
    def __init__(self):
        self.a = HungryAgent()

    def get_name(self):
        return "hungry"

    def start(self, d):
        try:
            self.a.start(StarterGameState.model_validate(d))
        except Exception:
            pass

    def move(self, d):
        return self.a.move(StarterGameState.model_validate(d))

    def end(self, d):
        try:
            self.a.end(StarterGameState.model_validate(d))
        except Exception:
            pass


def make_opponents():
    pool = [rg.RandomAgent(), rg.SafeFoodSeekingAgent()]
    if HAS_HUNGRY:
        pool.append(HungryWrap())
    return pool


def check_fog(data):
    you_id = data["you"]["id"]
    hidden_segments = 0
    seen_enemies = 0
    for s in data["board"]["snakes"]:
        if s["id"] == you_id:
            continue
        seen_enemies += 1
        hidden_segments += sum(1 for b in s["body"] if b is None or (b.get("x") == -1 and b.get("y") == -1))
    return hidden_segments, seen_enemies


def play_game(players, max_turns=400, fog_idx=None, fog_stats=None):
    cfg = hisss.restricted_standard_config()
    cfg.all_actions_legal = False
    cfg.num_players = len(players)
    env = hisss.BattleSnakeGame(cfg)

    n = len(players)
    elim_turn = [None] * n
    unsafe = [0] * n

    for i, p in enumerate(players):
        d = json.loads(hisss.to_battlesnake_json(env, i))
        try:
            p.start(d)
        except Exception:
            pass

    turn = 0
    prev_alive = set(env.players_alive())
    while not env.is_terminal() and turn < max_turns:
        turn += 1
        order = env.players_at_turn()
        actions = []
        for idx in order:
            d = json.loads(hisss.to_battlesnake_json(env, idx))
            if fog_idx is not None and idx == fog_idx and fog_stats is not None:
                hidden, seen = check_fog(d)
                fog_stats["checks"] += 1
                fog_stats["hidden_hits"] += int(hidden > 0)
                fog_stats["occluded_enemy_count"] += max(0, (len(env.players_alive()) - 1) - seen)
            try:
                mv = players[idx].move(d)
                mv = to_direction(mv)
            except Exception:
                mv = "up"
            legal = env.available_actions(idx)
            h = DIR_MAP.get(mv, hisss.UP)
            if h not in legal:
                unsafe[idx] += 1
                h = legal[0] if legal else hisss.UP
            actions.append(h)
        env.step(actions=tuple(actions))
        cur_alive = set(env.players_alive())
        for dead in prev_alive - cur_alive:
            elim_turn[dead] = turn
        prev_alive = cur_alive

    alive = env.players_alive()
    winner = alive[0] if len(alive) == 1 else None
    lengths = env.player_lengths()

    for i, p in enumerate(players):
        d = json.loads(hisss.to_battlesnake_json(env, i, include_eliminated=True))
        try:
            p.end(d)
        except Exception:
            pass

    return {
        "winner": winner,
        "turns": turn,
        "players": [
            {"elim_turn": elim_turn[i], "survived": elim_turn[i] is None,
             "length": int(lengths[i]), "unsafe": unsafe[i]}
            for i in range(n)
        ],
    }


def run_batch(weights_path, n_games, label):
    with open(weights_path) as f:
        weights = json.load(f)
    totals = defaultdict(float)
    placements = []
    per_game = []
    fog_stats = {"checks": 0, "hidden_hits": 0, "occluded_enemy_count": 0}

    for g in range(n_games):
        cand = Candidate(weights, f"{label}{g}")
        opp = make_opponents()
        random.shuffle(opp)
        players = [cand] + opp[:3]
        while len(players) < 4:
            players.append(rg.RandomAgent())
        random.shuffle(players)
        my_idx = players.index(cand)

        t0 = time.monotonic()
        result = play_game(players, fog_idx=my_idx, fog_stats=fog_stats)
        dt = time.monotonic() - t0

        me = result["players"][my_idx]
        win = result["winner"] == my_idx
        n_players = len(players)
        if win:
            placement = 1
        elif me["survived"]:
            placement = 2
        else:
            later = sum(1 for p in result["players"] if p["elim_turn"] and p["elim_turn"] > me["elim_turn"])
            placement = n_players - later
        placements.append(placement)

        totals["games"] += 1
        totals["wins"] += int(win)
        totals["survived"] += int(me["survived"])
        totals["turns"] += (me["elim_turn"] or result["turns"])
        totals["length"] += me["length"]
        totals["unsafe"] += me["unsafe"]
        totals["time"] += dt

        per_game.append({
            "game": g, "winner": win, "survived": me["survived"],
            "turns": me["elim_turn"] or result["turns"], "length": me["length"],
            "unsafe_moves": me["unsafe"], "placement": placement,
        })

    n = totals["games"]
    summary = {
        "label": label, "games": int(n),
        "win_rate": totals["wins"] / n,
        "survival_rate": totals["survived"] / n,
        "avg_placement": sum(placements) / n,
        "avg_turns": totals["turns"] / n,
        "avg_length": totals["length"] / n,
        "unsafe_moves": int(totals["unsafe"]),
        "avg_time_s": totals["time"] / n,
        "fog_checks": fog_stats["checks"],
        "fog_hidden_hits": fog_stats["hidden_hits"],
        "fog_occluded_enemy_observations": fog_stats["occluded_enemy_count"],
    }
    return summary, per_game


def print_row(name, a, b, suf="", mult=1):
    print(f"{name:22s}{a*mult:10.2f}{suf}{b*mult:12.2f}{suf}")


def compare(prod_path, trained_path, n_games, tag):
    print(f"\n--- {tag}: {n_games} games each ---")
    prod, prod_g = run_batch(prod_path, n_games, "prod")
    trained, trained_g = run_batch(trained_path, n_games, "trained")

    print(f"{'':22s}{'PRODUCTION':>10s}{'TRAINED':>13s}")
    print_row("Win rate", prod["win_rate"], trained["win_rate"], "%", 100)
    print_row("Survival", prod["survival_rate"], trained["survival_rate"], "%", 100)
    print_row("Avg placement", prod["avg_placement"], trained["avg_placement"])
    print_row("Avg survival turns", prod["avg_turns"], trained["avg_turns"])
    print_row("Avg length", prod["avg_length"], trained["avg_length"])
    print_row("Unsafe moves", prod["unsafe_moves"], trained["unsafe_moves"])
    print_row("Avg time/move (s)", prod["avg_time_s"], trained["avg_time_s"])

    fog_ok = prod["fog_checks"] > 0 and (prod["fog_hidden_hits"] > 0 or prod["fog_occluded_enemy_observations"] > 0)
    print(f"\nfog-of-war evidence: hidden_hits={prod['fog_hidden_hits']}/{prod['fog_checks']} checks, "
          f"occluded_enemy_observations={prod['fog_occluded_enemy_observations']}")
    if not fog_ok:
        print("WARNING: no evidence of hidden/occluded enemy data observed -- verify fog-of-war before trusting results")

    diff = trained["win_rate"] - prod["win_rate"]
    verdict = "BETTER" if diff > 0.02 else ("WORSE" if diff < -0.02 else "INCONCLUSIVE")
    print(f"\nRESULT: TRAINED WEIGHTS = {verdict}  (win rate diff {diff*100:+.1f} pts, "
          f"survival turns diff {trained['avg_turns']-prod['avg_turns']:+.1f})")

    return prod, trained, prod_g, trained_g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()

    prod_path = os.path.join(REPO, "weights.json")
    trained_path = os.path.join(REPO, "training", "checkpoints", "best_weights.json")
    out_dir = os.path.join(REPO, "testing")
    os.makedirs(out_dir, exist_ok=True)

    if not args.skip_smoke:
        compare(prod_path, trained_path, 10, "smoke test")
        print("\nsmoke test done, proceeding to full run")

    prod, trained, prod_g, trained_g = compare(prod_path, trained_path, args.games, "full run")

    out = {"production": prod, "trained": trained,
           "production_games": prod_g, "trained_games": trained_g}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    with open(os.path.join(out_dir, "results.csv"), "w") as f:
        f.write("label,game,winner,survived,turns,length,unsafe_moves,placement\n")
        for label, games in (("prod", prod_g), ("trained", trained_g)):
            for r in games:
                f.write(f"{label},{r['game']},{r['winner']},{r['survived']},{r['turns']},"
                        f"{r['length']},{r['unsafe_moves']},{r['placement']}\n")

    print(f"\nsaved -> {os.path.join(out_dir, 'results.json')}")
    print(f"saved -> {os.path.join(out_dir, 'results.csv')}")


if __name__ == "__main__":
    main()