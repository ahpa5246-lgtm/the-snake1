import json
import sys
import run_games as rg
import main as engine
from training.train_hisss import _CandidateAgent, play_hisss_game, load_production_weights

def main():
    print("Loading New Champion (weights.json)")
    new_weights = load_production_weights() # loads weights.json
    
    print("Loading Old Champion (weights.json.bak.1786319200)")
    with open("weights.json.bak.1786319200") as f:
        old_weights = json.load(f)

    # Convert keys to GamePhase enums just like train_hisss does
    new_w = {engine.GamePhase(p): w for p, w in new_weights.items()}
    old_w = {engine.GamePhase(p): w for p, w in old_weights.items()}

    # Let's run a 4-player game with 2 New and 2 Old champions
    players = [
        _CandidateAgent(new_weights, "NEW_1"),
        _CandidateAgent(old_weights, "OLD_1"),
        _CandidateAgent(new_weights, "NEW_2"),
        _CandidateAgent(old_weights, "OLD_2"),
    ]

    print("Running Self-Play 4-Player Match: 2 NEW vs 2 OLD...")
    wins = {"NEW": 0, "OLD": 0, "DRAW": 0}
    
    for i in range(10):
        res = play_hisss_game(players)
        w = res["winner"]
        if w is None:
            wins["DRAW"] += 1
        elif w in [0, 2]:
            wins["NEW"] += 1
        else:
            wins["OLD"] += 1
        print(f"Game {i+1}: Winner = {'NEW' if w in [0,2] else 'OLD' if w is not None else 'DRAW'} | Turns = {res['turns']}")
    
    print("Self-Play Results (10 games):", wins)

if __name__ == "__main__":
    main()
