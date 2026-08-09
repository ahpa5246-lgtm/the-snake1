import os
import sys

# Add the parent directory to the sys path so we can import main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import (
    TacticalEngine,
    GameContext,
    GamePhase,
    PHASE_WEIGHTS
)

def create_mock_context(
    our_head=(5, 5),
    our_body=None,
    our_len=3,
    our_health=100,
    our_tail=None,
    width=11,
    height=11,
    turn=10,
    occupied=None,
    hazard_set=None,
    hazard_dmg=14,
    enemy_data=None,
    visible_food=None,
    merged_food=None,
    phase=GamePhase.MID,
    ghost_zones=None,
    unseen_cells=None
):
    if our_body is None:
        our_body = [(5, 5), (5, 4), (5, 3)]
    if our_tail is None:
        our_tail = our_body[-1] if our_body else None
    if occupied is None:
        occupied = set(our_body)
        for x in range(width):
            occupied.add((x, -1))
            occupied.add((x, height))
        for y in range(height):
            occupied.add((-1, y))
            occupied.add((width, y))
    if hazard_set is None:
        hazard_set = set()
    if enemy_data is None:
        enemy_data = []
    if visible_food is None:
        visible_food = set()
    if merged_food is None:
        merged_food = set()
    if ghost_zones is None:
        ghost_zones = set()
    if unseen_cells is None:
        unseen_cells = set()

    return GameContext(
        our_head=our_head,
        our_body=our_body,
        our_len=our_len,
        our_health=our_health,
        our_tail=our_tail,
        width=width,
        height=height,
        turn=turn,
        occupied=occupied,
        hazard_set=hazard_set,
        hazard_dmg=hazard_dmg,
        enemy_data=enemy_data,
        visible_food=visible_food,
        merged_food=merged_food,
        phase=phase,
        weights=PHASE_WEIGHTS[phase],
        deadline=float('inf'),  # No timeout for tests
        ghost_zones=ghost_zones,
        unseen_cells=unseen_cells
    )


def test_is_certain_death_out_of_bounds():
    ctx = create_mock_context(our_head=(10, 10), width=11, height=11)
    
    # Candidate moving off the right edge (x=11)
    assert TacticalEngine._is_certain_death((11, 10), ctx) is True
    # Candidate moving off the top edge (y=11)
    assert TacticalEngine._is_certain_death((10, 11), ctx) is True


def test_is_certain_death_occupied():
    ctx = create_mock_context(our_head=(5, 5), occupied={(6, 5)})
    # Candidate moving into occupied cell
    assert TacticalEngine._is_certain_death((6, 5), ctx) is True


def test_is_certain_death_fatal_hazard():
    ctx = create_mock_context(our_head=(5, 5), hazard_set={(6, 5)}, hazard_dmg=14, our_health=10)
    # Candidate moving into hazard with health <= hazard_dmg (will die of hazard damage)
    assert TacticalEngine._is_certain_death((6, 5), ctx) is True


def test_is_certain_death_h2h_loss():
    enemy = {"id": "e1", "head_pos": (6, 6), "length": 5, "health": 100, "body": [(6,6), (7,6), (8,6)]}
    # We are length 3 (shorter)
    ctx = create_mock_context(our_head=(5, 5), our_len=3, enemy_data=[enemy])
    
    # Candidate (6, 5) is adjacent to enemy head (6, 6)
    assert TacticalEngine._is_certain_death((6, 5), ctx) is True


def test_flood_fill_space():
    # 3x3 box around (5,5) but entirely enclosed by occupied cells
    # (4,4) to (6,6)
    occ = set()
    for x in range(3, 8):
        occ.add((x, 3))
        occ.add((x, 7))
    for y in range(4, 7):
        occ.add((3, y))
        occ.add((7, y))
    
    space = TacticalEngine._flood_fill((5, 6), occ, limit=10, deadline=float('inf'))
    assert space == 9  # Enclosed space 3x3 is 9 cells


def test_bfs_dist():
    occ = {(6, 5), (6, 6), (6, 4)} # Wall blocking direct path
    dist = TacticalEngine._bfs_dist((5, 5), {(7, 5)}, occ, limit=10)
    assert dist > 2  # Has to go around the wall


def test_deep_escape_check():
    # Enclose (5,6) on 3 sides using an enemy body
    enemy = {
        "id": "e1", 
        "head_pos": (6,6), 
        "length": 5, 
        "health": 100, 
        "body": [(6,6), (5,7), (4,6), (4,7), (3,7)]
    }
    # Our body at (5,5), (5,4), (5,3)
    # Cand is (5,6). Neighbours: (6,6) is enemy, (5,7) is enemy, (4,6) is enemy, (5,5) is our body.
    # So space is exactly 1 (only cand).
    ctx = create_mock_context(
        our_head=(5, 5), 
        our_len=3, 
        our_body=[(5,5), (5,4), (5,3)],
        enemy_data=[enemy]
    )
    
    assert TacticalEngine._deep_escape_check((5, 6), ctx) is False


def test_score_move():
    ctx = create_mock_context(visible_food={(6, 5)}, merged_food={(6, 5)})
    # Test scoring a move towards food vs away
    score_towards = TacticalEngine._score_move((6, 5), ctx)
    score_away = TacticalEngine._score_move((4, 5), ctx)
    assert score_towards > score_away
