"""Rule invariants for the deterministic tactical search layer."""
import time
import unittest
from types import SimpleNamespace

from strategy_search import (game_mode, head_to_head_risk, is_constrictor,
                             move_body, safe_food_distance, survive_horizon)


def context(body, *, food=(), hazards=(), occupied=(), mode='standard', health=90, width=7, height=7):
    walls = {(x, -1) for x in range(width)} | {(x, height) for x in range(width)}
    walls |= {(-1, y) for y in range(height)} | {(width, y) for y in range(height)}
    return SimpleNamespace(
        our_body=list(body), occupied=walls | set(occupied) | set(body),
        visible_food=set(food), hazard_set=set(hazards), hazard_dmg=14,
        our_health=health, width=width, height=height, game_mode=mode,
        deadline=time.monotonic() + 2, enemy_data=[],
    )


class StrategySearchTests(unittest.TestCase):
    def test_constrictor_ruleset_detected(self):
        self.assertTrue(is_constrictor(game_mode({'game': {'ruleset': {'name': 'constrictor'}}})))

    def test_normal_tail_moves(self):
        result = move_body(((2, 2), (2, 1), (1, 1)), (1, 1), frozenset(), 90, set(), 14, set(), 7, 7, 'standard')
        self.assertEqual(result[0], ((1, 1), (2, 2), (2, 1)))

    def test_constrictor_tail_stays_blocked(self):
        result = move_body(((2, 2), (2, 1), (1, 1)), (1, 1), frozenset(), 90, set(), 14, set(), 7, 7, 'constrictor')
        self.assertIsNone(result)

    def test_constrictor_grows_on_every_turn(self):
        result = move_body(((2, 2), (2, 1), (1, 1)), (2, 3), frozenset(), 90, set(), 14, set(), 7, 7, 'constrictor')
        self.assertEqual(len(result[0]), 4)

    def test_stack_tail_does_not_vacate(self):
        self.assertIsNone(move_body(((2, 2), (2, 1), (1, 1), (1, 1)), (1, 1), frozenset(), 90, set(), 14, set(), 7, 7, 'standard'))

    def test_food_at_one_hp_is_safe(self):
        result = move_body(((2, 2),), (2, 3), frozenset({(2, 3)}), 1, {(2, 3)}, 14, set(), 7, 7, 'standard')
        self.assertEqual(result[2], 100)

    def test_nonfood_one_hp_dies(self):
        self.assertIsNone(move_body(((2, 2),), (2, 3), frozenset(), 1, set(), 14, set(), 7, 7, 'standard'))

    def test_h2h_is_possible_not_certain(self):
        enemy = {'head_pos': (4, 3), 'length': 4, 'body': [(4, 3), (5, 3), (6, 3)]}
        self.assertEqual(head_to_head_risk((3, 3), [enemy], {(4, 3), (5, 3), (6, 3)}, 3, 7, 7, 'standard'), 2)
        self.assertEqual(head_to_head_risk((2, 3), [enemy], {(4, 3), (5, 3), (6, 3)}, 3, 7, 7, 'standard'), 0)

    def test_winning_h2h_classification(self):
        enemy = {'head_pos': (4, 3), 'length': 2, 'body': [(4, 3), (4, 2)]}
        self.assertEqual(head_to_head_risk((3, 3), [enemy], {(4, 3), (4, 2)}, 3, 7, 7, 'standard'), 1)

    def test_survival_search_rejects_a_real_dead_end(self):
        ctx = context([(2, 2), (2, 1), (1, 1)], occupied={(1, 3), (3, 3), (2, 4)})
        self.assertFalse(survive_horizon((2, 3), ctx, depth=3))

    def test_survival_search_accepts_open_board(self):
        ctx = context([(2, 2), (2, 1), (1, 1)])
        self.assertTrue(survive_horizon((2, 3), ctx, depth=5))

    def test_hazard_route_requires_sufficient_health(self):
        self.assertIsNone(safe_food_distance((0, 0), {(2, 0)}, set(), {(1, 0)}, 12, 14, 3, 1, 'standard'))
        self.assertEqual(safe_food_distance((0, 0), {(2, 0)}, set(), {(1, 0)}, 30, 14, 3, 1, 'standard'), 2)

    def test_food_can_be_eaten_in_hazard(self):
        self.assertEqual(safe_food_distance((0, 0), {(1, 0)}, set(), {(1, 0)}, 1, 14, 3, 1, 'standard'), 1)


if __name__ == '__main__':
    unittest.main()
