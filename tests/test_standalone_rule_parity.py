"""Regression tests for the standalone runner's official turn-order invariants."""
import unittest

import run_games


class StandaloneRulesTests(unittest.TestCase):
    def test_previous_head_remains_occupied_as_neck(self):
        first = run_games.Snake("a", "A", [(2, 2), (2, 1), (2, 0)])
        second = run_games.Snake("b", "B", [(4, 2), (4, 1), (4, 0)])
        cells = run_games._build_occupied_sim(
            [first, second], {0: (3, 2), 1: (4, 3)})
        self.assertIn((2, 2), cells)
        self.assertIn((4, 2), cells)
        self.assertNotIn((2, 0), cells)
        self.assertNotIn((4, 0), cells)

    def test_eating_tail_is_not_vacated(self):
        snake = run_games.Snake("eater", "Eater", [(2, 2), (2, 1), (1, 1)])
        cells = run_games._build_occupied_sim(
            [snake], {0: (3, 2)}, retained_tails={"eater"})
        self.assertIn((1, 1), cells)

    def test_stacked_tail_always_blocks(self):
        snake = run_games.Snake("s", "S", [(2, 2), (2, 1), (1, 1), (1, 1)])
        self.assertIn((1, 1), run_games._build_occupied_sim([snake]))

    def test_single_cell_snake_must_not_vacate_its_head(self):
        snake = run_games.Snake("s", "S", [(1, 1)])
        self.assertIn((1, 1), run_games._build_occupied_sim([snake]))

    def test_food_restores_full_health_even_in_hazard(self):
        old = run_games.HAZARD_DMG
        try:
            run_games.HAZARD_DMG = 14
            self.assertEqual(run_games._next_health(1, ate=True, on_hazard=True), 100)
            self.assertEqual(run_games._next_health(50, ate=False, on_hazard=True), 35)
            self.assertEqual(run_games._next_health(50, ate=False, on_hazard=False), 49)
        finally:
            run_games.HAZARD_DMG = old


if __name__ == "__main__":
    unittest.main()
