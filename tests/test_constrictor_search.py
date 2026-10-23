"""Constrictor permanent-growth path search regressions."""
import time
import unittest

from constrictor_search import route_quality


class ConstrictorPathTests(unittest.TestCase):
    def test_rejects_blocked_cell_and_outside_board(self):
        self.assertEqual(route_quality((1, 1), width=3, height=3, occupied={(1, 1)}), (0, 0, 0))
        self.assertEqual(route_quality((-1, 0), width=3, height=3, occupied=set()), (0, 0, 0))

    def test_dead_end_has_only_one_survivable_turn(self):
        self.assertEqual(route_quality(
            (1, 1), width=3, height=3,
            occupied={(0, 1), (2, 1), (1, 0), (1, 2)},
        )[0], 1)

    def test_open_corridor_has_more_future_moves_than_dead_end(self):
        available = {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
                     (4, 1), (4, 2), (4, 3), (4, 4), (3, 4)}
        blocked = {(x, y) for x in range(5) for y in range(5)} - available
        self.assertGreater(route_quality((0, 0), width=5, height=5, occupied=blocked)[0], 5)
        self.assertEqual(route_quality((3, 4), width=5, height=5,
                                       occupied=blocked | {(4, 4)})[0], 1)

    def test_opponents_reduce_uncontested_path_score(self):
        occupied = {(2, 2)}
        with_enemy = route_quality((1, 1), width=5, height=5,
                                   occupied=occupied, opponent_heads=((4, 1),))
        no_enemy = route_quality((1, 1), width=5, height=5, occupied=occupied)
        self.assertLessEqual(with_enemy[1], no_enemy[1])

    def test_honors_exhausted_deadline(self):
        start = time.monotonic()
        quality = route_quality((5, 5), width=11, height=11,
                                occupied=set(), deadline=start)
        self.assertEqual(quality[0], 1)
        self.assertLess(time.monotonic() - start, 0.1)

    def test_production_route_selection_uses_longer_constrictor_plan(self):
        from unittest.mock import patch
        from main import TacticalEngine
        point = lambda x, y: {"x": x, "y": y}
        body = [point(2, 2), point(2, 1), point(2, 0)]
        our = {"id": "us", "health": 100, "length": 3,
               "head": body[0], "body": body}
        raw = {
            "game": {"id": "constrictor-test",
                     "ruleset": {"name": "constrictor",
                                 "settings": {"hazardDamagePerTurn": 0}}},
            "turn": 16,
            "board": {"width": 7, "height": 7,
                      "snakes": [our], "hazards": [], "food": []},
            "you": our,
        }
        def fake_quality(point, **kwargs):
            return (12, 12, 3) if point == (2, 3) else (3, 3, 1)
        with patch.object(TacticalEngine, "_is_safe",
                          side_effect=lambda position, ctx: position in ((2, 3), (3, 2))):
            with patch("main.constrictor_route_quality", side_effect=fake_quality):
                with patch.object(TacticalEngine, "_score_move", return_value=0):
                    direction = TacticalEngine.get_best_move(
                        raw, set(), "test", time.monotonic() + 1,
                        advisor=type("NoAdvisor", (), {"scores": staticmethod(
                            lambda *_args: {})}),
                    )
        self.assertEqual(direction, "up")

    def test_no_external_dependencies_or_randomness(self):
        first = route_quality((3, 3), width=7, height=7, occupied={(3, 2)})
        self.assertEqual(first, route_quality((3, 3), width=7, height=7, occupied={(3, 2)}))


if __name__ == "__main__":
    unittest.main()
