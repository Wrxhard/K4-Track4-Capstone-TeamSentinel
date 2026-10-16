import unittest
from dataclasses import replace

from alerts import AbandonmentMonitor
from sentinel import Track


def person(id=1, x=0, now=0):
    return Track(id, "person", (x, 0, x + 60, 180), 0.9, now)


def bag(now=0, observed=True):
    return Track(3, "backpack", (40, 100, 80, 140), 0.9, now, observed)


def relation(now=0, owner=1, score=0.9, predicate="holding"):
    return {
        "subject_id": owner,
        "object_id": 3,
        "score": score,
        "predicate": predicate,
        "evaluated_at": now,
    }


class AlertTests(unittest.TestCase):
    def seed(self):
        engine = AbandonmentMonitor(
            exit_seconds=3, unattended_seconds=5, owner_samples=2
        )
        engine.update([person(), person(2, 300), bag()], [relation()], 0)
        engine.update([person(now=1), person(2, 300, 1), bag(1)], [relation(1)], 1)
        return engine

    def test_owner_leaves_while_other_person_remains_one_alert(self):
        engine = self.seed()
        self.assertEqual(
            engine.update([person(2, 300, 2), bag(2)], [relation(2, 2, 0.1)], 2), []
        )
        events = engine.update([person(2, 300, 5), bag(5)], [relation(5, 2, 0.1)], 5)
        self.assertEqual(events[0]["reason"], "OWNER_LEFT_CAMERA")
        self.assertEqual(events[0]["owner_id"], 1)
        self.assertEqual(engine.update([person(2, 300, 6), bag(6)], [], 6), [])

    def test_occluded_object_no_new_alert_and_return_preserves_owner(self):
        engine = self.seed()
        engine.update([bag(2)], [], 2)
        self.assertEqual(engine.update([bag(2, False)], [], 8), [])
        self.assertEqual(engine.update([bag(9)], [], 9)[0]["owner_id"], 1)

    def test_owner_occluded_briefly_does_not_alert(self):
        engine = self.seed()
        self.assertEqual(
            engine.update([replace(person(now=1), observed=False), bag(2)], [], 2), []
        )
        self.assertEqual(engine.update([person(now=3), bag(3)], [], 3), [])

    def test_owner_returns_resolves_then_new_episode_can_alert(self):
        engine = self.seed()
        engine.update([bag(2)], [], 2)
        alert = engine.update([bag(5)], [], 5)[0]
        resolved = engine.update([person(now=6), bag(6)], [], 6)[0]
        self.assertEqual(resolved["type"], "RESOLVED")
        self.assertEqual(resolved["alert_id"], alert["alert_id"])
        engine.update([bag(7)], [], 7)
        second = engine.update([bag(10)], [], 10)[0]
        self.assertNotEqual(second["alert_id"], alert["alert_id"])

    def test_low_near_and_stale_relations_cannot_assign_owner(self):
        for edge in (relation(score=0.1), relation(predicate="near")):
            engine = AbandonmentMonitor(owner_samples=1)
            engine.update([person(), bag()], [edge], 0)
            self.assertIsNone(engine.snapshot()[0]["owner_id"])
        engine = AbandonmentMonitor(owner_samples=2)
        engine.update([person(), bag()], [relation()], 0)
        engine.update([person(now=1), bag(1)], [relation()], 1)
        self.assertIsNone(engine.snapshot()[0]["owner_id"])

    def test_ambiguous_candidates_remain_unassigned(self):
        engine = AbandonmentMonitor(owner_samples=1)
        engine.update(
            [person(), person(2, 50), bag()],
            [relation(), relation(owner=2, score=0.89)],
            0,
        )
        self.assertIsNone(engine.snapshot()[0]["owner_id"])

    def test_unattended_timeout_with_owner_visible(self):
        engine = self.seed()
        engine.update([person(x=500, now=2), bag(2)], [], 2)
        self.assertEqual(
            engine.update([person(x=500, now=8), bag(8)], [], 8)[0]["reason"],
            "UNATTENDED_TIMEOUT",
        )

    def test_moving_object_and_unowned_object_do_not_alert(self):
        engine = self.seed()
        self.assertEqual(
            engine.update([replace(bag(10), stationary=False)], [], 10), []
        )
        engine = AbandonmentMonitor()
        self.assertEqual(engine.update([bag()], [], 0), [])
        self.assertEqual(engine.update([bag(100)], [], 100), [])


if __name__ == "__main__":
    unittest.main()
