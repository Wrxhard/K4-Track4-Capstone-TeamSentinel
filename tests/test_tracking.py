import unittest

from sentinel import Detection, ObjectTracker, relation_inputs


def bag(x=100):
    return Detection((x, 100, x + 40, 140), 0.9, "backpack")


class TrackingTests(unittest.TestCase):
    def test_occluded_bag_keeps_id_and_last_stationary_position(self):
        tracker = ObjectTracker(fps=10, hold_seconds=2)
        first = tracker.update([bag()], 0)[0]
        hidden = tracker.update([], 0.1)[0]
        self.assertEqual(first.track_id, hidden.track_id)
        self.assertFalse(hidden.observed)
        self.assertEqual(hidden.bbox, first.bbox)
        self.assertEqual(relation_inputs([hidden]), [])
        returned = tracker.update([bag()], 0.2)[0]
        self.assertEqual(first.track_id, returned.track_id)
        self.assertTrue(returned.observed)

    def test_track_expires_by_elapsed_time(self):
        tracker = ObjectTracker(fps=10, hold_seconds=1)
        first = tracker.update([bag()], 0)[0]
        self.assertEqual(tracker.update([], 1.1), [])
        returned = tracker.update([bag()], 1.2)
        if not returned:
            returned = tracker.update([bag()], 1.3)
        self.assertNotEqual(first.track_id, returned[0].track_id)

    def test_overlapping_classes_do_not_share_ids(self):
        tracker = ObjectTracker(fps=10)
        tracks = tracker.update([bag(), Detection(bag().bbox, 0.9, "person")], 0)
        self.assertEqual(len({t.track_id for t in tracks}), 2)
        person = next(t for t in tracks if t.label == "person")
        after = tracker.update([bag()], 0.1)
        self.assertFalse(
            next(t for t in after if t.track_id == person.track_id).observed
        )

    def test_motion_uses_kalman_prediction_during_gap(self):
        tracker = ObjectTracker(fps=10, hold_seconds=2)
        for i in range(8):
            track = tracker.update([bag(100 + i * 3)], i / 10)[0]
        hidden = tracker.update([], 0.8)[0]
        self.assertGreater(hidden.bbox[0], track.bbox[0])
        self.assertFalse(hidden.observed)

    def test_timestamp_must_increase(self):
        tracker = ObjectTracker(fps=10)
        tracker.update([bag()], 0)
        with self.assertRaises(ValueError):
            tracker.update([], 0)


if __name__ == "__main__":
    unittest.main()
