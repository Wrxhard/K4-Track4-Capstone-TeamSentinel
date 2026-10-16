import unittest

from sentinel import Detection, ObjectTracker, relation_inputs


def bag(x=100):
    return Detection((x, 100, x + 40, 140), 0.9, "backpack")


class TrackingTests(unittest.TestCase):
    def test_new_object_does_not_take_id_from_same_frame_returning_object(self):
        tracker = ObjectTracker(fps=30)
        original = tracker.update([bag(100)], 0)[0]
        tracker.update([], 0.1)
        returned = [t for t in tracker.update([bag(100), bag(200)], 0.2) if t.observed]
        original_now = min(returned, key=lambda t: t.bbox[0])
        self.assertEqual(original_now.track_id, original.track_id)
        self.assertEqual(len({t.track_id for t in returned}), 2)

    def test_stationary_object_keeps_id_after_nine_second_occlusion(self):
        tracker = ObjectTracker(fps=30, hold_seconds=10)
        for i in range(10):
            original = tracker.update([bag(100 + i * 0.1)], i / 30)[0]
        for i in range(10, 280):
            tracker.update([], i / 30)
        returned = next(t for t in tracker.update([bag(100.9)], 280 / 30) if t.observed)
        self.assertEqual(returned.track_id, original.track_id)

    def test_nearby_different_bag_must_not_steal_lost_bag_id(self):
        tracker = ObjectTracker(fps=30)
        original = tracker.update([bag(100)], 0)[0]
        tracker.update([], 0.1)
        returned = tracker.update([bag(200)], 0.2)
        new = next(t for t in returned if t.observed)
        self.assertNotEqual(original.track_id, new.track_id)

    def test_ambiguous_return_between_two_bags_does_not_merge(self):
        tracker = ObjectTracker(fps=30)
        originals = tracker.update([bag(100), bag(160)], 0)
        tracker.update([], 0.1)
        returned = tracker.update([bag(130)], 0.2)
        new = next(t for t in returned if t.observed)
        self.assertNotIn(new.track_id, {t.track_id for t in originals})

    def test_weak_detection_keeps_existing_track_without_creating_noise_track(self):
        tracker = ObjectTracker(fps=30, new_track_confidence=0.35)
        original = tracker.update([bag()], 0)[0]
        weak = Detection(bag().bbox, 0.18, "backpack")
        returned = tracker.update([weak], 0.1)
        self.assertEqual(returned[0].track_id, original.track_id)
        self.assertTrue(returned[0].observed)
        returned = tracker.update(
            [weak, Detection((500, 100, 540, 140), 0.18, "backpack")], 0.2
        )
        self.assertEqual(len([t for t in returned if t.observed]), 1)

    def test_new_detection_after_empty_scene_is_visible_immediately(self):
        tracker = ObjectTracker(fps=30)
        tracker.update([], 0)
        tracker.update([Detection((0, 0, 60, 180), 0.9, "person")], 0.1)
        tracks = tracker.update([Detection((0, 0, 60, 180), 0.9, "person"), bag()], 0.2)
        self.assertTrue(any(t.label == "backpack" and t.observed for t in tracks))
        shifted = tracker.update([bag(500)], 0.3)
        self.assertTrue(any(t.label == "backpack" and t.observed for t in shifted))

    def test_detection_at_configured_confidence_does_not_disappear(self):
        tracker = ObjectTracker(fps=30)
        tracks = tracker.update([Detection((100, 100, 140, 140), 0.3, "backpack")], 0)
        self.assertEqual(len(tracks), 1)
        self.assertTrue(tracks[0].observed)

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
