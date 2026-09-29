import unittest

import numpy as np

from couplet import detect_couplet


class CoupletTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y = np.meshgrid(np.linspace(-2, 2, 41), np.linspace(-2, 2, 41))
        self.velocity = np.zeros_like(self.x)
        self.velocity[(self.x > -.8) & (self.x < -.3) & (abs(self.y) < .5)] = -35
        self.velocity[(self.x > .3) & (self.x < .8) & (abs(self.y) < .5)] = 45

    def detect(self, data):
        return detect_couplet(data, self.x, self.y, (0., 0.), radar_xy_nm=(0., -30.))

    def test_centroids_ignore_isolated_extreme(self):
        self.velocity[20, 20] = 160
        pair = self.detect(self.velocity)
        self.assertIsNotNone(pair)
        self.assertAlmostEqual(pair.inbound.velocity, -35)
        self.assertAlmostEqual(pair.outbound.velocity, 45)
        self.assertAlmostEqual(pair.vrot, 40)
        self.assertGreater(pair.raw_vrot, pair.vrot)

    def test_no_inbound_lobe(self):
        self.velocity[self.velocity < 0] = 0
        self.assertIsNone(self.detect(self.velocity))

    def test_reject_radially_separated_sides(self):
        v = np.zeros_like(self.velocity)
        v[(abs(self.x) < .3) & (self.y > -.8) & (self.y < -.3)] = -35
        v[(abs(self.x) < .3) & (self.y > .3) & (self.y < .8)] = 45
        self.assertIsNone(self.detect(v))


if __name__ == "__main__":
    unittest.main()
