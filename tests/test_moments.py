import unittest
from datetime import datetime

import numpy as np

from six_panel_vrot import matching_moment_sweep


class SplitCutRadar:
    nsweeps = 2
    elevation = {"data": np.array([.5, .5])}
    time = {"data": np.array([0., 20.])}
    fields = {
        "velocity": {"data": np.ma.array([[0.], [20.]], mask=[[True], [False]])},
        "reflectivity": {"data": np.ma.array([[40.], [0.]], mask=[[False], [True]])},
    }

    def get_slice(self, sweep):
        return slice(sweep, sweep + 1)


class FakePyart:
    class util:
        @staticmethod
        def datetime_from_radar(_radar):
            return datetime(2011, 3, 9, 13, 21)


class MomentTests(unittest.TestCase):
    def test_matches_reflectivity_from_separate_cut(self):
        self.assertEqual(matching_moment_sweep(SplitCutRadar(), 1,
                                              "reflectivity", FakePyart), 0)


if __name__ == "__main__":
    unittest.main()
