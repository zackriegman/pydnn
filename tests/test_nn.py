from pydnn import nn
import unittest


class TestNonlinearities(unittest.TestCase):
    def test_ReLU(self):
        self.assertEqual(
            [nn.relu(x) for x in [-10**8, -1, -0.5, 0, 0.5, 1, 10**8]],
            [0, 0, 0, 0, 0.5, 1, 10**8])