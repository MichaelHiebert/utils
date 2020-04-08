import unittest
from detection import Detection,BoundingBox

class TestDetection(unittest.TestCase):

    def test_intersection_no_overlap(self):
        r1 = BoundingBox('','',(0,0), (1,1))
        r2 = BoundingBox('','',(1,1), (2,2))

        self.assertEqual(r1._intersection(r2), 0.0)

    def test_intersection_total_overlap(self):
        r1 = BoundingBox('','',(0,0), (1,1))
        r2 = BoundingBox('','',(0,0), (1,1))

        self.assertEqual(r1._intersection(r2), 1.0)

    def test_intersection_half_overlap(self):
        r1 = BoundingBox('','',(0,0), (1,1))
        r2 = BoundingBox('','',(0,0.5), (1,1))

        self.assertEqual(r1._intersection(r2), 0.5)

    def test_intersection_random_overlap(self):
        r1 = BoundingBox('','',(0,1), (5,3))
        r2 = BoundingBox('','',(4.5,0), (5.5,6))

        self.assertEqual(r1._intersection(r2), 1.0)

    def test_intersection_overlap_over_one(self):
        r1 = BoundingBox('','',(0,1), (5,5))
        r2 = BoundingBox('','',(4.5,0), (5.5,6))

        self.assertEqual(r1._intersection(r2), 2.0)

    def test_union_no_overlap(self):
        r1 = BoundingBox('','',(0,0), (1,1))
        r2 = BoundingBox('','',(1,1), (2,2))

        self.assertEqual(r1._union(r2), 2.0)

    def test_union_total_overlap(self):
        r1 = BoundingBox('','',(0,0), (1,1))
        r2 = BoundingBox('','',(0,0), (1,1))

        self.assertEqual(r1._union(r2), 1.0)

    def test_union_half_overlap(self):
        r1 = BoundingBox('','',(0,0), (1,1))
        r2 = BoundingBox('','',(0,0.5), (1,1))

        self.assertEqual(r1._union(r2), 1.0)

    def test_union_random_overlap(self):
        r1 = BoundingBox('','',(0,1), (5,3))
        r2 = BoundingBox('','',(4.5,0), (5.5,6))

        self.assertEqual(r1._union(r2), 15.0)

    def test_union_overlap_over_one(self):
        r1 = BoundingBox('','',(0,1), (5,5))
        r2 = BoundingBox('','',(4.5,0), (5.5,6))

        self.assertEqual(r1._union(r2), 24.0)

if __name__ == '__main__':
    unittest.main()