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

    def test_iou_simple_overlap(self):
        r1 = BoundingBox('', '', (0,0), (1,1))
        r2 = BoundingBox('', '', (0.5,0.5), (1.5,1.5))

        self.assertEqual(r1.iou(r2), 0.25 / 1.75)
        self.assertEqual(r1.iou(r2), r2.iou(r1))

    def test_precision_one_class(self):
        # [2 2 10 20; 80 80 30 40]
        true1 = BoundingBox(1, 'A', (2,2), (12,22))
        true2 = BoundingBox(1, 'A', (80,80), (110,120))

        labels = [true1, true2]

        # [4 4 10 20; 50 50 30 10; 90 90 40 50];
        pred1 = BoundingBox(1, 'A', (4,4), (14,24))
        pred2 = BoundingBox(1, 'A', (50,50), (80,60))
        pred3 = BoundingBox(1, 'A', (80,80), (110,120))

        predictions = [pred1, pred2, pred3]

        det = Detection(labels=labels, predictions=predictions)

        pr,re,_ = det.metrics()

        self.assertEqual(pr, 2.0 / 3.0)
        self.assertEqual(re, 1.000)

    def test_precision_false_neg(self):
        # [2 2 10 20; 80 80 30 40]
        true1 = BoundingBox(1, 'A', (2,2), (12,22))
        true2 = BoundingBox(1, 'A', (80,80), (110,120))
        true3 = BoundingBox(1, 'A', (20,20),(30,30))

        labels = [true1, true2, true3]

        # [4 4 10 20; 50 50 30 10; 90 90 40 50];
        pred1 = BoundingBox(1, 'A', (4,4), (14,24))
        pred2 = BoundingBox(1, 'A', (50,50), (80,60))
        pred3 = BoundingBox(1, 'A', (80,80), (110,120))

        predictions = [pred1, pred2, pred3]

        det = Detection(labels=labels, predictions=predictions)

        pr,re,_ = det.metrics()

        self.assertEqual(pr, 2.0 / 3.0)
        self.assertEqual(re, 2.0 / 3.0)

    def test_precision_bad_predicts(self):
        # [2 2 10 20; 80 80 30 40]
        true1 = BoundingBox(1, 'A', (2,2), (12,22))
        true2 = BoundingBox(1, 'A', (80,80), (110,120))
        true3 = BoundingBox(1, 'A', (20,20),(30,30))

        labels = [true1, true2, true3]

        # [4 4 10 20; 50 50 30 10; 90 90 40 50];
        pred1 = BoundingBox(1, 'A', (4,4), (14,24))
        pred2 = BoundingBox(1, 'A', (50,50), (80,60))
        pred3 = BoundingBox(1, 'B', (80,80), (110,120))

        predictions = [pred1, pred2, pred3]

        det = Detection(labels=labels, predictions=predictions)

        pr,re,_ = det.metrics()

        self.assertEqual(pr, 1.0 / 3.0)
        self.assertEqual(re, 1.0 / 3.0)



if __name__ == '__main__':
    unittest.main()