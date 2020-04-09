# Detection ADT

## From CSV

CSV files should be of the form:
```
frame_id, object_label, top_left_x, top_left_y, bottom_right_x, bottom_right_y
```
or
```
frame_id, object_label, top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence
```

Here is an example of loading a `Detection` from a csv file:

```
from detection import Detection

if __name__ == "__main__":
    det = Detection()

    det.from_csv(label_filepath='example/labels.csv', pred_filepath='example/preds.csv')

    precision,recall,fscore = det.metrics()

    print('Precision: {}\nRecall: {}\nfscore: {}'.format(precision,recall,fscore))
```

## Streaming

Alternatively, to connect directly to a `Detection` without dumping data in a CSV, you can use `add_label()` or `add_prediction()`.

Here is an example from one of the test cases:
```
def test_precision_bad_predicts_stream(self):
    det = Detection()

    det.add_label(BoundingBox(1, 'A', (2,2), (12,22)))
    det.add_label(1, 'A', 80, 80, 110, 120)
    det.add_label(BoundingBox(1, 'A', (20,20),(30,30)))

    # [4 4 10 20; 50 50 30 10; 90 90 40 50];
    pred1 = BoundingBox(1, 'A', (4,4), (14,24))
    pred2 = BoundingBox(1, 'A', (50,50), (80,60))
    pred3 = BoundingBox(1, 'B', (80,80), (110,120))

    det.add_prediction(BoundingBox(1, 'A', (4,4), (14,24)))
    det.add_prediction(1, 'A', 50, 50, 80, 60)
    det.add_prediction(1, 'B', 80,80, 110,120)

    pr,re,_ = det.metrics()

    self.assertEqual(pr, 1.0 / 3.0)
    self.assertEqual(re, 1.0 / 3.0)
```