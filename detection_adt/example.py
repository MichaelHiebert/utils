from detection import Detection

if __name__ == "__main__":
    det = Detection()

    det.from_csv(label_filepath='example/labels.csv', pred_filepath='example/preds.csv')

    precision,recall,fscore = det.metrics()

    print('Precision: {}\nRecall: {}\nfscore: {}'.format(precision,recall,fscore))