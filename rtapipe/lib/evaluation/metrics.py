from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def computeMetrics(self, realLabels, predLabels):
    prec = precision_score(realLabels, predLabels)
    recall = recall_score(realLabels, predLabels)
    f1 = f1_score(realLabels, predLabels)
    tn, fp, fn, tp = confusion_matrix(realLabels, predLabels).ravel()
    print(tp, fp, tn, fn)
    fpr = fp / (fp+tn)
    fnr = fn / (fn+tp)

    score_str = f"Precision={round(prec,3)}\nRecall={round(recall,3)}\nF1={round(f1, 3)}\nFalse Positive Rate={round(fpr, 3)}\nFalse Negative Rate={round(fnr,3 )}"

    print(score_str)

    outputPath = self.outDir.joinpath("performance_metrics.txt")
    with open(outputPath, "w") as pf:
        pf.write(score_str)
    print(f"File {outputPath} created.")    