#Create PR curve
import numpy as np
import pandas as pd

def precision_recall_curve(model, csv_file, root_dir, iou_threshold=0.25):
    results = []
    for x in np.arange(0.1,0.8,0.1):
        model.score_thresh = x
        test_results = model.evaluate(csv_file=csv_file, root_dir=root_dir, iou_threshold=0.25)
        results.append(pd.DataFrame({"precision":[test_results["box_precision"]],"recall":[test_results["box_recall"]]}))
    results = pd.concat(results)
    axes = results.plot("recall","precision",style='.-')
    return axes
    