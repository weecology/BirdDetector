#Create PR curve
import numpy as np
import pandas as pd

def precision_recall_curve(model, csv_file, root_dir, iou_threshold=0.25):
    results = []
    test_results = model.evaluate(csv_file=csv_file, root_dir=root_dir, iou_threshold=0.25)
    image_predictions = model.predict_file(csv_file=csv_file, root_dir=root_dir)    

    if image_predictions.empty:
        return image_predictions, None
    
    for x in np.arange(0.1,0.8,0.1):
        df = test_results["results"] 
        scored_results = df[df.score > x]
        true_positive = sum(scored_results["match"])        
        recall = true_positive / scored_results.shape[0]
        precision = true_positive / image_predictions.shape[0]
        results.append(pd.DataFrame({"precision":[precision],"threshold":[x],"recall":[recall]}))
    
    results = pd.concat(results)
    results["fscore"] = 2 * ((results["precision"] * results["recall"])/ (results["precision"] + results["recall"]))
    axes = results.plot(x="recall",y="precision",style='.-', color="threshold")
    
    return results, axes
    