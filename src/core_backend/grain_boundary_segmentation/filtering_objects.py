import pandas as pd
import numpy as np

def filter_objects(objects_calculated, rf_model):
    """
    Applies the trained Random Forest model to classify all detected objects.
    Keeps only the objects classified as Class 1 (Surface Grain Boundaries) 
    and filters out subgrains, bottom boundaries, and artifacts.
    """
    print("\n" + "="*50)
    print("--- FILTERING OBJECTS VIA RANDOM FOREST ---")
    print("="*50)

    if not objects_calculated:
        print("[WARNING] No objects provided to filter.")
        return []

    if rf_model is None:
        print("[ERROR] Random Forest model is missing. Cannot filter objects.")
        return []

    # Convert each object's calculated features to a model input row.
    X_data = []
    for obj in objects_calculated:
        X_data.append(obj["values"])

    # Build the feature table expected by scikit-learn.
    df_X = pd.DataFrame(X_data)

    # Match the training feature order before prediction.
    try:
        training_features = rf_model.feature_names_in_
        df_X = df_X[training_features]
    except AttributeError:
        # Older scikit-learn versions may not expose the fitted feature names.
        pass
    except KeyError as e:
        print(f"\n[ERROR] Mismatch between calculated features and model features: {e}")
        print("Please re-run the calibration to train a new model with the current features.")
        return []

    # Predict a class for each feature row.
    print("Predicting classes for all objects...")
    predictions = rf_model.predict(df_X)

    # Attach each prediction to its corresponding object.
    filtered_objects = []
    class_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    for obj, pred in zip(objects_calculated, predictions):
        prediction_int = int(pred)
        
        # Preserve the predicted class for diagnostics and later pipeline stages.
        obj["predicted_label"] = prediction_int
        
        # Count predictions by class for the summary report.
        if prediction_int in class_counts:
            class_counts[prediction_int] += 1
            
        # Pass only surface grain-boundary objects to the next stage.
        if prediction_int == 1:
            filtered_objects.append(obj)

    # Report the classification totals.
    total = len(objects_calculated)
    print(f"\nClassification Summary (Total Objects: {total}):")
    print(f"  -> Kept:      {class_counts[1]} objects (Class 1 - Surface Grain Boundaries)")
    print(f"  -> Discarded: {class_counts[2]} objects (Class 2 - Subgrains)")
    print(f"  -> Discarded: {class_counts[3]} objects (Class 3 - Bottom Boundaries)")
    print(f"  -> Discarded: {class_counts[4]} objects (Class 4 - Artifacts/Noise)")
    
    kept_percentage = (class_counts[1] / total) * 100 if total > 0 else 0
    print(f"\nFiltering complete. Reduced data to {kept_percentage:.1f}% of original objects.")
    print("="*50 + "\n")

    return filtered_objects