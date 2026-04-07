def grade(prediction: str, ground_truth: str, threat_type: str) -> float:
    if prediction == ground_truth:
        return 1.0
    if prediction == "approve" and ground_truth == "flag":
        return 0.0
    if prediction == "approve" and ground_truth == "block":
        return -0.5
    if prediction == "flag" and ground_truth == "block":
        return 0.3
    if prediction == "block" and ground_truth == "flag":
        return 0.3
    if prediction == "block" and ground_truth == "approve":
        return -0.3
    if prediction == "flag" and ground_truth == "approve":
        return -0.3
    return 0.0