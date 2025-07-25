import pandas as pd
# =======================
# Research Questions
# -----------------------
RQ1 = "Which model types and which model sizes provide the most accurate commonsense knowledge?"
RQ2 = "How does the requested measurement unit affect the quality of the results?"
RQ3 = "Does the context vector overlap within models?"
RQ4 = "Does the context vector overlap between models?"

# =======================
# Research Questions
# -----------------------

def analyze_rq1(ground_truth_path, models=None):
    # Accuracy is measured against the ground truth.
    ground_truth = pd.read_csv(ground_truth_path)
    
    # Get sizes from models

    # Get sizes from groundtruth

    # Statistical analysis of averages

    # Statistical analysis of ranges
    

# =======================
# Main Program
# -----------------------
if __name__ == "__main__":
    pass