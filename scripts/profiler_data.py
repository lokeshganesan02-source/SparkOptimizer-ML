import os
from pathlib import Path

def profile_input_data(input_path: str) -> dict:
    """
    Profile dataset characteristics before Spark execution.
    """

    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"{input_path} does not exist")

    features = {}

    # Detect format
    if path.is_file():
        file_format = path.suffix.replace(".", "")
        file_count = 1
        total_size = path.stat().st_size
    else:
        files = list(path.rglob("*"))
        files = [f for f in files if f.is_file()]
        file_count = len(files)
        total_size = sum(f.stat().st_size for f in files)
        file_format = files[0].suffix.replace(".", "") if files else "unknown"

    # Size calculations
    size_mb = total_size / (1024 * 1024)
    size_gb = total_size / (1024 * 1024 * 1024)

    # Estimate partitions (128MB heuristic)
    estimated_partitions = max(1, int(size_mb / 128))

    features["input_size_mb"] = round(size_mb, 2)
    features["input_size_gb"] = round(size_gb, 3)
    features["file_count"] = file_count
    features["file_format"] = file_format
    features["estimated_partitions"] = estimated_partitions

    return features


# ----------------------------
# Test block
# ----------------------------
if __name__ == "__main__":

    test_path = "C:/Aravindh/data/emp_skewed.csv"

    result = profile_input_data(test_path)

    print("DATA PROFILE:")
    for k, v in result.items():
        print(f"{k}: {v}")