# save as: clean_history.py
# run once: python clean_history.py

import pandas as pd

raw = open("C:/Aravindh/data/job_history.csv").readlines()

# Separate old rows (input_size_mb = number) vs new rows (dataset = .csv filename)
new_rows = [raw[0]]  # keep original header temporarily

for line in raw[1:]:
    cols = line.strip().split(",")
    # New rows have emp_100k.csv / emp_500k.csv etc in column 2
    if ".csv" in cols[1]:
        new_rows.append(line)

# Write with CORRECT header
with open("C:/Aravindh/data/job_history.csv", "w") as f:
    f.write("job_name,dataset,shuffle_partitions,executor_memory,execution_time_sec,timestamp\n")
    for line in new_rows[1:]:   # skip old header
        f.write(line)

print(f"✅ Cleaned! Kept {len(new_rows)-1} new rows, discarded old skew_join data.")