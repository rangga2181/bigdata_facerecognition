import pandas as pd
from pathlib import Path

log_path = Path("d:/BIG-Data/tuber/checkpoints/logs/training_log.csv")

if log_path.exists():
    log_df = pd.read_csv(log_path)
    print("CSV loaded successfully! Shape:", log_df.shape)
    print("Columns:", list(log_df.columns))
    # Filter log_df hanya untuk epoch run terakhir jika ada penumpukan log history
    run_starts = log_df[log_df["epoch"] == 1].index
    if len(run_starts) > 0:
        log_df = log_df.iloc[run_starts[-1]:].reset_index(drop=True)
    print("Filtered log_df (last run) shape:", log_df.shape)
    print(log_df.head())
else:
    print("Log file not found.")
