import os
import requests
import rarfile
import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def load_azure_data(url="https://github.com/Azure/AzurePublicDataset/raw/master/data/AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar", output_dir="data"):
    filename = os.path.basename(url)
    os.makedirs(output_dir, exist_ok=True)

    # if not already downloaded, then download it
    if not Path(filename).exists():
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=4096):
                f.write(chunk)

        print("Finished downloading...")
    else:
        print("Loading cached data...")

    # extract whatever files to data/
    if rarfile.is_rarfile(filename):
        with rarfile.RarFile(filename) as rf:
            rf.extractall(path=output_dir)
            print(f"Extracted files to: {output_dir}")
    else:
        print(f"Error: '{filename}' is not a valid RAR file.")

def preprocess_azure_data(azure_data_path = "data/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt", num_samples = 100, bin_size = 1) -> pd.DataFrame:
    print("Starting dataset processing...")
    start_time = time.monotonic()
    # read the dataset and create new start timestamp (more useful for us)
    df = pd.read_csv(azure_data_path)
    df["start_timestamp"] = df["end_timestamp"] - df["duration"]
    df = df.sort_values(by="start_timestamp", ascending=True)

    # extract just a single function, in this case we will go with the most frequent (can change in the future)
    single_function_df = df.loc[df["func"] == df["func"].mode()[0]].copy()
    single_function_df["end_timestamp"] = single_function_df['end_timestamp'] - single_function_df['start_timestamp'].iloc[0]
    # make the start timestamp 
    single_function_df["start_timestamp"] = single_function_df['start_timestamp'] - single_function_df['start_timestamp'].iloc[0]
    single_function_df = single_function_df.head(num_samples)

    print("Extracted function...")

    # define some reference time based on current
    reference_time = pd.to_datetime(datetime.now() + timedelta(0, 1)) # add 10 seconds just because idk

    # gather invocation counts by bucket size (in seconds)
    bins = range(0, int(single_function_df["start_timestamp"].max()) + bin_size, bin_size)
    single_function_df["time_bucket"] = pd.cut(single_function_df["start_timestamp"], bins=bins, right=False)
    bucket_counts = single_function_df['time_bucket'].value_counts().sort_index()

    print("Extracted invocation counts...")

    # convert to proper timestamps for Prophet to use (and us, for comparisons to current time)
    train_df = bucket_counts.reset_index()
    train_df["time_bucket"] = reference_time + (pd.to_timedelta(train_df.index * bin_size, unit="s") / 10)
    train_df = train_df.rename(columns={"time_bucket": "ds", "count": "y"})

    end_time = time.monotonic()
    print(f"Done processing (took {(end_time - start_time):.2f} seconds)")

    return train_df


if __name__ == "__main__":
    # load_azure_data()
    df = preprocess_azure_data()
    print(df.head())
