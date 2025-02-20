import os
import requests
import rarfile
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

if __name__ == "__main__":
    load_azure_data()
