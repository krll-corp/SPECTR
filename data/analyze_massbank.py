import requests
import json

def analyze_massbank_json(url):
    """
    Function to download and analyze MassBank JSON data structure.

    Args:
        url (str): URL to the JSON dataset.

    Returns:
        None: Prints a sample entry for inspection.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        massbank_data = response.json()
        # Extract a sample entry to display the data structure
        if isinstance(massbank_data, list):
            sample_entry = massbank_data[100]
        else:
            sample_entry = massbank_data
        print("Sample entry from the MassBank dataset:")
        print(json.dumps(sample_entry, indent=2))
    except Exception as e:
        print(f"An error occurred while analyzing MassBank JSON: {e}")

# Analyze MassBank JSON data structure
dataset_url = "https://github.com/MassBank/MassBank-data/releases/download/2024.11/MassBank.json"
analyze_massbank_json(dataset_url)
