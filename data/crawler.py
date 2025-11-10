import requests
from bs4 import BeautifulSoup
import json

def extract_peaks_from_webpage(record_url):
    """
    Extract peak data from a MassBank record webpage.

    Args:
        record_url (str): URL of the MassBank record webpage.

    Returns:
        dict: A dictionary containing the substance name and peak data.
    """
    try:
        print(f"Fetching data from: {record_url}")
        response = requests.get(record_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract substance name
        substance_name = soup.find("h1").get_text(strip=True)
        print(f"Substance name found: {substance_name}")

        # Locate the PK$PEAK section in the page
        peak_section = soup.find("b", string="PK$PEAK:")
        if not peak_section:
            print("PK$PEAK section not found.")
            return {"substance": substance_name, "peaks": None}

        # Extract and parse the peak data
        peak_data = []
        # Locate the data section following PK$PEAK
        peak_container = peak_section.find_next("br")
        while peak_container:
            peak_text = peak_container.next_sibling
            if peak_text:
                print(f"Raw peak data text: {peak_text}")
                lines = peak_text.splitlines()
                for line in lines:
                    print(f"Processing line: {line}")
                    if line.strip() and not line.startswith("PK$") and not line.startswith("//"):
                        parts = line.split()
                        if len(parts) == 3:
                            peak_data.append({"m/z": float(parts[0]), "intensity": int(parts[1]), "relative_intensity": int(parts[2])})
            if "//" in str(peak_container):
                break
            peak_container = peak_container.find_next("br")

        return {"substance": substance_name, "peaks": peak_data}

    except Exception as e:
        print(f"An error occurred while extracting peaks from {record_url}: {e}")
        return {"substance": None, "peaks": None}

def save_to_jsonl(data, file_path):
    """
    Save the extracted data to a JSONL file.

    Args:
        data (list): List of dictionaries containing substance and peak data.
        file_path (str): Path to the JSONL file.
    """
    try:
        with open(file_path, 'w') as file:
            for entry in data:
                file.write(json.dumps(entry) + '\n')
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving to JSONL: {e}")

def main():
    """
    Main function to extract peaks from sample MassBank record webpages and save them to a JSONL file.
    """
    sample_urls = [
        "https://massbank.eu/MassBank/RecordDisplay?id=MSBNK-Waters-WA000361#Dataset",
        "https://massbank.eu/MassBank/RecordDisplay?id=MSBNK-Waters-WA000907#Dataset",
        "https://massbank.eu/MassBank/RecordDisplay?id=MSBNK-Waters-WA002919#Dataset"
    ]

    extracted_data = []
    for url in sample_urls:
        result = extract_peaks_from_webpage(url)
        if result["peaks"]:
            extracted_data.append(result)

    save_to_jsonl(extracted_data, "extracted_peaks.jsonl")

if __name__ == "__main__":
    main()
