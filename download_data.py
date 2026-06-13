import urllib.request  # built-in Python tool to download files from the internet
import os              # built-in Python tool to interact with the file system (create folders, etc.)

# List of ECB annual report PDFs to download
# Each entry is a pair: (the URL where the file lives, the name we want to save it as)
# All URLs verified directly from ecb.europa.eu
urls = [
    ("https://www.ecb.europa.eu/pub/pdf/annrep/ecb.ar2025~b7f898b33d.en.pdf", "ecb_2025.pdf"),
    ("https://www.ecb.europa.eu/pub/pdf/annrep/ecb.ar2024~8402d8191f.en.pdf", "ecb_2024.pdf"),
    ("https://www.ecb.europa.eu/pub/pdf/annrep/ecb.ar2023~d033c21ac2.en.pdf", "ecb_2023.pdf"),
    ("https://www.ecb.europa.eu/pub/pdf/annrep/ecb.ar2022~8ae51d163b.en.pdf", "ecb_2022.pdf"),
    ("https://www.ecb.europa.eu/pub/pdf/annrep/ecb.ar2021~14d7439b2d.en.pdf", "ecb_2021.pdf"),
]

# Create the folder data/raw/ if it doesn't already exist
# This is where all the downloaded PDFs will be stored
os.makedirs("data/raw", exist_ok=True)

# Loop through each (url, filename) pair and download it one by one
for url, filename in urls:
    path = f"data/raw/{filename}"  # full path where the file will be saved
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, path)  # actually downloads the file from the URL and saves it
    print(f"Saved to {path}")

print("\nAll done!")
