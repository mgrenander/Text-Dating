import hathitrust_api.data_api as da
import os
import sys
import math
import json
import time
from requests.exceptions import HTTPError, RequestException


def create_folders(base):
    """Creates folders for the texts.
    Each folder is in the format YYYY-XXXX, where XXXX is YYYY+25 years"""
    for i in range(1650, 1925, 25):
        foldername = base + "/{}-{}".format(i, i + 25)
        if not os.path.exists(foldername):
            os.makedirs(foldername)

    # Create N/A category
    if not os.path.exists(base + "/NA"):
        os.makedirs(base + "/NA")


def getnumpages(data_api, id):
    """Downloads meta and return the number of pages in the book"""
    while True:
        try:
            meta = data_api.getmeta(id, json=True)
            break
        except HTTPError:
            print("Request for meta for book failed. Trying again.")
            time.sleep(5)
            pass

    meta_parsed = json.loads(meta)
    return int(meta_parsed["htd:numpages"])


def download_book(data_api, id, book_date_id):
    """Download book with data api"""

    # Format the line properly to work with the Data API
    hathi_id, date = book_date_id.split("\t")
    hathi_id = hathi_id.replace("+=", ":/").replace("=", "/")

    # Find folder to save text
    folder = "Texts/"
    if date == "N/A":
        folder += "NA/" + str(id)
    else:
        rounded = math.floor(int(date) / 25) * 25
        folder += str(rounded) + "-" + str(rounded+25) + "/" + str(id)

    # Create folder if we have not already
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Get number of pages in the book
    numpages = getnumpages(data_api, hathi_id)

    # Check if we already downloaded all pages of this book
    if os.path.exists(folder + "/{}.txt".format(numpages)) or os.path.exists(folder + "/parsed.txt"):
        print("Already downloaded book " + str(id) + " in folder " + folder)
        return

    # Download book, page by page
    print("Now downloading book " + str(id) + " in folder " + folder)
    print("---> Pages downloaded (Total={}): ".format(numpages))
    for i in range(1, numpages + 1):
        filename = folder + "/{}.txt".format(i)

        # If file already exists we don't re-download
        if os.path.exists(filename):
            continue

        attempts = 0
        while True:
            try:
                ocrpage = data_api.getpageocr(hathi_id, i)
                break
            except (HTTPError, RequestException) as e:
                # We attempt to download the page a max of 10 times, then exit with error if we are still failing
                # Note there are lots of issues with the HathiTrust rejecting requests after a few page downloads
                attempts += 1
                time.sleep(5)  # Wait a few seconds before trying again
                if attempts <= 20:
                    print("Failed to download page " + str(i) + " with " + str(attempts) + " tries. Trying again.")
                    pass
                else:
                    return
                    # raise HTTPError("Failed to download book " + str(id) + " at page " + str(i)) from e
        f = open(filename, 'wb')
        f.write(ocrpage)
        f.close()

        print(str(i), end=' ', flush=True)
    print()

# ------------- Main script begins --------------- #
# Create folders if they have not been already
create_folders("Texts")
create_folders("Processed")
create_folders("Combined")
create_folders("Samples")

# Create data_api instance to create connection to HathiTrust
oauth_key = sys.argv[1]
oauth_secret_key = sys.argv[2]
data_api = da.DataAPI(oauth_key, oauth_secret_key)

# Download the books according to Bamman et al. (2017)
ids = open("stratified.txt").read().split("\n")[1:]
for i in range(0, len(ids)):
    if i == 19 or i == 31 or i == 50:
        continue

    download_book(data_api, i, ids[i])
