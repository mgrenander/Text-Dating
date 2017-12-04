import hathitrust_api.data_api as da
import os
import sys
import math


def create_folders():
    """Creates folders for the texts.
    Each folder is in the format YYYY-XXXX, where XXXX is YYYY+25 years"""
    for i in range(1750, 1925, 25):
        foldername = "Texts/{}-{}".format(i, i + 25)
        if not os.path.exists(foldername):
            os.makedirs(foldername)

    # Create N/A category
    if not os.path.exists("Texts/NA"):
        os.makedirs("Texts/NA")


def download_book(data_api, id_date):
    """Download book with data api"""

    # Format the line properly to work with the Data API
    id, date = id_date.split("\t")
    id = id.replace("+=", ":/").replace("=", "/")

    # Find folder to save text
    folder = "Texts/"
    if date == "N/A":
        folder += "NA"
    else:
        rounded = math.floor(int(date) / 25) * 25
        folder += str(rounded) + "-" + str(rounded+25) + "/"

    # Download book
    ocrtext = data_api.getdocumentocr(id)

    # Save book to folder
    newest = 0
    if os.listdir(folder) != []:
        newest = max([int(f) for f in os.listdir(folder)]) + 1
    os.makedirs(folder + str(newest))

    for i in range(0, len(ocrtext)):
        filename = folder + "{}/{}.txt".format(newest, i)
        f = open(filename, 'wb')
        f.write(ocrtext[i])
        f.close()

# Create folders if they have not been already
create_folders()

# Create data_api instance to create connection to HathiTrust
oauth_key = sys.argv[1]
oauth_secret_key = sys.argv[2]
data_api = da.DataAPI(oauth_key, oauth_secret_key)

# Download the books according to Bamman et al. (2017)
ids = open("stratified.txt").read().split("\n")[1:2]
for id_date in ids:
    download_book(data_api, id_date)
