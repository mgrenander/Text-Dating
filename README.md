# Text Dating
Estimating the date of publication for texts from 1625 - 1925 using convolutional neural networks. Created for a Natural Language Processing (COMP 550) class at McGill University.

## Running the Book Downloader
The preprocessor relies on hathitrust-api, a python wrapper for the HathiTrust API. It is available here:

https://github.com/rlmv/hathitrust-api

Note that this API is broken for Python 3.6 and requires minor changes. If you are looking to run this code and require help setting this up, contact us.

Note also that the API relies on requests and requests-oauthlib packages. You will also need a HathiTrust Access Key and Secret Key, available here:

https://babel.hathitrust.org/cgi/kgs/request

Usage for the downloader is as follows:

```python
python downloader.py <access_key> <secret_key>
```

This will create the appropriate folders and download books as specified in the `stratified.txt` file.

## Preprocessing

After you have downloaded enough books, run the preprocessor: ```data_preprocessor.py```.
This will remove metadata from the texts and consolidate them into one document per time period, stored in the ```Combined``` folder as ```document.txt```.
Lastly, run the sample creator: ```sample_creator.py```. This script will tranform the raw text data supplied by the ```document.txt``` files into the one-hot encoding matrices required by the convolutional neural network. 

Alternatively, run the ```naive_bayes_preprocessor.py``` file to transform the data into a readable format for the Naive Bayes baseline we have implemented.

After running, both the ```sample_creator.py``` and ```naive_bayes_preprocessor.py``` scripts will output pickle files containing the data required for the models. These will need to be placed in the appropriate folders as specified below.

## Naive Bayes Baseline

## Convolutional Network on Google Cloud