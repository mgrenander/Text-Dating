# Text-Dating
Estimating date of publication for novels from 1500 - present

## Running the data preprocessor
The preprocessor relies on hathitrust-api, a python wrapper for the HathiTrust API. It is available here:

https://github.com/rlmv/hathitrust-api

Note that this API is broken for Python 3.6 and requires minor changes. Soon I will fork the API and write the required changes.

Note also that the API relies on requests and requests-oauthlib packages. You will also need a HathiTrust Access Key and Secret Key, available here:

https://babel.hathitrust.org/cgi/kgs/request

Usage for the downloader is as follows:

```python
python downloader.py <access_key> <secret_key>
```

This will create the appropriate folders and download books as specified in the `stratified.txt` file.