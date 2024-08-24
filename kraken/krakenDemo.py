import time
import os
import requests
import urllib.parse
import hashlib
import hmac
import base64

# Read Kraken API key and secret stored in environment variables
api_url = "https://api.kraken.com"
api_key = os.environ['API_KEY_KRAKEN']
api_sec = os.environ['API_SEC_KRAKEN']

def get_kraken_signature(urlpath, data, secret):

    postdata = urllib.parse.urlencode(data)
    encoded = (str(data['nonce']) + postdata).encode()
    message = urlpath.encode() + hashlib.sha256(encoded).digest()

    mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    sigdigest = base64.b64encode(mac.digest())
    return sigdigest.decode()

# Attaches auth headers and returns results of a POST request
def kraken_request(uri_path, data, api_key, api_sec):
    headers = {}
    headers['API-Key'] = api_key
    # get_kraken_signature() as defined in the 'Authentication' section
    headers['API-Sign'] = get_kraken_signature(uri_path, data, api_sec)
    req = requests.post((api_url + uri_path), headers=headers, data=data)
    return req


# resp = requests.get('https://api.kraken.com/0/public/Spread?pair=BTCUSD')
resp = requests.get('https://api.kraken.com/0/public/OHLC?pair=XBTUSD')
# Construct the request and print the result

# Purchase or sell 0.0001 bitcoin (minimum amount, ~6.5 USD)
# resp = kraken_request('/0/private/AddOrder', {
#     "nonce": str(int(1000*time.time())),
#     "ordertype": "market",
#     "type": "sell",
#     "volume": 0.0001,
#     "pair": "BTCUSD"
# }, api_key, api_sec)

print(resp.json())