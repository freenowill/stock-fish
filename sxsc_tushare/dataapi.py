import os
import requests
import pandas as pd
import simplejson as json


class DataApi:
    __http_url = 'http://221.204.19.233:7172'

    def __init__(self, token):
        self.__token = token

    def query(self, api_name, **kwargs):
        params = {'api_name': api_name, 'token': self.__token, 'params': kwargs, 'fields': ''}
        r = requests.post(self.__http_url, data=json.dumps(params))
        result = r.json()
        if result.get('code') != 0:
            raise Exception(result.get('msg', 'Unknown error'))
        data = result.get('data', {})
        columns = data.get('fields', [])
        items = data.get('items', [])
        df = pd.DataFrame(items, columns=columns)
        return df

    def __getattr__(self, name):
        def wrapper(**kwargs):
            return self.query(name, **kwargs)
        return wrapper


def get_api(env='prd', token=None):
    if token is None:
        token = os.environ.get('TUSHARE', '')
    if env == 'prd':
        DataApi.__http_url = 'http://221.204.19.233:7172'
    else:
        DataApi.__http_url = 'http://10.5.42.55:7172'
    return DataApi(token)
