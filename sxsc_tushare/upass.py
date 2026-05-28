import os

TOKEN_FILE = os.path.expanduser('~/jsdata_tk.csv')


def get_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            return f.read().strip()
    return os.environ.get('TUSHARE', '')


def set_token(token):
    with open(TOKEN_FILE, 'w') as f:
        f.write(token)
    os.environ['TUSHARE'] = token
