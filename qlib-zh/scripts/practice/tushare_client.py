#!/usr/bin/env python3
"""Minimal Tushare Pro HTTP client used by practice scripts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import request

import pandas as pd


@dataclass
class TushareClient:
    token: str
    base_url: str = "https://api.tushare.pro"
    timeout: int = 30

    def query(self, api_name: str, fields: str = "", **params) -> pd.DataFrame:
        payload = {
            "api_name": api_name,
            "token": self.token,
            "params": params,
            "fields": fields,
        }
        req = request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("code", -1) != 0:
            raise RuntimeError(f"Tushare API {api_name} 失败: {data.get('msg', data)}")
        result = data.get("data") or {}
        items = result.get("items") or []
        fields_list = result.get("fields") or []
        return pd.DataFrame(items, columns=fields_list)

    def __getattr__(self, name: str):
        def _wrapper(**kwargs):
            fields = kwargs.pop("fields", "")
            return self.query(name, fields=fields, **kwargs)

        return _wrapper