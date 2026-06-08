#!/usr/bin/env python3
"""Small wrapper around gm.api for the practice pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


def _ensure_gm_imports():
    try:
        from gm.api import (
            get_fundamentals_n,
            get_history_constituents,
            get_history_instruments,
            get_instruments,
            history,
            set_serv_addr,
            set_token,
        )
    except ImportError as exc:
        raise RuntimeError(
            "当前运行环境未安装掘金量化 gm SDK，且无法从 PyPI 获取。请在可用的 gm 环境中运行。"
        ) from exc

    return {
        "get_fundamentals_n": get_fundamentals_n,
        "get_history_constituents": get_history_constituents,
        "get_history_instruments": get_history_instruments,
        "get_instruments": get_instruments,
        "history": history,
        "set_serv_addr": set_serv_addr,
        "set_token": set_token,
    }


@dataclass
class GmClient:
    token: str | None = None

    def __post_init__(self):
        self.token = (self.token or os.getenv("GM_TOKEN", "")).strip()
        if not self.token:
            raise RuntimeError("缺少 GM_TOKEN，请先 export GM_TOKEN=... 后再运行")
        funcs = _ensure_gm_imports()
        serv_addr = os.getenv("GM_SERV_ADDR", "").strip()
        if serv_addr:
            funcs["set_serv_addr"](serv_addr)
        funcs["set_token"](self.token)
        self._funcs = funcs

    @staticmethod
    def _as_dataframe(obj) -> pd.DataFrame:
        if obj is None:
            return pd.DataFrame()
        if isinstance(obj, pd.DataFrame):
            return obj.copy()
        return pd.DataFrame(obj)

    @staticmethod
    def _join_symbols(symbols: Iterable[str] | str) -> str:
        if isinstance(symbols, str):
            return symbols
        return ",".join(list(symbols))

    def history(self, symbols, start_time, end_time, fields, frequency="1d") -> pd.DataFrame:
        history = self._funcs["history"]
        joined = self._join_symbols(symbols)
        for key in ("symbols", "symbol"):
            try:
                df = history(
                    **{
                        key: joined,
                        "frequency": frequency,
                        "start_time": start_time,
                        "end_time": end_time,
                        "fields": fields,
                        "df": True,
                    }
                )
                return self._as_dataframe(df)
            except TypeError:
                continue
        raise RuntimeError("gm history 调用失败")

    def get_instruments(self, symbols, fields="") -> pd.DataFrame:
        get_instruments = self._funcs["get_instruments"]
        joined = self._join_symbols(symbols)
        if not joined:
            try:
                df = get_instruments(fields=fields, df=True)
                return self._as_dataframe(df)
            except TypeError:
                pass
        for key in ("symbols", "symbol"):
            try:
                df = get_instruments(**{key: joined, "fields": fields, "df": True})
                return self._as_dataframe(df)
            except TypeError:
                continue
        raise RuntimeError("gm get_instruments 调用失败")

    def get_history_instruments(self, symbols, start_date, end_date, fields="") -> pd.DataFrame:
        func = self._funcs["get_history_instruments"]
        joined = self._join_symbols(symbols)
        for key in ("symbols", "symbol"):
            try:
                df = func(
                    **{
                        key: joined,
                        "start_date": start_date,
                        "end_date": end_date,
                        "fields": fields,
                        "df": True,
                    }
                )
                return self._as_dataframe(df)
            except TypeError:
                continue
        raise RuntimeError("gm get_history_instruments 调用失败")

    def get_history_constituents(self, index, start_date, end_date) -> pd.DataFrame:
        func = self._funcs["get_history_constituents"]
        for key in ("index", "symbol"):
            try:
                df = func(**{key: index, "start_date": start_date, "end_date": end_date})
                return self._as_dataframe(df)
            except TypeError:
                continue
        raise RuntimeError("gm get_history_constituents 调用失败")

    def get_fundamentals_n(self, table, symbols, end_date, count=1, fields="") -> pd.DataFrame:
        func = self._funcs["get_fundamentals_n"]
        joined = self._join_symbols(symbols)
        for key in ("symbols", "symbol"):
            try:
                df = func(
                    **{
                        key: joined,
                        "table": table,
                        "end_date": end_date,
                        "count": count,
                        "fields": fields,
                        "df": True,
                    }
                )
                return self._as_dataframe(df)
            except TypeError:
                continue
        raise RuntimeError("gm get_fundamentals_n 调用失败")