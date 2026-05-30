"""
新闻源插件系统

每个新闻源是一个独立插件，可单独启用/禁用。
新增来源只需实现 BaseNewsSource 并注册到 SOURCES 列表。
"""
import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional
from email.utils import parsedate_to_datetime

import requests
from bs4 import BeautifulSoup
from loguru import logger

from market_data.a_stock_provider import NewsItem, GubaPost


# 新闻时间窗口：近 N 天
NEWS_WINDOW_DAYS = 3
_NEWS_CUTOFF = (datetime.now() - timedelta(days=NEWS_WINDOW_DAYS)).strftime('%Y-%m-%d')
_GUBA_CUTOFF = (datetime.now() - timedelta(days=NEWS_WINDOW_DAYS)).strftime('%m-%d')


# ── 抽象基类 ──

class BaseNewsSource(ABC):
    """新闻源插件基类"""
    name: str = "base"
    enabled: bool = True

    @abstractmethod
    def fetch(self, symbol: str) -> List[NewsItem]:
        ...


class BaseGubaSource(ABC):
    """股吧源插件基类"""
    name: str = "base"
    enabled: bool = True

    @abstractmethod
    def fetch(self, symbol: str) -> List[GubaPost]:
        ...


# ── 新浪财经新闻 ──

class SinaNewsSource(BaseNewsSource):
    """新浪财经个股新闻，境外环境常被403"""
    name = "新浪财经"
    enabled = False  # 境外不可用，被HTTP 403

    def fetch(self, symbol: str) -> List[NewsItem]:
        today = datetime.now().strftime('%Y-%m-%d')
        prefix = 'sh' if symbol.strip().zfill(6).startswith(('6', '9')) else 'sz'
        code = symbol.strip().zfill(6)
        url = f'https://vip.stock.finance.sina.com.cn/corp/go.php/vCB_AllNewsStock/symbol/{prefix}{code}.phtml'
        r = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        r.encoding = 'gb2312'
        soup = BeautifulSoup(r.text, 'html.parser')
        datelist = soup.select_one('.datelist ul')
        if not datelist:
            return []
        results = []
        for a_tag in datelist.select('a'):
            title = a_tag.text.strip()
            href = a_tag.get('href', '')
            if not title:
                continue
            publish_time = ''
            prev = a_tag.previous_sibling
            if prev is not None and prev.name is None:
                text = prev.strip()
                m = re.search(r'(\d{4}-\d{2}-\d{2}\s*\d{2}:\d{2})', text)
                if m:
                    publish_time = m.group(1)
            if publish_time[:10] < _NEWS_CUTOFF:
                continue
            results.append(NewsItem(
                title=title, url=href, publish_time=publish_time, source=self.name,
            ))
        return results


# ── 东方财富股吧 ──

class EastMoneyGubaSource(BaseGubaSource):
    """东方财富股吧，当日 80+ 帖子"""
    name = "东方财富股吧"

    def fetch(self, symbol: str) -> List[GubaPost]:
        today = datetime.now().strftime('%m-%d')
        code = symbol.strip().zfill(6)
        url = f'https://guba.eastmoney.com/list,{code},f_1.html'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
            'Accept-Language': 'zh-CN,zh;q=0.9',
        }
        r = requests.get(url, timeout=15, headers=headers)
        soup = BeautifulSoup(r.text, 'html.parser')
        results = []
        for item in soup.select('.listitem'):
            try:
                title_div = item.select_one('.title a, a[href*="news"]')
                if not title_div:
                    continue
                title = title_div.text.strip()
                author = (item.select_one('.author') or type('', (), {'text': type('', (), {'strip': lambda: ''})})()).text.strip()
                read_div = item.select_one('.read')
                read_count = int(read_div.text.strip()) if read_div and read_div.text.strip().isdigit() else 0
                reply_div = item.select_one('.reply')
                comment_count = int(reply_div.text.strip()) if reply_div and reply_div.text.strip().isdigit() else 0
                update_div = item.select_one('.update')
                publish_time = update_div.text.strip() if update_div else ''
                if publish_time[:5] < _GUBA_CUTOFF:
                    continue
                results.append(GubaPost(
                    title=title, author=author, publish_time=publish_time,
                    read_count=read_count, comment_count=comment_count,
                ))
            except Exception:
                continue
        return results


# ── Yahoo Finance RSS ──

class YahooFinanceNewsSource(BaseNewsSource):
    """Yahoo Finance RSS，英文新闻，A股覆盖稀疏"""
    name = "Yahoo Finance"
    enabled = True

    def fetch(self, symbol: str) -> List[NewsItem]:
        code = symbol.strip().zfill(6)
        suffix = 'SS' if code.startswith(('6', '9')) else 'SZ'
        url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={code}.{suffix}&region=CN&lang=zh-CN'
        r = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(r.text, 'xml')
        results = []
        for item in soup.find_all('item'):
            title = item.title.text.strip() if item.title else ''
            link = item.link.text.strip() if item.link else ''
            pubdate_str = item.pubDate.text if item.pubDate else ''
            try:
                pubdate = parsedate_to_datetime(pubdate_str).strftime('%Y-%m-%d %H:%M')
            except Exception:
                pubdate = ''
            if not title or pubdate[:10] < _NEWS_CUTOFF:
                continue
            results.append(NewsItem(title=title, url=link, publish_time=pubdate, source=self.name))
        return results


# ── 雪球热度 ──

class XueqiuPopularitySource(BaseNewsSource):
    """雪球关注热度，通过 akshare stock_hot_follow_xq，数据缓存"""
    name = "雪球"
    _cache = None

    def fetch(self, symbol: str) -> List[NewsItem]:
        try:
            import akshare as ak
            import io, sys as _sys, warnings
            if XueqiuPopularitySource._cache is not None:
                df = XueqiuPopularitySource._cache
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    old = _sys.stderr
                    _sys.stderr = io.StringIO()
                    try:
                        df = ak.stock_hot_follow_xq()
                    finally:
                        _sys.stderr = old
                XueqiuPopularitySource._cache = df
            if df is None or df.empty:
                return []
            code = symbol.strip().zfill(6)
            prefix = 'SH' if code.startswith(('6', '9')) else 'SZ'
            match = df[df['股票代码'] == f'{prefix}{code}']
            if match.empty:
                return []
            row = match.iloc[0]
            today = datetime.now().strftime('%Y-%m-%d')
            return [NewsItem(
                title=f"雪球关注 {float(row['关注']):.0f}人  |  最新价 ¥{row.get('最新价', '?')}",
                source=self.name, publish_time=today,
            )]
        except Exception:
            return []


# ── NewsNow 聚合 API（财联社+雪球+华尔街见闻）──

class NewsNowSource(BaseNewsSource):
    """newsnow.busiyi.world 聚合 API，一次请求覆盖 3 个财经源"""
    name = "NewsNow聚合"
    _session = None
    _stock_names: dict = {}  # symbol → name 缓存

    SOURCES = ['cls-hot', 'xueqiu', 'wallstreetcn']

    @classmethod
    def _get_session(cls):
        if cls._session is None:
            cls._session = requests.Session()
            cls._session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Accept-Language': 'zh-CN,zh;q=0.9',
                'Referer': 'https://newsnow.busiyi.world/',
            })
            cls._session.get('https://newsnow.busiyi.world/', timeout=10)
        return cls._session

    def fetch(self, symbol: str) -> List[NewsItem]:
        try:
            s = self._get_session()
            code = symbol.strip().zfill(6)
            today = datetime.now().strftime('%Y-%m-%d')
            results = []
            for src in self.SOURCES:
                try:
                    r = s.get(f'https://newsnow.busiyi.world/api/s?id={src}&latest', timeout=10)
                    if r.status_code != 200:
                        continue
                    data = r.json()
                    for item in data.get('items', []):
                        title = item.get('title', '')
                        if not title:
                            continue
                        # 标题中包含股票代码或简称才收录
                        if code in title or self._match_stock_name(symbol, title):
                            results.append(NewsItem(
                                title=title,
                                url=item.get('url', ''),
                                publish_time=today,
                                source=f'NewsNow-{src}',
                            ))
                except Exception:
                    continue
            return results
        except Exception:
            return []

    @classmethod
    def _match_stock_name(cls, symbol: str, title: str) -> bool:
        """检查标题是否提及该股票简称"""
        if symbol not in cls._stock_names:
            return False
        name = cls._stock_names[symbol]
        return name in title and len(name) >= 2

    @classmethod
    def set_stock_name(cls, symbol: str, name: str):
        """外部注入股票名称，用于标题匹配"""
        cls._stock_names[symbol] = name


# ── 财联社（预留，API 当前不可用）──

class CLSNewsSource(BaseNewsSource):
    """财联社电报，需 Referer 绕过。当前不可用，保留接口。"""
    name = "财联社"
    enabled = False

    def fetch(self, symbol: str) -> List[NewsItem]:
        # TODO: 找到可用 API endpoint 后实现
        return []


# ── 注册表 ──

NEWS_SOURCES: List[BaseNewsSource] = [
    SinaNewsSource(),
    NewsNowSource(),             # 财联社+雪球+华尔街见闻聚合
    YahooFinanceNewsSource(),
    XueqiuPopularitySource(),
    CLSNewsSource(),             # 待 API 可用后启用
]

GUBA_SOURCES: List[BaseGubaSource] = [
    EastMoneyGubaSource(),
]
