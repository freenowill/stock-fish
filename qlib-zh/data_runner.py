"""
Qlib 数据更新执行器 — 从 GitHub 下载最新的 qlib_bin.tar.gz 并解压到 ~/.qlib/qlib_data/cn_data。

用法:
    from data_runner import run_data_update
    result = run_data_update(progress_callback=cb)
    print(result["message"])
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Callable

import requests

# ---- 常量 ----
GITHUB_REPO = "chenditc/investment_data"
RELEASE_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
ASSET_NAME = "qlib_bin.tar.gz"
QLIB_DATA_DIR = Path.home() / ".qlib" / "qlib_data" / "cn_data"

# 下载超时（秒）
DOWNLOAD_TIMEOUT = 600
# 每下载 chunk 后推送间隔（秒）
PROGRESS_INTERVAL = 1.0

# ---- 代理配置 ----
_PROXY_URL = os.environ.get("QLIB_DATA_PROXY", "")
_PROXIES = {"http": _PROXY_URL, "https": _PROXY_URL} if _PROXY_URL else None


def _make_session() -> requests.Session:
    """创建带代理配置的 requests Session."""
    session = requests.Session()
    if _PROXIES:
        session.proxies.update(_PROXIES)
    return session


def _log(progress_callback: Callable | None, message: str, **extra):
    """统一日志推送."""
    if progress_callback:
        progress_callback({"message": message, **extra})


def _get_latest_release_info() -> dict:
    """获取最新 release 的下载 URL 和 tag."""
    session = _make_session()
    resp = session.get(
        RELEASE_API_URL,
        headers={"User-Agent": "StockFish/1.0"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    tag = data.get("tag_name", "unknown")
    for asset in data.get("assets", []):
        if asset.get("name") == ASSET_NAME:
            return {
                # 使用 API asset URL（需 Accept: application/octet-stream header）
                "download_url": asset["url"],
                "tag": tag,
                "size": asset.get("size", 0),
            }

    raise FileNotFoundError(
        f"在 {GITHUB_REPO} 的最新 release ({tag}) 中未找到 {ASSET_NAME}"
    )


def _download_file(
    url: str,
    dest_path: Path,
    total_size: int = 0,
    progress_callback: Callable | None = None,
) -> None:
    """流式下载文件并推送进度."""
    _log(progress_callback, f"开始下载: {url}")
    _log(progress_callback, f"目标文件: {dest_path}")
    if total_size > 0:
        _log(progress_callback, f"文件大小: {total_size / 1024 / 1024:.1f} MB")

    session = _make_session()
    resp = session.get(
        url,
        headers={
            "Accept": "application/octet-stream",
            "User-Agent": "StockFish/1.0",
        },
        stream=True,
        timeout=DOWNLOAD_TIMEOUT,
    )
    resp.raise_for_status()

    # 实际 content-length
    content_length = int(resp.headers.get("content-length", 0))
    if content_length == 0:
        content_length = total_size

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    last_log_time = 0.0
    import time as _time

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                now = _time.time()
                if content_length > 0 and (now - last_log_time) >= PROGRESS_INTERVAL:
                    pct = downloaded / content_length * 100
                    mb_dl = downloaded / 1024 / 1024
                    mb_total = content_length / 1024 / 1024
                    _log(
                        progress_callback,
                        f"下载中... {mb_dl:.1f}/{mb_total:.1f} MB ({pct:.1f}%)",
                        progress=round(pct / 100, 4),
                    )
                    last_log_time = now

    _log(progress_callback, f"下载完成: {downloaded / 1024 / 1024:.1f} MB")


def _extract_tar_gz(
    archive_path: Path,
    extract_to: Path,
    progress_callback: Callable | None = None,
) -> None:
    """解压 tar.gz 到目标目录."""
    _log(progress_callback, f"开始解压: {archive_path}")
    _log(progress_callback, f"解压到: {extract_to}")

    extract_to.mkdir(parents=True, exist_ok=True)

    # 先清空目标目录（保留目录本身）
    for item in extract_to.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    # 解压
    import time as _time

    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        total_members = len(members)
        _log(progress_callback, f"共 {total_members} 个文件/目录")

        # 找到公共前缀以 strip-components=1
        # qlib_bin.tar.gz 内部结构通常为 cn_data/xxx
        common_prefix = None
        for m in members:
            name = m.name
            if "/" in name:
                prefix = name.split("/")[0]
                if common_prefix is None:
                    common_prefix = prefix
                elif prefix != common_prefix:
                    common_prefix = None
                    break
            else:
                common_prefix = None
                break

        # strip-components=1
        last_log_time = 0.0
        for i, member in enumerate(members):
            # 调整路径：去掉第一级目录
            if common_prefix and member.name.startswith(common_prefix + "/"):
                member.name = member.name[len(common_prefix) + 1:]
            elif common_prefix and member.name == common_prefix:
                # 跳过顶级目录本身
                continue

            if member.name:  # 空路径跳过
                tar.extract(member, path=extract_to)

            now = _time.time()
            if (now - last_log_time) >= PROGRESS_INTERVAL:
                pct = (i + 1) / total_members * 100
                _log(
                    progress_callback,
                    f"解压中... {i + 1}/{total_members} ({pct:.0f}%)",
                    progress=round(0.5 + pct / 200, 4),  # 进度映射 0.5~1.0
                )
                last_log_time = now

    _log(progress_callback, "解压完成，正在整理目录结构...")

    # 修复：如果数据被解压到了一个子目录中（如 qlib_bin/ 或 cn_data/），
    # 把内容移到 extract_to 顶层
    for _ in range(3):  # 最多处理 3 层嵌套
        subdirs = [d for d in extract_to.iterdir() if d.is_dir()]
        files = [f for f in extract_to.iterdir() if f.is_file()]
        # 如果只有一个子目录且没有文件，把子目录内容移到顶层
        if len(subdirs) == 1 and len(files) == 0:
            nested = subdirs[0]
            _log(progress_callback, f"检测到嵌套目录 {nested.name}，上移内容...")
            for item in nested.iterdir():
                target = extract_to / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), str(target))
            nested.rmdir()
        else:
            break
    _log(progress_callback, "目录整理完成")


def _verify_data_dir(progress_callback: Callable | None = None) -> list[str]:
    """验证解压后的数据目录是否完整，返回存在的关键文件列表."""
    checks = [
        QLIB_DATA_DIR / "calendars" / "day.txt",
        QLIB_DATA_DIR / "instruments" / "csi300.txt",
        QLIB_DATA_DIR / "instruments" / "all.txt",
    ]
    found = []
    missing = []
    for p in checks:
        if p.exists():
            found.append(str(p.relative_to(QLIB_DATA_DIR)))
        else:
            missing.append(str(p.relative_to(QLIB_DATA_DIR)))

    _log(progress_callback, f"数据验证: 找到 {len(found)}/{len(checks)} 关键文件")
    if missing:
        _log(progress_callback, f"缺失文件: {', '.join(missing)}")

    return found


def run_data_update(
    progress_callback: Callable | None = None,
) -> dict:
    """
    下载最新 qlib_bin.tar.gz 并解压到 ~/.qlib/qlib_data/cn_data。

    Args:
        progress_callback: 可选回调，接收 dict(status, message, progress, ...)

    Returns:
        {"success": True/False, "message": "...", "tag": "...", "verified_files": [...]}
    """
    import time as _time
    start_time = _time.time()

    try:
        # Step 1: 获取 release 信息
        _log(progress_callback, "正在获取最新 release 信息...", progress=0.0)
        release_info = _get_latest_release_info()
        tag = release_info["tag"]
        download_url = release_info["download_url"]
        size = release_info["size"]

        _log(
            progress_callback,
            f"最新 release: {tag} | {size / 1024 / 1024:.1f} MB",
            progress=0.05,
        )

        # Step 2: 下载
        _log(progress_callback, f"下载链接: {download_url}", progress=0.05)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            _download_file(download_url, tmp_path, total_size=size, progress_callback=progress_callback)

            # Step 3: 解压
            _extract_tar_gz(tmp_path, QLIB_DATA_DIR, progress_callback=progress_callback)
        finally:
            # 清理临时文件
            if tmp_path.exists():
                tmp_path.unlink()
                _log(progress_callback, "已清理临时文件")

        # Step 4: 验证
        verified = _verify_data_dir(progress_callback)

        elapsed = _time.time() - start_time
        msg = (
            f"✅ qlib 数据更新完成！"
            f" 版本: {tag}, 耗时: {elapsed:.0f}s, "
            f"验证通过: {len(verified)} 文件"
        )
        _log(progress_callback, msg, progress=1.0, status="completed")

        return {
            "success": True,
            "message": msg,
            "tag": tag,
            "verified_files": verified,
            "elapsed_seconds": round(elapsed, 1),
        }

    except Exception as e:
        error_msg = f"❌ 数据更新失败: {e}"
        _log(progress_callback, error_msg, status="failed", error=str(e))
        return {
            "success": False,
            "message": error_msg,
            "error": str(e),
        }


def _print_progress(data):
    """CLI 进度回调."""
    msg = data.get("message", "")
    status = data.get("status", "")
    pct = data.get("progress", 0)
    if pct:
        print(f"[{pct*100:.0f}%] {msg}", file=sys.stderr)
    else:
        print(f"[data] {msg}", file=sys.stderr)


# ---- CLI 入口 ----
if __name__ == "__main__":
    result = run_data_update(progress_callback=_print_progress)
    print(json.dumps(result, ensure_ascii=False, indent=2))
