#!/usr/bin/env python3
"""
stage6_final_result.py
最终结果整合 & 调仓建议：
  1. 从 second_screen 读取已选股票
  2. 从 model_predict/scores.csv 查找最新排名，新增"最新排名"列
  3. 保存 result.csv
  4. 若有股票最新排名跌入后50%，则用 second_screen 里排名更高的股票替换
  5. 保存 result_update.csv
输出:
  <output>/result.csv
  <output>/result_update.csv
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

warnings.filterwarnings("ignore")


def _normalize_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"^[A-Za-z]+", "", regex=True).str.zfill(6)


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} 不存在: {path}")
    try:
        return pd.read_csv(
            path,
            dtype={
                "code": str,
                "name": str,
                "instrument": str,
                "symbol": str,
                "pred_date": str,
                "listed_date": str,
                "ipo_date": str,
            },
        )
    except EmptyDataError:
        return pd.DataFrame()


def _resolve_scores_csv(pred_dir: str) -> Path:
    pred_path = Path(pred_dir)
    scores_csv = pred_path / "scores.csv"
    if scores_csv.exists():
        return scores_csv
    candidates = sorted(pred_path.glob("walk_forward/*/model_predict/scores.csv"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"scores.csv 不存在: {scores_csv}")


def _attach_latest_rank(df: pd.DataFrame, scores_df: pd.DataFrame) -> pd.DataFrame:
    latest_cols = [c for c in ["code", "stock", "date", "percentile", "latest_rank_idx", "latest_rank_pct", "latest_score"] if c in scores_df.columns]
    merged = df.merge(scores_df[latest_cols], on="code", how="left")
    if "name" in merged.columns:
        merged["name"] = merged["name"].fillna(merged["code"]).astype(str).str.zfill(6)
    merged["最新排名"] = merged["latest_rank_idx"]
    merged["最新排名_pct"] = merged["latest_rank_pct"]
    if "stock" not in merged.columns:
        merged["stock"] = merged["code"]
    # current_holding is set by caller based on actual previous week holdings
    if "current_holding" not in merged.columns:
        merged["current_holding"] = False
    if "score" not in merged.columns:
        merged["score"] = merged["latest_score"]
    merged = merged.drop(columns=[c for c in ["latest_rank_idx", "latest_rank_pct", "latest_score"] if c in merged.columns])
    return merged


def _apply_portfolio_weights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "volatility_60d" in out.columns:
        vol = pd.to_numeric(out["volatility_60d"], errors="coerce")
        inv_vol = 1.0 / vol.replace(0, np.nan)
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)
        if inv_vol.notna().sum() > 0:
            inv_vol = inv_vol.fillna(inv_vol.dropna().median() if inv_vol.notna().any() else 1.0)
            out["risk_parity_weight"] = inv_vol / inv_vol.sum()
            out["weight"] = out["risk_parity_weight"]
            return out

    base = pd.to_numeric(out.get("weight"), errors="coerce")
    if base.notna().sum() == 0:
        out["weight"] = 1.0 / max(len(out), 1)
    else:
        base = base.fillna(0.0)
        if base.sum() <= 0:
            out["weight"] = 1.0 / max(len(out), 1)
        else:
            out["weight"] = base / base.sum()
    out["risk_parity_weight"] = out.get("risk_parity_weight", out["weight"])
    return out


def import_env(key, default):
    import os
    return os.environ.get(key, default)


def final_result(second_screen_csv: str, pred_dir: str,
                 output_dir: str, hold_num: int = 5,
                 bottom_pct: float = 0.50,
                 previous_holdings: set | None = None):
    if not (0 < bottom_pct < 1):
        raise ValueError("bottom_pct must be in (0, 1)")
    if previous_holdings is None:
        previous_holdings = set()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── 1. 加载二筛结果 ───────────────────────────
    ss_path = Path(second_screen_csv)
    ss_df = _load_csv(ss_path, "second_screen.csv")
    fallback_candidate_df = None

    if ss_df.empty:
        print(f"⚠ second_screen.csv 为空，回退使用 scores.csv 生成最终结果: {ss_path}")
        scores_csv = _resolve_scores_csv(pred_dir)
        scores_df = _load_csv(scores_csv, "scores.csv")
        if scores_df.empty:
            empty_result = out_path / "result.csv"
            empty_update = out_path / "result_update.csv"
            pd.DataFrame().to_csv(empty_result, index=False, encoding="utf-8-sig")
            pd.DataFrame().to_csv(empty_update, index=False, encoding="utf-8-sig")
            print(f"⚠ scores.csv 也为空，已输出空结果: {empty_result}, {empty_update}")
            return
        if "code" not in scores_df.columns and "instrument" in scores_df.columns:
            scores_df["code"] = _normalize_code(scores_df["instrument"])
        if "code" not in scores_df.columns:
            raise ValueError("scores.csv 中缺少 code/instrument 列，无法生成最终结果")
        scores_df["code"] = _normalize_code(scores_df["code"])
        if "stock" not in scores_df.columns:
            scores_df["stock"] = scores_df["code"]
        if "score" not in scores_df.columns:
            raise ValueError("scores.csv 中缺少 score 列，无法按分数排序")
        scores_df = scores_df.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
        ss_df = scores_df.head(hold_num).copy()
        fallback_candidate_df = scores_df.copy()
        print(f"✓ 以 scores.csv 顶替空的二筛结果，取前 {len(ss_df)} 只股票进入最终结果")

    if "code" not in ss_df.columns and "instrument" in ss_df.columns:
        ss_df["code"] = _normalize_code(ss_df["instrument"])
    if "code" not in ss_df.columns:
        raise ValueError("second_screen.csv 中缺少 code/instrument 列，无法继续")
    ss_df["code"] = _normalize_code(ss_df["code"])
    if "stock" not in ss_df.columns:
        ss_df["stock"] = ss_df["code"]
    print(f"✓ 加载二筛结果: {len(ss_df)} 只股票")
    disp0 = ["code"] + [c for c in ["name", "score", "rank_pct"] if c in ss_df.columns]
    print(ss_df[disp0].to_string(index=False))

    candidates_path = ss_path.with_name("second_screen_candidates.csv")
    if fallback_candidate_df is not None:
        candidate_df = fallback_candidate_df
    elif candidates_path.exists():
        candidate_df = _load_csv(candidates_path, "second_screen_candidates.csv")
        print(f"✓ 加载二筛完整候选池: {len(candidate_df)} 只股票")
    else:
        candidate_df = ss_df.copy()
        print("  ⚠ 未找到 second_screen_candidates.csv，回退为仅使用当前持仓候选")

    if "code" not in candidate_df.columns and "instrument" in candidate_df.columns:
        candidate_df["code"] = _normalize_code(candidate_df["instrument"])
    if "code" not in candidate_df.columns:
        raise ValueError("候选池中缺少 code/instrument 列，无法继续")
    candidate_df["code"] = _normalize_code(candidate_df["code"])
    if "stock" not in candidate_df.columns:
        candidate_df["stock"] = candidate_df["code"]

    # ── 2. 从 scores.csv 获取最新排名 ─────────────
    scores_csv = _resolve_scores_csv(pred_dir)
    scores_df = _load_csv(scores_csv, "scores.csv")
    if scores_df.empty:
        empty_result = out_path / "result.csv"
        empty_update = out_path / "result_update.csv"
        pd.DataFrame().to_csv(empty_result, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(empty_update, index=False, encoding="utf-8-sig")
        print(f"⚠ scores.csv 为空，已输出空结果: {empty_result}, {empty_update}")
        return
    if "code" not in scores_df.columns and "instrument" in scores_df.columns:
        scores_df["code"] = _normalize_code(scores_df["instrument"])
    if "code" not in scores_df.columns:
        raise ValueError("scores.csv 中缺少 code/instrument 列，无法继续")
    scores_df["code"] = _normalize_code(scores_df["code"])
    if "score" not in scores_df.columns:
        raise ValueError("scores.csv 中缺少 score 列，无法计算最新排名")

    total_stocks = len(scores_df)

    # 计算最新排名（基于 score 降序，rank_pct = rank/total * 100%）
    scores_df = scores_df.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
    scores_df["latest_rank_idx"] = scores_df.index + 1
    scores_df["latest_rank_pct"] = ((scores_df["latest_rank_idx"] / total_stocks * 100).round(2).astype(str) + "%")
    scores_df["latest_score"] = pd.to_numeric(scores_df["score"], errors="coerce")

    # ── 3. 拼接最新排名到二筛结果 ─────────────────
    result = _attach_latest_rank(ss_df.copy(), scores_df)
    candidate_df = _attach_latest_rank(candidate_df.copy(), scores_df)
    result = _apply_portfolio_weights(result)
    candidate_df = _apply_portfolio_weights(candidate_df)
    result["是否需要替换"] = False

    # Only mark stocks actually held from previous week as current_holding
    previous_holdings_set = set(str(c).zfill(6) for c in (previous_holdings or set()))
    result["current_holding"] = result["code"].astype(str).str.zfill(6).isin(previous_holdings_set)

    # ── 4. 保存 result.csv ────────────────────────
    result_csv = out_path / "result.csv"
    result.to_csv(result_csv, index=False, encoding="utf-8-sig")

    print(f"\n◆ result.csv 内容:")
    disp_cols = ["code"] + [c for c in ["name", "score", "weight", "rank_pct",
                                        "最新排名", "最新排名_pct",
                                        "valuation_label", "risk_label",
                                        "annualized_return", "max_drawdown",
                                        "sharpe_ratio", "ICIR", "monthly_win_rate"] if c in result.columns]
    print(result[disp_cols].to_string(index=False))
    print(f"\n✓ result.csv 保存: {result_csv}")

    # ── 5. 判断是否需要替换（仅对前周实际持仓中排名在后50%的）────────
    threshold_rank = int(total_stocks * (1 - bottom_pct))
    print(f"\n─── 调仓检查（后{int(bottom_pct*100)}% 阈值 = rank > {threshold_rank} / {total_stocks}）───")
    if previous_holdings_set:
        print(f"  前周持仓: {sorted(previous_holdings_set)}")

    # Only replace stocks that are ACTUAL previous holdings AND in the bottom 50%
    holding_mask = result["current_holding"]
    to_replace_codes = result[holding_mask & (result["最新排名"] > threshold_rank)]["code"].tolist()
    result.loc[result["code"].isin(to_replace_codes), "是否需要替换"] = True

    if not to_replace_codes:
        if previous_holdings_set:
            print("  ✓ 无需调仓，所有前周持仓股票最新排名均未进入后50%")
        else:
            print("  ✓ 首周建仓，无需调仓")
        result_update = result.copy()
        result_update["调仓说明"] = "持仓不变"
    else:
        print(f"  ⚠ 以下前周持仓股票最新排名进入后50%，需替换: {to_replace_codes}")
        already_hold = set(result["code"].tolist())
        candidates = candidate_df[~candidate_df["code"].isin(already_hold)].copy()
        candidates = candidates.sort_values(["最新排名", "score"], ascending=[True, False], na_position="last").reset_index(drop=True)

        result_update = result.copy()
        result_update["调仓说明"] = ""
        result_update["是否需要替换"] = result_update["code"].isin(to_replace_codes)
        result_update["当前持仓"] = result_update["code"]

        for old_code in to_replace_codes:
            old_rank_pct = result_update.loc[result_update["code"] == old_code, "最新排名_pct"].values[0]
            if candidates.empty:
                idx = result_update[result_update["code"] == old_code].index[0]
                result_update.at[idx, "调仓说明"] = f"保留{old_code}：候选池中无可替代标的"
                print(f"  ⚠ 保留 {old_code}：候选池中无可替代标的")
                continue

            new_row = candidates.iloc[0].copy()
            candidates = candidates.iloc[1:].reset_index(drop=True)
            new_code = new_row["code"]

            idx = result_update[result_update["code"] == old_code].index[0]
            for col in result_update.columns:
                if col == "调仓说明":
                    continue
                if col in new_row.index:
                    result_update.at[idx, col] = new_row[col]
            result_update.at[idx, "当前持仓"] = old_code
            result_update.at[idx, "是否需要替换"] = True
            result_update.at[idx, "调仓说明"] = f"卖出{old_code}(最新排名{old_rank_pct})→买入{new_code}"
            print(f"  替换: 卖出 {old_code}（最新排名 {old_rank_pct}）→ 买入 {new_code}")

        result_update.loc[result_update["调仓说明"] == "", "调仓说明"] = "继续持有"

    if "当前持仓" not in result_update.columns:
        result_update["当前持仓"] = result_update["code"]
    result_update = _apply_portfolio_weights(result_update)

    # ── 6. 保存 result_update.csv ─────────────────
    result_update_csv = out_path / "result_update.csv"
    result_update.to_csv(result_update_csv, index=False, encoding="utf-8-sig")

    print(f"\n◆ result_update.csv 内容:")
    disp_cols2 = ["code"] + [c for c in ["name", "score", "rank_pct",
                                          "percentile", "weight", "当前持仓",
                                          "最新排名", "最新排名_pct",
                                          "valuation_label", "risk_label", "是否需要替换",
                                          "调仓说明"] if c in result_update.columns]
    print(result_update[disp_cols2].to_string(index=False))
    print(f"\n✓ result_update.csv 保存: {result_update_csv}")

    # ── 7. 打印操盘摘要 ───────────────────────────
    print("\n═══════════════════════════════════════════════")
    print("  本周五操盘建议摘要")
    print("═══════════════════════════════════════════════")
    cash = float(import_env("CASH_TOTAL", 20000))
    fee = float(import_env("TX_FEE_RATE", 0.0001))
    stamp = float(import_env("STAMP_DUTY_RATE", 0.0005))
    hold = int(import_env("HOLD_NUM", 5))

    per_stock = cash / hold
    print(f"  账户资金: ¥{cash:,.0f}")
    print(f"  持仓数量: {hold} 支")
    print(f"  每支仓位: ¥{per_stock:,.0f}（建议）")
    print(f"  交易费率: 万分之{fee*10000:.0f} + 印花税万分之{stamp*10000:.0f}（卖出）")
    print("  持仓股票:")
    for _, row in result_update.iterrows():
        note = row.get("调仓说明", "")
        latest_pct = row.get("最新排名_pct", "N/A")
        print(f"    {row['code']}  最新排名={latest_pct}  {note}")
    print("═══════════════════════════════════════════════")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--second-screen", required=True, dest="second_screen")
    ap.add_argument("--pred-dir", required=True, dest="pred_dir")
    ap.add_argument("--output", required=True)
    ap.add_argument("--hold-num", type=int, default=5, dest="hold_num")
    ap.add_argument("--bottom-pct", type=float, default=0.50, dest="bottom_pct")
    args = ap.parse_args()
    final_result(args.second_screen, args.pred_dir, args.output, args.hold_num, args.bottom_pct)


if __name__ == "__main__":
    main()