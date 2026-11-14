#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_weekly.py —— 周度频率的黄金择时模型 (日度噪声大, 用户要求至少周度)
与日度 pipeline.py 同构, 但:
  - 面板按 W-FRI 重采样为周度, 收益/差分由周度水平计算
  - 窗口按周度缩放: EWLS 26周/半衰13周, z-score 52周, 耦合26周, 缺口去趋势52周, 得分z 104周
输出 output/features_weekly.csv, output/summary_weekly.json
"""
import os, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore"); np.seterr(all="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW  = os.path.join(ROOT, "data", "raw")
OUT  = os.path.join(ROOT, "output")

def load_fred(sid):
    df = pd.read_csv(os.path.join(RAW, f"fred_{sid}.csv"), parse_dates=[0])
    df.columns = ["date", "value"]; df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")["value"]

def load_yahoo(sym):
    df = pd.read_csv(os.path.join(RAW, f"yahoo_{sym.replace('^','')}.csv"), parse_dates=["Date"]).set_index("Date")
    px = df["AdjClose"].where(df["AdjClose"].notna(), df["Close"])
    return px.astype(float).dropna()

def build_weekly_panel():
    gold = load_yahoo("GLD"); dxy = load_yahoo("DX-Y.NYB"); spx = load_yahoo("^GSPC"); oil = load_yahoo("CL=F")
    cols = {
        "gold": gold, "dxy": dxy, "spx": spx, "oil": oil,
        "dgs10": load_fred("DGS10"), "dgs2": load_fred("DGS2"),
        "real10": load_fred("DFII10"), "be": load_fred("T10YIE"),
        "credit": load_fred("BAA10Y"), "hy_oas": load_fred("BAMLH0A0HYM2"),
        "ig_oas": load_fred("BAMLC0A0CM"), "nfci": load_fred("NFCI"),
        "nfci_lev": load_fred("NFCINONFINLEVERAGE"), "vix": load_fred("VIXCLS"),
    }
    df = pd.DataFrame(cols).ffill()
    df = df[df["gold"].notna()].copy()
    w = df.resample("W-FRI").last().dropna(subset=["gold"])
    return w

def logret(s): return np.log(s).diff()

CORE_FACTORS = ["dxy_ret", "spx_ret", "d_real10", "d_be", "oil_ret", "d_credit", "d_nfci"]
FACTOR_LABELS = {"dxy_ret":"美元(DXY)","spx_ret":"美股(标普)","d_real10":"实际利率(10Y TIPS)",
                 "d_be":"通胀补偿","oil_ret":"原油(WTI)","d_credit":"信用利差(Baa-10Y)","d_nfci":"金融条件(NFCI)"}

def make_features(d):
    d = d.copy()
    d["gold_ret"] = logret(d["gold"]); d["dxy_ret"] = logret(d["dxy"])
    d["spx_ret"] = logret(d["spx"]); d["oil_ret"] = logret(d["oil"])
    d["d_real10"] = d["real10"].diff(); d["d_be"] = d["be"].diff(); d["d_nom10"] = d["dgs10"].diff()
    d["d_credit"] = d["credit"].diff(); d["d_hy"] = d["hy_oas"].diff(); d["d_ig"] = d["ig_oas"].diff()
    d["d_nfci"] = d["nfci"].diff(); d["d_vix"] = d["vix"].diff()
    return d

def ewls_beta(X, y, w):
    W = np.diag(w); Xw = X.T @ W
    return np.linalg.lstsq(Xw @ X, Xw @ y, rcond=None)[0]

def rolling_ewls(y, X, window=26, halflife=13):
    n, k = X.shape; lam = 0.5 ** (1.0 / halflife)
    betas = np.full((n, k), np.nan); r2 = np.full(n, np.nan); fitted = np.full(n, np.nan); resid = np.full(n, np.nan)
    for t in range(window - 1, n):
        Xw = X[t-window+1:t+1]; yw = y[t-window+1:t+1]
        w = lam ** np.arange(window-1, -1, -1)
        b = ewls_beta(Xw, yw, w)
        betas[t] = b; f = X[t] @ b; e = y[t] - f; fitted[t] = f; resid[t] = e
        ss_tot = np.sum(w*(yw - np.average(yw, weights=w))**2); ss_res = np.sum(w*(yw - Xw@b)**2)
        r2[t] = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return betas, r2, fitted, resid

def causal_zscore(s, win=52):
    mu = s.rolling(win, min_periods=win//2).mean(); sd = s.rolling(win, min_periods=win//2).std()
    return (s - mu) / sd

def kalman_betas(y, X, q=1e-3, r=1.0, min_obs=20):
    n, k = X.shape; beta = np.zeros(k); P = np.eye(k)*1.0; Q = np.eye(k)*q
    betas = np.full((n, k), np.nan)
    for t in range(n):
        x = X[t]; beta_p = beta; P_p = P + Q
        S_ = x @ P_p @ x + r; K = P_p @ x / S_; inn = y[t] - x @ beta_p
        beta = beta_p + K*inn; P = P_p - np.outer(K, x) @ P_p
        if t+1 >= min_obs: betas[t] = beta
    return betas

def mutual_collateral(d):
    d = d.copy()
    rets = pd.DataFrame({"gold": d["gold_ret"], "dxy": d["dxy_ret"], "spx": d["spx_ret"],
                         "oil": d["oil_ret"], "bond": -8.0*d["d_nom10"], "credit": -4.0*d["d_credit"]})
    win = 26; n = len(d)
    ar = np.full(n, np.nan); avgcorr = np.full(n, np.nan)
    R = rets.to_numpy(); Rz = (R - np.nanmean(R, axis=0)) / np.nanstd(R, axis=0); Rz = np.nan_to_num(Rz)
    for t in range(win-1, n):
        W = Rz[t-win+1:t+1]; C = np.corrcoef(W, rowvar=False)
        ev = np.linalg.eigvalsh(C); ar[t] = ev[-1]/ev.sum()
        k = C.shape[0]; avgcorr[t] = C[np.triu_indices(k,1)].mean()
    d["ar"] = ar; d["avgcorr"] = avgcorr
    d["leverage_z"] = causal_zscore(d["nfci_lev"]); d["credit_z"] = causal_zscore(d["credit"]); d["vix_z"] = causal_zscore(d["vix"])
    d["ar_z"] = causal_zscore(pd.Series(ar, index=d.index))
    d["cci"] = 0.35*d["ar_z"] + 0.25*d["leverage_z"] + 0.20*d["credit_z"] + 0.20*d["vix_z"]
    d["cci_trigger"] = d["cci"] - d["cci"].rolling(4, min_periods=2).mean()
    return d

def build_signal(d):
    d = d.copy()
    d["risk_idx"] = 0.6*d["vix_z"] + 0.4*d["credit_z"]
    d["regime"] = 2; d.loc[d["risk_idx"] < -0.5, "regime"] = 1; d.loc[d["risk_idx"] > 1.0, "regime"] = 3
    d["cumresid"] = d["oos_resid"].cumsum()
    d["gap"] = d["cumresid"] - d["cumresid"].rolling(52, min_periods=26).mean()
    d["sig_coupling"] = np.tanh(causal_zscore(d["avgcorr"], 52)/2.0)
    d["sig_credit"]   = np.tanh(causal_zscore(d["credit"], 52)/2.0)
    d["sig_gap"]      = np.tanh(causal_zscore(d["gap"].fillna(0.0), 104)/2.0)
    # 股债相关性(保留参考)
    bond_ret = -8.0 * d["dgs10"].diff()
    d["sb_corr"] = d["spx_ret"].rolling(26).corr(bond_ret)
    d["sig_sbcorr"] = np.tanh(causal_zscore(d["sb_corr"], 52)/2.0)
    # 新因子: 油金比动量(黄金相对原油的26周动量, 商品轮动/去美元化 vs 能源周期)
    d["oil_gold"] = (np.log(d["gold"]) - np.log(d["oil"])).diff(26)
    d["sig_oilgold"] = np.tanh(causal_zscore(d["oil_gold"], 52)/2.0)
    # 挑战者模型(训练段7因子拟合, 归一化权重 4:1:4): 信用压力 + 公允价值动量 + 油金比动量
    d["score"] = 0.444*d["sig_credit"] + 0.111*d["sig_gap"] + 0.444*d["sig_oilgold"]
    d["score"] = d["score"].rolling(1, min_periods=1).mean()
    d["pos_lf"] = (d["score"] > 0.0).astype(float)
    ma40 = np.log(d["gold"]).rolling(40).mean()
    trend_down = (np.log(d["gold"]) < ma40).astype(float)
    d["pos_ls"] = ((d["score"] > 0.25).astype(float) - ((d["score"] < -0.25).astype(float)*trend_down))
    d["fv_sig"] = d["sig_gap"]; d["cci_sig"] = d["sig_coupling"]; d["reg_sig"] = d["regime"]-2.0
    return d

def backtest(d, pos_col, cost=0.0010):
    pos = d[pos_col].fillna(0.0); ret = d["gold_ret"].fillna(0.0)
    return pos.shift(1).fillna(0.0)*ret - cost*pos.diff().abs().fillna(0.0)

def metrics(ret, ann=52):
    ret = ret.dropna()
    if len(ret)==0: return {}
    tot=(1+ret).prod()-1; cagr=(1+tot)**(ann/len(ret))-1; vol=ret.std()*np.sqrt(ann)
    sharpe=cagr/vol if vol>0 else np.nan
    cum=(1+ret).cumprod(); dd=(cum/cum.cummax()-1); mdd=dd.min()
    return {"total_return":tot,"cagr":cagr,"vol":vol,"sharpe":sharpe,"max_drawdown":mdd,"calmar":cagr/abs(mdd) if mdd!=0 else np.nan}

def full_sample_explain(d):
    y=d["gold_ret"]; X=d[CORE_FACTORS]; m=y.notna() & X.notna().all(axis=1)
    y,X = y[m].to_numpy(), X[m].to_numpy()
    b=np.linalg.lstsq(X,y,rcond=None)[0]; yhat=X@b
    r2=1-np.sum((y-yhat)**2)/np.sum((y-y.mean())**2)
    contrib=b*np.cov(X.T,y)[-1,:-1]/np.var(y)
    return r2, dict(zip(CORE_FACTORS,contrib)), b

def turning_points(d, pos_col, fwd=12):
    pos=d[pos_col].fillna(0.0); flip=(pos!=pos.shift(1))&pos.notna()
    out=[]
    for i in d.index[flip]:
        j=d.index.get_loc(i); f=d["gold_ret"].iloc[j+1:j+1+fwd].sum()
        out.append({"date":str(i.date()),"pos":float(pos.loc[i]),"score":round(float(d["score"].loc[i]),3),
                    "fwd_ret":round(float(f),4) if not np.isnan(f) else None})
    return out

def main():
    df = build_weekly_panel(); df = make_features(df)
    print(f"周度面板: {df.index[0].date()} .. {df.index[-1].date()}, 周数 {len(df)}")
    y = df["gold_ret"].to_numpy(); X = df[CORE_FACTORS].to_numpy()
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1); y = np.where(mask, y, 0.0); X = np.nan_to_num(X)
    betas, r2, fitted, resid = rolling_ewls(y, X, 26, 13)
    beta_lag = np.vstack([np.full((1, X.shape[1]), np.nan), betas[:-1]])
    oos = np.full(len(y), np.nan); oos[1:] = y[1:] - np.einsum("ij,ij->i", X[1:], beta_lag[1:])
    for i,f in enumerate(CORE_FACTORS): df[f"b_{f}"]=betas[:,i]; df[f"k_{f}"]=np.nan
    df["r2_ewls"]=r2; df["resid"]=resid; df["oos_resid"]=oos
    ys = causal_zscore(pd.Series(y, index=df.index)).to_numpy()
    Xs = df[CORE_FACTORS].apply(lambda c: causal_zscore(c)).to_numpy(); Xs=np.nan_to_num(Xs); ys=np.nan_to_num(ys)
    kb = kalman_betas(ys, Xs)
    for i,f in enumerate(CORE_FACTORS): df[f"k_{f}"]=kb[:,i]
    df = mutual_collateral(df); df = build_signal(df)
    df["strat_lf"]=backtest(df,"pos_lf"); df["strat_ls"]=backtest(df,"pos_ls"); df["bh"]=df["gold_ret"]
    r2_full, contrib, b_full = full_sample_explain(df)
    print(f"周度全样本 R² = {r2_full:.3f}")
    for f in CORE_FACTORS:
        print(f"  {FACTOR_LABELS[f]:22s} beta={b_full[list(CORE_FACTORS).index(f)]:+.4f} contrib={contrib[f]:+.4f}")
    start = df["score"].first_valid_index(); seg = df.loc[start:]
    m_bh=metrics(seg["bh"]); m_lf=metrics(seg["strat_lf"]); m_ls=metrics(seg["strat_ls"])
    print(f"回测 {start.date()}..{seg.index[-1].date()}")
    print(f"  Buy&Hold: CAGR={m_bh['cagr']:+.1%} Sharpe={m_bh['sharpe']:.2f} MDD={m_bh['max_drawdown']:+.1%}")
    print(f"  仅做多  : CAGR={m_lf['cagr']:+.1%} Sharpe={m_lf['sharpe']:.2f} MDD={m_lf['max_drawdown']:+.1%}")
    print(f"  多空    : CAGR={m_ls['cagr']:+.1%} Sharpe={m_ls['sharpe']:.2f} MDD={m_ls['max_drawdown']:+.1%}")
    cols = (["gold","gold_ret","dxy","spx","oil","real10","be","dgs10","dgs2","credit","hy_oas","ig_oas",
             "nfci","nfci_lev","vix"]+CORE_FACTORS+[f"b_{f}" for f in CORE_FACTORS]+[f"k_{f}" for f in CORE_FACTORS]
            +["r2_ewls","resid","oos_resid","cumresid","gap","sig_coupling","sig_credit","sig_gap","sb_corr","sig_sbcorr",
              "oil_gold","sig_oilgold","fv_sig",
              "ar","avgcorr","leverage_z","credit_z","vix_z","ar_z","cci","cci_trigger","risk_idx","regime",
              "cci_sig","reg_sig","score","pos_lf","pos_ls","strat_lf","strat_ls","bh"])
    df[cols].to_csv(os.path.join(OUT,"features_weekly.csv"))
    summary = {"sample":{"start":str(df.index[0].date()),"end":str(df.index[-1].date())},
               "backtest_start":str(start.date()),"freq":"weekly",
               "full_r2_weekly":r2_full,"full_contrib":contrib,
               "full_beta":dict(zip(CORE_FACTORS, b_full.tolist())),
               "metrics":{"bh":m_bh,"long_flat":m_lf,"long_short":m_ls},
               "turns_longflat":turning_points(df,"pos_lf")}
    with open(os.path.join(OUT,"summary_weekly.json"),"w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
    print(f"已保存 features_weekly.csv / summary_weekly.json")

if __name__ == "__main__":
    main()
