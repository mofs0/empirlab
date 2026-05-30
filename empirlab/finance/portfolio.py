"""
ML Long-Short Portfolio
========================
Reference: Gu, S., Kelly, B., & Xiu, D. (2020).
           Empirical asset pricing via machine learning.
           Review of Financial Studies, 33(5), 2223-2273.

Walk-forward long-short portfolio construction from ML return predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class MLPortfolio:
    """Walk-forward long-short portfolio from ML predicted returns.

    Parameters
    ----------
    q_long : float, default 0.9
        Quantile threshold for the long leg (top decile by default).
    q_short : float, default 0.1
        Quantile threshold for the short leg.
    weighting : {'equal', 'value'}
        'equal' — equal-weight within each leg.
        'value' — proportional to abs(predicted return).
    rebalance : str, default 'monthly'
        Rebalancing frequency: 'daily', 'monthly', 'quarterly'.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> dates = pd.date_range("2020-01-01", periods=24, freq="ME")
    >>> n_stocks = 50
    >>> rng = np.random.default_rng(0)
    >>> preds = pd.DataFrame(rng.standard_normal((24, n_stocks)), index=dates)
    >>> rets  = pd.DataFrame(rng.standard_normal((24, n_stocks)) * 0.05, index=dates)
    >>> port  = MLPortfolio()
    >>> perf  = port.backtest(preds, rets)
    >>> print(perf[["long_ret", "short_ret", "ls_ret", "sharpe"]].tail())
    """

    def __init__(self, q_long: float = 0.9, q_short: float = 0.1,
                 weighting: str = "equal", rebalance: str = "monthly"):
        self.q_long    = q_long
        self.q_short   = q_short
        self.weighting = weighting
        self.rebalance = rebalance

    # ------------------------------------------------------------------
    def _weights(self, pred_row: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Return long and short weight vectors for a single period."""
        lo = pred_row.quantile(self.q_short)
        hi = pred_row.quantile(self.q_long)
        long_mask  = pred_row >= hi
        short_mask = pred_row <= lo

        if self.weighting == "equal":
            w_long  = long_mask.astype(float)
            w_short = short_mask.astype(float)
        else:
            w_long  = pred_row.clip(lower=0) * long_mask
            w_short = pred_row.abs().clip(lower=0) * short_mask

        if w_long.sum()  > 0: w_long  /= w_long.sum()
        if w_short.sum() > 0: w_short /= w_short.sum()
        return w_long, w_short

    # ------------------------------------------------------------------
    def backtest(self, predictions: pd.DataFrame,
                 returns: pd.DataFrame) -> pd.DataFrame:
        """Walk-forward backtest.

        Parameters
        ----------
        predictions : DataFrame (T, N)
            Predicted returns, indexed by date.
        returns : DataFrame (T, N)
            Realised returns at the same frequency, same columns.

        Returns
        -------
        DataFrame with columns:
            long_ret, short_ret, ls_ret, long_n, short_n,
            cum_long, cum_short, cum_ls, sharpe
        """
        preds = predictions.copy()
        rets  = returns.reindex(columns=preds.columns)
        records = []

        for t, date in enumerate(preds.index[:-1]):
            pred_row = preds.iloc[t]
            ret_row  = rets.iloc[t + 1]  # next period realised return

            w_long, w_short = self._weights(pred_row)
            lr = float((w_long  * ret_row).sum())
            sr = float((w_short * ret_row).sum())
            records.append({
                "date":     preds.index[t + 1],
                "long_ret": lr,
                "short_ret": sr,
                "ls_ret":   lr - sr,
                "long_n":   int((w_long  > 0).sum()),
                "short_n":  int((w_short > 0).sum()),
            })

        result = pd.DataFrame(records).set_index("date")
        result["cum_long"]  = (1 + result["long_ret"]).cumprod()
        result["cum_short"] = (1 + result["short_ret"]).cumprod()
        result["cum_ls"]    = (1 + result["ls_ret"]).cumprod()

        # Annualised Sharpe (assume monthly returns by default)
        freq_map = {"daily": 252, "monthly": 12, "quarterly": 4}
        ann = freq_map.get(self.rebalance, 12)
        ls  = result["ls_ret"]
        result["sharpe"] = (ls.mean() / ls.std(ddof=1) * np.sqrt(ann)
                            if ls.std() > 0 else 0.0)
        self.result_ = result
        return result

    # ------------------------------------------------------------------
    def performance_summary(self) -> pd.DataFrame:
        """Summary statistics after backtest."""
        if not hasattr(self, "result_"):
            raise RuntimeError("Call backtest() first.")
        r = self.result_
        freq_map = {"daily": 252, "monthly": 12, "quarterly": 4}
        ann = freq_map.get(self.rebalance, 12)

        rows = {}
        for col, label in [("long_ret", "Long"), ("short_ret", "Short"), ("ls_ret", "L/S")]:
            s = r[col]
            ann_ret = (1 + s).prod() ** (ann / len(s)) - 1
            vol = s.std(ddof=1) * np.sqrt(ann)
            sr = ann_ret / vol if vol > 0 else np.nan
            # Max drawdown
            cum = (1 + s).cumprod()
            dd = (cum / cum.cummax() - 1).min()
            rows[label] = {"Ann. Return": ann_ret, "Ann. Vol": vol,
                           "Sharpe": sr, "Max DD": dd}
        return pd.DataFrame(rows).T
