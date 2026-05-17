"""Walk-forward cross-sectional return prediction pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from empirlab.finance.factor_model import MLFactorModel
from empirlab.utils.metrics import ic, rmse


class ReturnPredictor:
    """Walk-forward cross-sectional return predictor.

    Trains an MLFactorModel each period on an expanding (or rolling) window
    and evaluates on the next period (one-step-ahead OOS forecast).

    Parameters
    ----------
    method : str, default 'enet'
    window : int or None
        Training window length in periods. None = expanding window.

    Examples
    --------
    >>> pred = ReturnPredictor(method='ridge', window=60)
    >>> results = pred.walk_forward(chars, returns, date_col='date')
    >>> print(results[['date','ic','rmse']].tail())
    """

    def __init__(self, method: str = "enet", window: int = None):
        self.method = method
        self.window = window

    def walk_forward(
        self,
        chars: pd.DataFrame,
        returns: pd.Series,
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Run walk-forward prediction.

        chars and returns must share a MultiIndex (date, asset) or have
        a 'date' column at level 0.

        Returns
        -------
        DataFrame with columns: date, ic, rmse, r2_oos
        """
        dates = chars.index.get_level_values(date_col).unique().sort_values()
        records = []
        for t in range(1, len(dates)):
            start   = max(0, t - self.window) if self.window else 0
            tr_dates = dates[start:t]
            te_date  = dates[t]

            X_tr = chars.loc[tr_dates].values
            y_tr = returns.loc[tr_dates].values
            X_te = chars.loc[[te_date]].values
            y_te = returns.loc[[te_date]].values

            if len(X_tr) < 10 or len(np.unique(y_tr)) < 2:
                continue  # skip periods with too little data

            model = MLFactorModel(method=self.method)
            model.fit(X_tr, y_tr)
            yhat  = model.predict(X_te)

            records.append({
                "date":   te_date,
                "ic":     ic(yhat, y_te),
                "rmse":   rmse(yhat, y_te),
                "r2_oos": model.r2_oos(X_te, y_te),
            })
        return pd.DataFrame(records)
