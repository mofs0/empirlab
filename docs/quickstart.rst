Quick Start
===========

Double Machine Learning
-----------------------

.. code-block:: python

    from empirlab.causal import DoubleML
    from empirlab.causal.datasets import make_plr_data

    X, y, d = make_plr_data(n=2000, p=20, theta=1.2, seed=42)
    dml = DoubleML(ml_l="lasso", ml_m="lasso", n_folds=5)
    dml.fit(X, y, d)
    print(dml.summary())

Causal Forest
-------------

.. code-block:: python

    from empirlab.causal import CausalForest
    from empirlab.causal.datasets import make_hte_data

    X, y, d, tau_true = make_hte_data(n=2000, p=10)
    cf = CausalForest(n_estimators=2000).fit(X, y, d)
    lb, ub = cf.confidence_interval(X, alpha=0.05)
    print(cf.summary(X))

Synthetic DiD
-------------

.. code-block:: python

    import numpy as np
    from empirlab.causal import SyntheticDiD

    Y = np.random.standard_normal((30, 20))
    Y[0, 14:] += 2.5                        # treat unit 0 at period 14
    sdid = SyntheticDiD(n_boot=200)
    sdid.fit(Y, treated_units=[0], T_pre=14)
    print(sdid.summary())

ML Factor Model
---------------

.. code-block:: python

    from empirlab.finance import MLFactorModel

    model = MLFactorModel(method="enet")
    model.fit(chars_train, returns_train)
    r_hat = model.predict(chars_test)
    print(f"OOS R² = {model.r2_oos(chars_test, returns_test):.4f}")

Financial Sentiment
-------------------

.. code-block:: python

    from empirlab.llm import FinSentiment

    pipe = FinSentiment(model="finbert")
    scores = pipe.score([
        "Earnings beat expectations by 15%",
        "Revenue missed targets amid weak demand",
    ])
    # [0.87, -0.79]
