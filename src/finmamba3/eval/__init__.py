"""Evaluation harnesses: backtest, regime-shift split, baseline runners.

New in this version
-------------------
* ``regime_diagnostics`` – diagnose RegimeFiLM head structure (persistence,
  usage distribution, vol-regime correlation).
* ``regime_gap_cli``     – market-level regime gap evaluation using
  ``regime_split`` infrastructure.
* ``linear_probe``       – linear probing of posterior latents to verify the
  latent encodes vol, imbalance, spread, and direction structure.

Inline validation additions (``train._validation_metrics``)
------------------------------------------------------------
* ``Val/identity_baseline_mse``        – trivial copy-last-obs baseline MSE.
* ``Val/prior_vs_identity_ratio``      – prior MSE / identity MSE; < 1 means
  the backbone beats the trivial predictor.
* ``Val/prior_entropy_mse_corr``       – Pearson r between per-step prior
  entropy and per-step prediction error (calibration signal).
* ``Val/regime_gap/*``                 – window-level vol-split gap metrics
  (high_vol − low_vol for next-step MSE and direction accuracy).
* ``Val/direction_head_accuracy``      – direction head accuracy at the
  configured bucket threshold (distinct from the decoded-prior accuracy).

Inline imagination additions (``train._imagine_and_log``)
----------------------------------------------------------
* ``Imagine/absret_autocorr_lag{1,5,10}`` – |Δmid| autocorrelation.
* ``Imagine/sign_autocorr_lag1``           – sign autocorrelation (mean-rev).
* ``Imagine/spread_autocorr_lag1``         – spread-level autocorrelation.
"""

