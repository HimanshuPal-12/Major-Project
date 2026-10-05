# FD002 Engine Monitor

Streamlit research replay of 260 simulated NASA C-MAPSS engines across five held-out folds. Ten methods, sensor traces, delayed feedback, source-cited research retrieval and retrospective metrics are included. Scores are anomaly severity, not calibrated failure probabilities. This is a saved research replay.

## Deploy on Streamlit Community Cloud

1. Publish the contents of this directory in a dedicated GitHub repository.
2. At https://share.streamlit.io/ select Create app and select that repository and its branch.
3. Set the main file to `dashboard.py` and choose Python 3.12 in Advanced settings.
4. Deploy; verify Fleet, Engine monitor, Research evidence, and Study & method.

No secrets or paid API are required. Hosting requires your GitHub/Streamlit account. Optional Qwen weights are omitted; research explanations use the explicit retrieval-only fallback. Prediction partitions preserve every original CSV row and column, with exact pandas round-trip validation.

Source dataset: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
The corpus manifest records each research excerpt's citation and hash. Source integrity is not mechanical-domain approval.
