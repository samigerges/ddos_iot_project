# Website deployment

The public app is https://ddos-cnrkbuuhsodaj4gwwgpp.streamlit.app/. The website
update targets this repository's `main` branch with `app.py` as its entry point.

`requirements.txt` pins the same main library versions used by the local Docker
image. `requirements-docker.txt` includes that file. `packages.txt` supplies the
OpenMP runtime needed by the native estimators on Streamlit Community Cloud.
Use Python 3.11 or newer on the hosting service.

The overview reads the small reports listed in `configs/website_reports.json`.
They contain the saved model comparisons and validation/test results used by the
three charts. Their combined size is about 335 KB. The Git ignore rules retain
these display reports while keeping generated datasets and new trained model
bundles local. Previously tracked model bundles remain in the repository for
saved-model prediction. Raw training CSVs are supplied separately.

After updating source, theme configuration, or the allowlisted display reports,
commit and push to `main`. Streamlit Community Cloud updates the app from GitHub.
See the official deployment and dependency documentation:

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

The original full research snapshot is preserved locally on
`codex/local-research-snapshot-382f9a9`; it is not part of the website upload.
