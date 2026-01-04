# Deployment Guide: Energy Demand Forecasting App

This guide walks you through deploying your Streamlit app to Streamlit Cloud.

## 1. Pre-Deployment Check (Already Complete)

I have prepared your repository for a smooth deployment:
- **Self-Contained App**: Copied model and data files into `streamlit_app/model` and `streamlit_app/data`.
- **Paths Updated**: Modified `app.py` to look for files within the `streamlit_app` folder.
- **Dependencies**: Verified `requirements.txt` includes all necessary libraries (`tensorflow-cpu`, `plotly`, `pandas`, etc.).

## 2. Commit and Push

Before deploying, you must push the changes to GitHub.

```bash
git add streamlit_app/ deployment/
git commit -m "Prepare app for Streamlit Cloud deployment: assets and documentation"
git push origin main
```

## 3. Deploy to Streamlit Cloud

1.  Go to [share.streamlit.io](https://share.streamlit.io/).
2.  Click **"New app"**.
3.  Select your repository: `energy-demand-forecasting-lstm`.
4.  **Main file path**: Change this to:
    ```
    streamlit_app/app.py
    ```
    *(Note: The "App URL" will be automatically generated)*
5.  Click **"Deploy!"**.

## 4. Troubleshooting Common Issues

-   **"Module not found"**: Ensure `requirements.txt` is in the `streamlit_app` folder (it is).
-   **"File not found"**: The app now looks for data in `./data` relative to `app.py`. Ensure git tracked the `.parquet` and `.keras` files (we verified they are not ignored).
-   **Memory Constraints**: We are using `tensorflow-cpu` to save memory. If the app crashes due to memory, we might need to reduce the model size or data window, but currently, it uses a lightweight LSTM which should be fine.

## 5. Verify Deployment

Once deployed, the app should show:
-   The "Strategic Energy Demand Forecasting" title.
-   An interactive Plotly chart.
-   Sidebar controls for "Observation Point" and "Forecast Horizon".

If you see the chart, **Success!** 🚀
