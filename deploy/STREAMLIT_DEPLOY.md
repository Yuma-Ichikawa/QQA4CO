# Deploying QQA4CO to Streamlit Community Cloud

This document is the operator's runbook for the public dashboard at
<https://parallelquasiquantum4co.streamlit.app/>.

## 1. Why is the URL redirecting to `/-/auth/app`?

A `curl -I` on the public URL currently returns a `303` redirect to
`https://parallelquasiquantum4co.streamlit.app/-/auth/app?redirect_uri=…`,
then cycles through `/-/login?payload=…` and back. This is **not** an
application-side problem — it means the app is configured as **Private** on
Streamlit Community Cloud, so every visitor is forced through SSO.

### Fix (one-time, 60 seconds)

1. Sign in at <https://share.streamlit.io/> with the GitHub account that
   owns `Yuma-Ichikawa/QQA4CO`.
2. In the dashboard, locate the `parallelquasiquantum4co` app.
3. Click the three-dot menu → **Settings** → **Sharing**.
4. Switch *"Who can view this app?"* from **"Only specific people"** to
   **"Anyone with the link can view"**.
5. Save. The redirect stops within ~30 seconds.

A quick sanity-check:

```bash
curl -I https://parallelquasiquantum4co.streamlit.app/_stcore/health
# Expected: HTTP/2 200   and body "ok"
```

## 2. Deployment artefacts

The live app is driven by these files — everything runs from the repository
root with `app/streamlit_app.py` as the entry point.

| File | Purpose |
| --- | --- |
| `app/streamlit_app.py` | Home / problem-definition page |
| `app/pages/1_Solve.py` | Live anneal + visualisation |
| `app/pages/2_Visualize.py` | Rich plots from the last run |
| `app/pages/3_Compare.py` | Hyper-parameter sweeps |
| `app/_common.py` | Theme / problem builder helpers |
| `app/requirements.txt` | Entrypoint-local redirect to the pinned CPU-only dependency list; takes precedence over root `uv.lock` |
| `requirements.txt` | Reproducible Python 3.12 deployment set with the current CPU-only PyTorch wheel |
| `runtime.txt` | `python-3.12` — matches the current Community Cloud default |
| `.streamlit/config.toml` | Theme + server / browser / client settings |

If you need to debug a deploy failure, temporarily flip
`.streamlit/config.toml::client.showErrorDetails` from `false` to `true`,
push, watch the Streamlit Cloud logs, then revert.

## 3. Deploy / redeploy flow

1. Merge your changes into `main`.
2. In the Streamlit Cloud dashboard → app → **Manage app** → **Reboot**.
   The image is rebuilt and the new code is served within ~2 minutes.
3. Visit `/_stcore/health` first — it skips all application code, so any
   `200 ok` means the container is up and the UI is reachable.
4. Visit `/` to walk through Home → Solve → Visualize.

Community Cloud selects the Python version in deployment settings. Existing
apps cannot change it in place: to move an older deployment to Python 3.12,
record its subdomain/settings, delete it, and redeploy with Python 3.12 in
**Advanced settings**. Dependency-only updates continue to redeploy in place.

## 4. Embedding the app elsewhere

Once the app is public, you can embed it as an iframe:

```html
<iframe
  src="https://parallelquasiquantum4co.streamlit.app/?embed=true"
  width="100%"
  height="900"
  style="border:0"
  loading="lazy">
</iframe>
```

The `?embed=true` query parameter hides the Streamlit header/footer so the
app feels native.

## 5. Health-check script

Use `scripts/check_streamlit_deploy.py` locally (or in CI) to verify that
the deployment is live:

```bash
uv run python scripts/check_streamlit_deploy.py
# exits 0 when the health endpoint returns "ok"; non-zero otherwise.
```

## 6. Common issues

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Redirect loop to `/-/auth/app` | Private sharing | Set sharing to *Anyone with the link can view* (see §1) |
| App shows a generic error page | Build failure (usually `pip`) | Temporarily enable `showErrorDetails`, redeploy, read logs |
| Cold start feels slow (~30 s) | Free tier image spin-up | Expected; warm starts are sub-second |
| `torch` wheel OOM during build | CUDA wheel pulled | Keep the exact `torch==...+cpu` pin and CPU `--find-links` source in `requirements.txt` |
| `plotly` missing | Not in `requirements.txt` | Keep the exact Plotly deployment pin in `requirements.txt` (not just the `[plotly]` extra) |
