# BTC 15-Min Directional & Flip Predictor (Kalshi)

Backend for logging BTC trade/order-book data every 15-min window, training a
directional model (P(up)) and a late-window "flip" model, and serving live
scores. Frontend/dashboard lives in a separate GitHub repo; this repo is the
data + model + serving backend, designed to run inside AWS free tier.

## Order of operations

1. **AWS setup (Colab)** — run `colab/01_aws_setup.ipynb` cells once, from your
   own machine/Colab, using an IAM user's access keys (not root). Creates:
   - DynamoDB tables (`btc_ticks`, `btc_windows`)
   - S3 bucket (model artifacts)
   - IAM role + instance profile for the EC2 box
2. **EC2 box** — launch a `t3.micro` (or `t2.micro`), attach the instance
   profile created above, clone this repo, install requirements.
3. **Run ingestion** (`ingestion/run_ingestion.py`) as a systemd service or
   `screen`/`tmux` session — logs ticks + order book continuously, writes
   window summaries to DynamoDB on each 15-min boundary.
4. **Let it run 1-2 weeks minimum** before trusting any model output. This is
   a cold-start problem — an undertrained model is worse than no model.
5. **Train** (`models/train.py`) — pulls closed windows from DynamoDB, builds
   walk-forward train/test splits, trains both models, saves to S3 + local
   `models/artifacts/`.
6. **Serve** (`serving/app.py`) — FastAPI app scoring live windows, exposing
   JSON for your GitHub-hosted dashboard to poll.

## Directory structure

```
ingestion/        # WebSocket clients: trades + order book -> DynamoDB
features/         # Shared feature computation (used by ingestion, training, serving)
models/           # Training scripts for directional + flip models
serving/          # FastAPI live-scoring app
aws_setup/        # Plain boto3 scripts (same logic as the Colab notebook, importable)
colab/            # Colab notebook cells for one-time AWS provisioning
```

## Cost notes

- DynamoDB on-demand or provisioned within free tier (25GB + 25 RCU/WCU) —
  free tier does not expire.
- EC2 t2/t3.micro — free for 12 months from account creation only. After that,
  budget ~$7-8/mo or move ingestion to a scheduled Lambda + REST polling
  (loses some tick resolution).
- S3 — 5GB free tier, plenty for model artifacts.
