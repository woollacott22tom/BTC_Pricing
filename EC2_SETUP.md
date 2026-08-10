# EC2 box setup (after AWS resources are provisioned via Colab)

```bash
# 1. Clone your repo
git clone https://github.com/YOUR_USERNAME/btc-kalshi-predictor.git
cd btc-kalshi-predictor

# 2. Install dependencies
sudo apt update && sudo apt install -y python3-pip
pip3 install -r requirements.txt --break-system-packages

# 3. Edit the systemd unit files: set MODEL_BUCKET and WorkingDirectory to match your paths
sudo cp aws_setup/systemd/btc-ingestion.service /etc/systemd/system/
sudo cp aws_setup/systemd/btc-serving.service /etc/systemd/system/
sudo systemctl daemon-reload

# 4. Start ingestion now -- let this run for 1-2 weeks before training anything
sudo systemctl enable --now btc-ingestion

# 5. Check it's actually writing data
sudo journalctl -u btc-ingestion -f

# 6. Once you have >= 200 closed windows in DynamoDB (btc_windows table),
#    train models manually the first time:
python3 models/train.py

# 7. Then start serving (uses whatever models exist in models/artifacts/)
sudo systemctl enable --now btc-serving

# 8. Set up weekly retraining via cron
crontab -e
# add this line (retrains every Sunday 3am UTC):
# 0 3 * * 0 cd /home/ubuntu/btc-kalshi-predictor && /usr/bin/python3 models/train.py >> /home/ubuntu/train.log 2>&1
```

## Verifying the instance profile is attached

```bash
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/
# should print: btc-kalshi-ec2-role
```

If this returns nothing, the instance profile wasn't attached at launch --
you can attach it after the fact via EC2 console: Actions > Security >
Modify IAM role.

## Dashboard CORS

Once your GitHub Pages / static dashboard has a real domain, tighten
`serving/app.py`'s `allow_origins=["*"]` to that exact origin.
