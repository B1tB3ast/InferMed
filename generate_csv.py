import csv
import re

# Raw training log
log = """
Labeled] Epoch 1/50, Loss: 0.2769
[Labeled] Epoch 2/50, Loss: 0.2386
[Labeled] Epoch 3/50, Loss: 0.2253
[Labeled] Epoch 4/50, Loss: 0.2201
[Labeled] Epoch 5/50, Loss: 0.2132
[Labeled] Epoch 6/50, Loss: 0.2101
[Labeled] Epoch 7/50, Loss: 0.2044
[Labeled] Epoch 8/50, Loss: 0.1998
[Labeled] Epoch 9/50, Loss: 0.1962
[Labeled] Epoch 10/50, Loss: 0.1901
Checkpoint saved: model_epoch_10.pth
[Labeled] Epoch 11/50, Loss: 0.1843
[Labeled] Epoch 12/50, Loss: 0.1810
[Labeled] Epoch 13/50, Loss: 0.1755
[Labeled] Epoch 14/50, Loss: 0.1703
[Labeled] Epoch 15/50, Loss: 0.1650
[Labeled] Epoch 16/50, Loss: 0.1615
[Labeled] Epoch 17/50, Loss: 0.1571
[Labeled] Epoch 18/50, Loss: 0.1520
[Labeled] Epoch 19/50, Loss: 0.1467
[Labeled] Epoch 20/50, Loss: 0.1430
Checkpoint saved: model_epoch_20.pth
[Labeled] Epoch 21/50, Loss: 0.1349
[Labeled] Epoch 22/50, Loss: 0.1345
[Labeled] Epoch 23/50, Loss: 0.1288
[Labeled] Epoch 24/50, Loss: 0.1272
[Labeled] Epoch 25/50, Loss: 0.1221
[Labeled] Epoch 26/50, Loss: 0.1151
[Labeled] Epoch 27/50, Loss: 0.1116
[Labeled] Epoch 28/50, Loss: 0.1095
[Labeled] Epoch 29/50, Loss: 0.1062
[Labeled] Epoch 30/50, Loss: 0.1039
Checkpoint saved: model_epoch_30.pth
[Labeled] Epoch 31/50, Loss: 0.0964
[Labeled] Epoch 32/50, Loss: 0.0964
[Labeled] Epoch 33/50, Loss: 0.0925
[Labeled] Epoch 34/50, Loss: 0.0880
[Labeled] Epoch 35/50, Loss: 0.0877
[Labeled] Epoch 36/50, Loss: 0.0867
[Labeled] Epoch 37/50, Loss: 0.0793
[Labeled] Epoch 38/50, Loss: 0.0770
[Labeled] Epoch 39/50, Loss: 0.0759
[Labeled] Epoch 40/50, Loss: 0.0723
Checkpoint saved: model_epoch_40.pth
[Labeled] Epoch 41/50, Loss: 0.0704
[Labeled] Epoch 42/50, Loss: 0.0690
[Labeled] Epoch 43/50, Loss: 0.0666
[Labeled] Epoch 44/50, Loss: 0.0671
[Labeled] Epoch 45/50, Loss: 0.0617
[Labeled] Epoch 46/50, Loss: 0.0611
[Labeled] Epoch 47/50, Loss: 0.0589
[Labeled] Epoch 48/50, Loss: 0.0581
[Labeled] Epoch 49/50, Loss: 0.0536
[Labeled] Epoch 50/50, Loss: 0.0515
Checkpoint saved: model_epoch_50.pth
🔹 Step 2: Generating pseudo-labels...
Generated 0 pseudo-labeled samples with confidence > 0.9
🔹 Step 3: Retraining on labeled + pseudo-labeled data...
[Retrain] Epoch 1/50, Loss: 0.0551
[Retrain] Epoch 2/50, Loss: 0.0494
[Retrain] Epoch 3/50, Loss: 0.0481
[Retrain] Epoch 4/50, Loss: 0.0485
[Retrain] Epoch 5/50, Loss: 0.0487
[Retrain] Epoch 6/50, Loss: 0.0443
[Retrain] Epoch 7/50, Loss: 0.0485
[Retrain] Epoch 8/50, Loss: 0.0428
[Retrain] Epoch 9/50, Loss: 0.0415
[Retrain] Epoch 10/50, Loss: 0.0560
Checkpoint saved: model_epoch_10.pth
[Retrain] Epoch 11/50, Loss: 0.0405
[Retrain] Epoch 12/50, Loss: 0.0399
[Retrain] Epoch 13/50, Loss: 0.0391
[Retrain] Epoch 14/50, Loss: 0.0408
[Retrain] Epoch 15/50, Loss: 0.0390
[Retrain] Epoch 16/50, Loss: 0.0373
[Retrain] Epoch 17/50, Loss: 0.0347
[Retrain] Epoch 18/50, Loss: 0.0340
[Retrain] Epoch 19/50, Loss: 0.0322
[Retrain] Epoch 20/50, Loss: 0.0319
Checkpoint saved: model_epoch_20.pth
[Retrain] Epoch 21/50, Loss: 0.0347
[Retrain] Epoch 22/50, Loss: 0.0351
[Retrain] Epoch 23/50, Loss: 0.0349
[Retrain] Epoch 24/50, Loss: 0.0294
[Retrain] Epoch 25/50, Loss: 0.0288
[Retrain] Epoch 26/50, Loss: 0.0281
[Retrain] Epoch 27/50, Loss: 0.0271
[Retrain] Epoch 28/50, Loss: 0.0344
[Retrain] Epoch 29/50, Loss: 0.0260
[Retrain] Epoch 30/50, Loss: 0.0261
Checkpoint saved: model_epoch_30.pth
[Retrain] Epoch 31/50, Loss: 0.0256
[Retrain] Epoch 32/50, Loss: 0.0258
[Retrain] Epoch 33/50, Loss: 0.0276
[Retrain] Epoch 34/50, Loss: 0.0280
[Retrain] Epoch 35/50, Loss: 0.0241
[Retrain] Epoch 36/50, Loss: 0.0205
[Retrain] Epoch 37/50, Loss: 0.0327
[Retrain] Epoch 38/50, Loss: 0.0225
[Retrain] Epoch 39/50, Loss: 0.0269
[Retrain] Epoch 40/50, Loss: 0.0258
Checkpoint saved: model_epoch_40.pth
[Retrain] Epoch 41/50, Loss: 0.0159
[Retrain] Epoch 42/50, Loss: 0.0275
[Retrain] Epoch 43/50, Loss: 0.0206
[Retrain] Epoch 44/50, Loss: 0.0204
[Retrain] Epoch 45/50, Loss: 0.0230
[Retrain] Epoch 46/50, Loss: 0.0213
[Retrain] Epoch 47/50, Loss: 0.0237
[Retrain] Epoch 48/50, Loss: 0.0226
[Retrain] Epoch 49/50, Loss: 0.0186
[Retrain] Epoch 50/50, Loss: 0.0187
""".strip()  # Truncated for clarity — use full log

# Define regex to extract type, epoch, and loss
pattern = r"\[(Labeled|Retrain)\] Epoch (\d+)/50, Loss: ([\d.]+)"

# Find all matches
entries = re.findall(pattern, log)

# Save to CSV
csv_filename = "training_log_semi_supervised.csv"
with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["type", "epoch", "loss"])  # Header
    for t, e, l in entries:
        writer.writerow([t, int(e), float(l)])

print(f"Saved log to {csv_filename}")
