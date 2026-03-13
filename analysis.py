import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

from cnn_randomforest import (
    load_data,
    ImageDataset,
    CNN,
    BATCH_SIZE,
    EPOCHS,
    DEVICE
)

def main():
    CSV = "Data_working_final/metadata.csv"
    IMG_DIR = "Data_working_final/images"

    Xtr, Xte, yw_tr, yw_te, yt_tr, yt_te, wc, tc = load_data(CSV, IMG_DIR)

    df = pd.read_csv(CSV)
    le_w = LabelEncoder()
    le_t = LabelEncoder()
    le_w.fit(df["Weather"])
    le_t.fit(df["Time of Day"])

    train_loader = DataLoader(
        ImageDataset(Xtr, yw_tr, yt_tr),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        ImageDataset(Xte, yw_te, yt_te),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = CNN(wc, tc).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(EPOCHS):
        model.train()
        for x, yw, yt in train_loader:
            x, yw, yt = x.to(DEVICE), yw.to(DEVICE), yt.to(DEVICE)
            ow, ot = model(x)
            loss = loss_fn(ow, yw) + loss_fn(ot, yt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    pw, pt = [], []
    with torch.no_grad():
        for x, _, _ in test_loader:
            x = x.to(DEVICE)
            ow, ot = model(x)
            pw.append(ow.argmax(1).cpu().numpy())
            pt.append(ot.argmax(1).cpu().numpy())

    pw = np.concatenate(pw)
    pt = np.concatenate(pt)

    joint_err = np.where((pw != yw_te) | (pt != yt_te))[0]

    print("Test samples:", len(yw_te))
    print("Joint errors:", len(joint_err))

    w_err = Counter(le_w.inverse_transform([yw_te[i]])[0]
                    for i in joint_err if pw[i] != yw_te[i])
    t_err = Counter(le_t.inverse_transform([yt_te[i]])[0]
                    for i in joint_err if pt[i] != yt_te[i])

    print("\nWeather errors:")
    for k, v in w_err.items():
        print(f"  {k}: {v}")

    print("\nTime-of-day errors:")
    for k, v in t_err.items():
        print(f"  {k}: {v}")

    if len(joint_err) > 0:
        idx = np.random.choice(joint_err, min(5, len(joint_err)), replace=False)
        for i in idx:
            tw = le_w.inverse_transform([yw_te[i]])[0]
            tt = le_t.inverse_transform([yt_te[i]])[0]
            pw_txt = le_w.inverse_transform([pw[i]])[0]
            pt_txt = le_t.inverse_transform([pt[i]])[0]

            plt.imshow(Xte[i])
            plt.axis("off")
            plt.title(
                f"True: {tw}, {tt}\nPred: {pw_txt}, {pt_txt}"
            )
            plt.show()

if __name__ == "__main__":
    main()
