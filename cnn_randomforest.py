import os
import copy
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score


IMAGE_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(csv_path, image_dir):
    df = pd.read_csv(csv_path)

    images, weather, time = [], [], []

    for _, row in df.iterrows():
        path = os.path.join(image_dir, row["image_filename"])
        if os.path.exists(path):
            img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
            images.append(np.array(img))
            weather.append(row["Weather"])
            time.append(row["Time of Day"])

    images = np.array(images, dtype=np.float32) / 255.0

    le_w = LabelEncoder()
    le_t = LabelEncoder()

    weather = le_w.fit_transform(weather)
    time = le_t.fit_transform(time)

    Xtr, Xte, yw_tr, yw_te, yt_tr, yt_te = train_test_split(images, weather, time, test_size=0.2, random_state=42
    )

    return Xtr, Xte, yw_tr, yw_te, yt_tr, yt_te, len(le_w.classes_), len(le_t.classes_)


class ImageDataset(Dataset):
    def __init__(self, X, yw, yt):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
        self.yw = torch.tensor(yw, dtype=torch.long)
        self.yt = torch.tensor(yt, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.yw[i], self.yt[i]


class CNN(nn.Module):
    def __init__(self, wc, tc):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.weather = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, wc)
        )
        self.time = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, tc)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.weather(x), self.time(x)


def train_eval_cnn(Xtr, Xte, yw_tr, yw_te, yt_tr, yt_te, wc, tc, lr):
    train_loader = DataLoader(ImageDataset(Xtr, yw_tr, yt_tr), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(ImageDataset(Xte, yw_te, yt_te), batch_size=BATCH_SIZE)

    model = CNN(wc, tc).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0
    best_weights = copy.deepcopy(model.state_dict())
    patience = 5
    wait = 0

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
        correct, total = 0, 0
        with torch.no_grad():
            for x, yw, yt in test_loader:
                x, yw, yt = x.to(DEVICE), yw.to(DEVICE), yt.to(DEVICE)
                ow, ot = model(x)
                pw, pt = ow.argmax(1), ot.argmax(1)
                correct += ((pw == yw) & (pt == yt)).sum().item()
                total += yw.size(0)

        acc = correct / total

        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_weights)
    model.eval()

    pw, pt = [], []
    with torch.no_grad():
        for x, _, _ in test_loader:
            ow, ot = model(x.to(DEVICE))
            pw.append(ow.argmax(1).cpu().numpy())
            pt.append(ot.argmax(1).cpu().numpy())

    pw = np.concatenate(pw)
    pt = np.concatenate(pt)

    w_acc = (pw == yw_te).mean()
    t_acc = (pt == yt_te).mean()
    joint_acc = ((pw == yw_te) & (pt == yt_te)).mean()

    w_f1 = f1_score(yw_te, pw, average="macro")
    t_f1 = f1_score(yt_te, pt, average="macro")

    print("\n================ CNN Results ================")
    print(f"Weather Accuracy : {w_acc:.4f}")
    print(f"Time Accuracy    : {t_acc:.4f}")
    print(f"Joint Accuracy   : {joint_acc:.4f}")
    print(f"Weather F1-score : {w_f1:.4f}")
    print(f"Time F1-score    : {t_f1:.4f}")


def train_eval_rf(Xtr, Xte, yw_tr, yw_te, yt_tr, yt_te, t_count):
    Xtr = Xtr.reshape(len(Xtr), -1)
    Xte = Xte.reshape(len(Xte), -1)

    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=t_count, n_jobs=-1, random_state=42))

    model.fit(Xtr, np.column_stack((yw_tr, yt_tr)))
    preds = model.predict(Xte)

    pw, pt = preds[:, 0], preds[:, 1]

    w_acc = (pw == yw_te).mean()
    t_acc = (pt == yt_te).mean()
    joint_acc = ((pw == yw_te) & (pt == yt_te)).mean()

    w_f1 = f1_score(yw_te, pw, average="macro")
    t_f1 = f1_score(yt_te, pt, average="macro")

    print("\n============== Random Forest Results ==============")
    print(f"Weather Accuracy : {w_acc:.4f}")
    print(f"Time Accuracy    : {t_acc:.4f}")
    print(f"Joint Accuracy   : {joint_acc:.4f}")
    print(f"Weather F1-score : {w_f1:.4f}")
    print(f"Time F1-score    : {t_f1:.4f}")


def main(lr, t_count):
    CSV = "Data_working_final/metadata.csv"
    IMG_DIR = "Data_working_final/images"

    Xtr, Xte, yw_tr, yw_te, yt_tr, yt_te, wc, tc = load_data(CSV, IMG_DIR)

    train_eval_cnn(Xtr, Xte, yw_tr, yw_te, yt_tr, yt_te, wc, tc, lr)
    train_eval_rf(Xtr, Xte, yw_tr, yw_te, yt_tr, yt_te, t_count)


if __name__ == "__main__":
    main(0.001, 100)