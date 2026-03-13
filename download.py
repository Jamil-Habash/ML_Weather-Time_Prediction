import os
import requests
import pandas as pd
from PIL import Image


def download_images_with_metadata(
    in_csv,
    out_dir,
    image_url_col="Image URL",
    label_cols=('Description', 'Country', 'Weather', 'Time of Day', 'Season', 'Activity', 'Mood/Emotion'),
    img_subdir="images",
    meta_filename="metadata.csv",
    timeout=10
):

    img_dir = os.path.join(out_dir, img_subdir)
    os.makedirs(img_dir, exist_ok=True)

    df = pd.read_csv(in_csv)

    rows_out = []
    global_img_id = 1
    downloaded_count = 0
    failed_count = 0

    for _, row in df.iterrows():
        img_url = str(row.get(image_url_col, "")).strip()
        filename = f"{global_img_id}.jpg"
        img_path = os.path.join(img_dir, filename)

        downloaded = False

        if img_url and img_url.lower().startswith("http"):
            try:
                r = requests.get(img_url, timeout=timeout, stream=True)

                content_type = r.headers.get("Content-Type", "")
                if r.status_code != 200 or not content_type.startswith("image/"):
                    failed_count += 1
                    global_img_id += 1
                    continue

                with open(img_path, "wb") as f:
                    for chunk in r.iter_content(1024):
                        if chunk:
                            f.write(chunk)

                if os.path.getsize(img_path) < 1024:
                    os.remove(img_path)
                    failed_count += 1
                    global_img_id += 1
                    continue

                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception:
                    os.remove(img_path)
                    failed_count += 1
                    global_img_id += 1
                    continue

                downloaded = True
                downloaded_count += 1

            except Exception:
                if os.path.exists(img_path):
                    os.remove(img_path)
                failed_count += 1
        else:
            failed_count += 1

        if downloaded:
            row_out = {"image_filename": filename}
            for col in label_cols:
                row_out[col] = row[col] if col in row else None
            rows_out.append(row_out)

        global_img_id += 1

    meta_path = os.path.join(out_dir, meta_filename)
    pd.DataFrame(rows_out).to_csv(meta_path, index=False)

    return {
        "downloaded": downloaded_count,
        "failed": failed_count,
        "metadata_csv": meta_path
    }