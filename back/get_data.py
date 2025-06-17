import os
import requests
import sys
import numpy as np
import cv2
import json
import time
import h5py
from io import BytesIO
from datetime import datetime, timedelta
from astropy.io import fits
from scipy.ndimage import zoom
from matplotlib import pyplot as plt
from dateutil import tz

# ---------- 設定 ----------
AIA_WAVELENGTHS = [
    '0094', '0131', '0171', '0193', '0211', '0304', '0335', '1600', '4500'
]
SAVE_ROOT = 'data/images'
H5_SAVE_ROOT = 'ml/datasets/all_data_hours'
XRS_PATH = 'data/xrs.json'
XRS_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"


def fetch_and_process_aia_image(wavelength, dt):
    """
    Fetch an AIA FITS image for the given wavelength and datetime,
    downsample to 256×256, then crop, flip, and restore to 256×256.
    """
    from io import BytesIO
    import requests
    from astropy.io import fits
    import numpy as np
    from scipy.ndimage import zoom

    ymd = dt.strftime('%Y%m%d')
    hour = dt.strftime('%H')
    year = dt.year
    month = dt.strftime('%m')
    day = dt.strftime('%d')

    if year >= 2023:
        url = f"https://sdo5.nascom.nasa.gov/data/aia/synoptic/{year}/{month}/{day}/H{hour}00/" \
              f"AIA{ymd}_{hour}0000_{wavelength}.fits"
    else:
        url = f"https://jsoc1.stanford.edu/data/aia/synoptic/{year}/{month}/{day}/H{hour}00/" \
              f"AIA{ymd}_{hour}00_{wavelength}.fits"

    try:
        resp = requests.get(url)
        resp.raise_for_status()

        # 1) Load FITS
        hdul = fits.open(BytesIO(resp.content))
        img = hdul[1].data.astype(np.float32)

        # 2) Downsample to 256×256
        blurred = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1.5, sigmaY=1.5)

        img256 = cv2.resize(blurred, (256, 256), interpolation=cv2.INTER_AREA)


        # 3) Crop margins
        v_crop, h_crop = 20, 15
        cropped = img256[v_crop:-v_crop, h_crop:-h_crop]

        # 4) Flip vertically (to match code2 order)
        cropped = cropped[::-1, :]

        # 5) Resize back to 256×256
        zh = img256.shape[0] / cropped.shape[0]
        zw = img256.shape[1] / cropped.shape[1]
        img_fixed = zoom(cropped, (zh, zw), order=1)

        return img_fixed.astype(np.float32)

    except Exception as e:
        print(f"❌ AIA {wavelength} fetch/process failed: {e}")
        # Fallback to a black image if anything goes wrong
        return np.zeros((256, 256), dtype=np.float32)


def download_hmi_image(dt):
    ymd = dt.strftime('%Y%m%d')
    hour = dt.strftime('%H')
    year = dt.year
    month = dt.strftime('%m')
    day = dt.strftime('%d')

    url = f"https://jsoc1.stanford.edu/data/hmi/images/{year}/{month}/{day}/{ymd}_{hour}0000_M_1k.jpg"

    try:
        resp = requests.get(url)
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE).astype(np.float32)  # raw (1024×1024)

        # 3) ダウンサンプリング to 256×256
        blurred = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1.5, sigmaY=1.5)

        img256 = cv2.resize(blurred, (256, 256), interpolation=cv2.INTER_AREA)

        # 4) 文字領域の反転コピー
        text_h = 7
        src = img256[-text_h:, :]
        img256[-text_h:, :] = np.zeros_like(src)

        return img256.astype(np.float32)

    except Exception as e:
        print(f"❌ HMI fetch/process failed: {e}")
        return np.zeros((256, 256), dtype=np.float32)


def save_png(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def save_h5(aia_images, hmi_image, dt):
    try:
        os.makedirs(H5_SAVE_ROOT, exist_ok=True)
        filename = dt.strftime("%Y%m%d_%H0000.h5")
        filepath = os.path.join(H5_SAVE_ROOT, filename)

        aia_images_fixed = []
        for img in aia_images:
            if img.ndim == 3 and img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            aia_images_fixed.append(img.astype(np.float32).reshape(256, 256))

        if hmi_image.ndim == 3 and hmi_image.shape[-1] == 3:
            hmi_image = cv2.cvtColor(hmi_image, cv2.COLOR_BGR2GRAY)
        hmi_image = hmi_image.astype(np.float32).reshape(256, 256)

        X = np.stack(aia_images_fixed + [hmi_image])  # shape: (10, 256, 256)
        timestamp = dt.strftime("%Y%m%d_%H0000").encode()

        with h5py.File(filepath, 'w') as f:
            f.create_dataset("X", data=X)
            f.create_dataset("timestamp", data=timestamp)

        print(f"✅ H5保存: {filepath}")
    except Exception as e:
        print(f"❌ H5保存失敗: {e}")


def update_xrs_json(dt):
    try:
        response = requests.get(XRS_URL)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"❌ XRS API取得失敗: {e}")
        return

    max_flux = 0
    for item in data:
        if item.get("energy") != "0.1-0.8nm":
            continue
        ts = item.get("time_tag")
        try:
            obs_time = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
            obs_time = obs_time.replace(tzinfo=tz.tzutc())
        except:
            continue

        if dt - timedelta(hours=1) < obs_time <= dt:
            try:
                flux = float(item.get("flux", 0))
                max_flux = max(max_flux, flux)
            except:
                continue

    time_str = dt.strftime('%Y%m%d%H')
    try:
        if os.path.exists(XRS_PATH):
            with open(XRS_PATH, 'r') as f:
                xrs_data = json.load(f)
        else:
            xrs_data = {}
        xrs_data[time_str] = max_flux
        with open(XRS_PATH, 'w') as f:
            json.dump(xrs_data, f, indent=2)
        print(f"✅ XRS更新: {time_str} → {max_flux}")
    except Exception as e:
        print(f"❌ xrs.jsonの更新失敗: {e}")


def main():
    if len(sys.argv) > 1:
        try:
            arg = sys.argv[1]
            dt = datetime.now(tz=tz.tzutc()).replace(minute=0, second=0, microsecond=0)
            year = dt.year  # 現在の年を取得
            month = int(arg[:2])  # 引数の最初の2文字を月として解釈
            day = int(arg[2:4])  # 次の2文字を日として解釈
            hour = int(arg[4:6])  # 最後の2文字を時として解釈
            dt = datetime(year, month, day, hour, tzinfo=tz.tzutc())
        except Exception as e:
            print(f"❌ 引数の処理に失敗しました: {e}")
            return
    else:
        now_jst = datetime.now(tz=tz.gettz('Asia/Tokyo')) - timedelta(minutes=30)
        dt = now_jst.astimezone(tz=tz.tzutc()).replace(minute=0, second=0, microsecond=0)

    # 1時間ごとに処理
    update_xrs_json(dt)

    while True:
        update_xrs_json(dt)
        update_xrs_json(dt - timedelta(hours=1))

        # H5ファイル、pngが存在するか確認
        time_str = dt.strftime('%H')
        date_str = dt.strftime('%m%d')
        h5_path = os.path.join(H5_SAVE_ROOT, dt.strftime("%Y%m%d_%H0000.h5"))
        png_paths = [
            os.path.join(SAVE_ROOT, date_str, f"{time_str}_aia_{wl}.png")
            for wl in AIA_WAVELENGTHS
        ]
        png_paths.append(os.path.join(SAVE_ROOT, date_str, f"{time_str}_hmi.png"))

        if os.path.exists(h5_path) and all(os.path.exists(png) for png in png_paths):
            print(f"✅ H5ファイルとすべてのPNGファイルが既に存在: {h5_path}")
            break

        # AIA画像とHMI画像の処理（1時間ごと）
        aia_images = []
        for wl in AIA_WAVELENGTHS:
            image_data = fetch_and_process_aia_image(wl, dt)
            aia_images.append(image_data)

            if not np.all(image_data == 0):  # 真っ黒な画像でない場合のみ保存
                image_display = np.log1p(image_data)
                image_display = cv2.normalize(image_display, None, 0, 255, cv2.NORM_MINMAX)
                image_uint8 = image_display.astype(np.uint8)
                png_path = os.path.join(SAVE_ROOT, date_str, f"{time_str}_aia_{wl}.png")
                save_png(image_uint8, png_path)
                print(f"✅ AIA保存: {png_path}")
            else:
                print(f"⚠️ AIA {wl} の画像が取得できなかったため、PNGは保存されませんでした")

        hmi_path = os.path.join(SAVE_ROOT, date_str, f"{time_str}_hmi.png")
        hmi = download_hmi_image(dt)
        if not np.all(hmi == 0):  # 真っ黒な画像でない場合のみ保存
            save_png(hmi.astype(np.uint8), hmi_path)
            print(f"✅ HMI保存: {hmi_path}")
        else:
            print(f"⚠️ HMI画像が取得できなかったため、PNGは保存されませんでした")

        save_h5(aia_images, hmi, dt)

        # 1時間前に遡る（1時間ごとの処理に変更）
        break  # 単発実行にして、GitHub Actionsで1時間ごとに呼び出す


if __name__ == '__main__':
    main()