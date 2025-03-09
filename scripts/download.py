from os import makedirs
import wget
import gdown
import zipfile

TRAIN_URL = "https://drive.google.com/file/d/1Bug0xjN-qpWh2ZYiRmwGW4BT_YjsumSf/view?usp=drive_link"
TEST_URL = "https://storage.codenrock.com/companies/codenrock-13/contests/kryptonite-ml-challenge/test_public.zip"


def get_data(url: str, filename: str, gd: bool = False) -> None:
    makedirs("./data/zipped", exist_ok=True)
    if not gd:
        out_filename = wget.download(url, out=f"./data/zipped/{filename}.zip")
    else:
        out_filename = f"./data/zipped/{filename}.zip"
        gdown.download(url, out_filename, fuzzy=True)

    with zipfile.ZipFile(out_filename, "r") as zip_ref:
        zip_ref.extractall(f"./data")


def main() -> None:
    get_data(TRAIN_URL, "train", True)
    get_data(TEST_URL, "test")
