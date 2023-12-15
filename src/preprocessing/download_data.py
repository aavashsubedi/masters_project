import os
import tarfile

from tqdm import tqdm
import urllib.request

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    # with DownloadProgressBar(unit='B', unit_scale=True,
    #                         miniters=1, desc=url.split('/')[-1]) as t:
    urllib.request.urlretrieve(url, filename=output_path)

def download_warcraft_data():
    # INSTRUCTIONS: Go to first URL. Start downloading warcraft_maps.tar.gz to your device. While it downloads, right-click and 
    # copy the download link. Paste it into the warcraft_url variable below. Then run this script.
    data_dir = "data/"
    warcraft_url = "https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.YJCQ5S#"
    warcraft_url = "https://dev-edmond-objstor-hdd.s3.gwdg.de/10.17617/3.YJCQ5S/17f0bf4420a-86466ec10fec?response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27warcraft_maps.tar.gz&response-content-type=application%2Fgzip&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20231214T003125Z&X-Amz-SignedHeaders=host&X-Amz-Expires=86400&X-Amz-Credential=W7RIGMB4SLQMPMLDY4FF%2F20231214%2Fdataverse%2Fs3%2Faws4_request&X-Amz-Signature=8b5a35c8605b16126994fea81c1445b64d202c1b6997f5022e1ac45c09313dc7"
    data_path = "data/warcraft_shortest_path.tar.gz"

    if not os.path.exists(data_path):
        print("Downloading data")
        download_url(warcraft_url, data_path)

    print("Extracting data")
    _file = tarfile.open(data_path)
    _file.extractall(data_dir)
    _file.close()
    print("Done!")
    return None
download_warcraft_data()