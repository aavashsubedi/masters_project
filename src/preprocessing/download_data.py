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

    data_dir = "/workspaces/masters_project/data/"
    warcraft_url = "https://edmond.mpdl.mpg.de/imeji/exportServlet?format=file&id=http://edmond.mpdl.mpg.de/imeji/item/PPjoEvUh9_PVTPSD"
    warcraft_url = "https://dev-edmond-objstor-hdd.s3.gwdg.de/10.17617/3.YJCQ5S/17f0bf2e721-081390eaa903?response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27mnist_matching.tar.gz&response-content-type=application%2Fgzip&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230928T150106Z&X-Amz-SignedHeaders=host&X-Amz-Expires=86400&X-Amz-Credential=W7RIGMB4SLQMPMLDY4FF%2F20230928%2Fdataverse%2Fs3%2Faws4_request&X-Amz-Signature=536af648b10de22e339f39ea2235cb5b2c414102c81c60b266a37b2bad6945ee"
    warcraft_url = "https://dev-edmond-objstor-hdd.s3.gwdg.de/10.17617/3.YJCQ5S/17f0bf4420a-86466ec10fec?response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27warcraft_maps.tar.gz&response-content-type=application%2Fgzip&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230928T145858Z&X-Amz-SignedHeaders=host&X-Amz-Expires=86400&X-Amz-Credential=W7RIGMB4SLQMPMLDY4FF%2F20230928%2Fdataverse%2Fs3%2Faws4_request&X-Amz-Signature=8cea79707995047ea90f2ae71c0a39f9e927f662707aec0f020775b2ff483040"
    data_path = "/workspaces/masters_project/data/warcraft_shortest_path.tar.gz"

    if not os.path.exists(data_path):
        print("Downloading data")
        download_url(warcraft_url, data_path)
    import pdb; pdb.set_trace()
    print("Extracting data")
    _file = tarfile.open(data_path)
    _file.extractall(data_dir)
    _file.close()
    print("Done!")
    return None
download_warcraft_data()