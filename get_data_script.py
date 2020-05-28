
import urllib.request
import os
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import pandas as pd


sas_url = 'https://lilablobssc.blob.core.windows.net/nacti-unzipped?St=2020-01-01T00%3A00%3A00Z&se=2034-01-01T00%3A00%3A00Z&sp=rl&sv=2019-07-07&sr=c&sig=rsgUcvoniBuVjkjzubh6gliU3XGvpE2A30Y0XPW4Vc%3D'
json_filename = r'C:\Users\Evan\PycharmProjects\BirdHouse\lila_dataset\nacti_metadata.json\nacti_metadata.json'
output_dir = r'C:\Users\Evan\PycharmProjects\BirdHouse\lila_dataset_squirrel'
meta_data_csv = r'C:\Users\Evan\PycharmProjects\BirdHouse\lila_dataset\nacti_metadata.json\nacti_metadata_csv.csv'
species_of_interest = 'squirrel'

#NATCI index the meta_data_csv

df = pd.read_csv(meta_data_csv, low_memory=False)
print(df.head(5))
squr_df =df.loc[df['family'] == 'sciuridae' ]
filename_list = squr_df['filename'].tolist()

overwrite_files = True
n_download_threads = 1
# Number of concurrent download threads (when not using AzCopy)


# %% Environment prep and derived constants

base_url = sas_url.split('?')[0]
sas_token = sas_url.split('?')[1]
os.makedirs(output_dir, exist_ok=True)
# %% Support functions

def download_image(fn):
    url = base_url + '/' + fn

    target_file = os.path.join(output_dir, fn)
    if ((not overwrite_files) and (os.path.isfile(target_file))):
        return
    # print('Downloading {} to {}'.format(url,target_file))
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    urllib.request.urlretrieve(
        url, target_file)


# %% Download those image files

    # Loop over files

if n_download_threads <= 1:

    for fn in tqdm(filename_list):
        download_image(fn)

else:
    pool = ThreadPool(n_download_threads)
    tqdm(pool.imap(download_image, filename_list), total=len(filename_list))

print('Done!')
