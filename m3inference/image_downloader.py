import pandas as pd 
import os 
import requests 
import shutil
import concurrent.futures
from tqdm import tqdm


class ProfileImageDownloader:

    def __init__(self,json_file_path,save_path='.'):

        def return_name_file(n):
            if str(n) == 'None':
                return 'default.png'
            return n.split('/')[-1] 

        if not os.path.exists(os.path.join(save_path,'profile_image')):
            os.makedirs(os.path.join(save_path,'profile_image'))

        self.save_path = os.path.join(save_path,'profile_image')

        self.base_url = 'https://f002.backblazeb2.com/file/all-gather-media/'
        self.df = pd.read_json(json_file_path)

        unique_urls = self.df[self.df['img_path'].isna() == False]['img_path'].unique()
        self.urls = self.base_url + pd.Series(unique_urls)
        self.urls = self.urls.tolist()

        
        self.df['img_path'] = self.df['img_path'].apply(return_name_file)
        

    def save_df_to_json(self,json_file_path):
        self.df.to_json(json_file_path,orient='records')    

    def download(self,url):
        file_name = url.split('/')[-1]
        resp = requests.get(url,stream=True)

        if resp.status_code == 200:
            resp.raw.decode_content = True
            with open(os.path.join(self.save_path,file_name),'wb') as f:
                shutil.copyfileobj(resp.raw, f)
            return True 
        return False



import argparse

parser = argparse.ArgumentParser(description='M3 training')

parser.add_argument('--file_path',type=str)
parser.add_argument('--save_path',type=str,default='.',required=False)
parser.add_argument('--num_workers',type=int,default=50,required=False)

args = parser.parse_args()

if __name__ == "__main__":

    data = ProfileImageDownloader(args.file_path,args.save_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_image = {executor.submit(data.download,url):url for url in data.urls}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_image)):
            url = future_to_image[future]

            try:
                result = future.result()
            except Exception as e:
                print("{} generates an exception {}".format(url,e))
