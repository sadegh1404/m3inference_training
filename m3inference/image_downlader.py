import pandas as pd 
import os 
import requests 
import shutil
import concurrent.futures



class ProfileImageDowloader:

    def __init__(self,json_file_path,save_path='.'):

        def return_name_file(n):
            if str(n) == 'None':
                return 'default.png'
            return n.split('/')[-1] 

        if not os.path.exists(os.path.join(save_path,'profile_image')):
            os.mkdir(os.path.join(save_path,'profile_image'))

        self.save_path = os.path.join(save_path,'profile_image')

        self.base_url = 'https://f002.backblazeb2.com/file/all-gather-media/'
        self.df = pd.read_json(json_file_path)

        unique_urls = self.df[self.df['img_path'].isna() == False]['img_path'].unique()
        self.urls = self.base_url + pd.Series(unique_urls)
        self.urls = self.urls.tolist()

        
        self.df['img_path'] = self.df['img_path'].apply(return_name_file)
        
        
    def download(self,url):
        file_name = url.split('/')[-1]
        resp = requests.get(url,stream=True)

        if resp.status_code == 200:
            resp.raw.decode_content = True
            with open(os.path.join(self.save_path,file_name),'wb') as f:
                shutil.copyfileobj(resp.raw, f)
            return True 
        return False



if __name__ == "__main__":

    data = ProfileImageDowloader('balance_m3_training_data.json')
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for idx,url in enumerate(data.urls):
            print('start ',idx)
            executor.submit(data.download,url)