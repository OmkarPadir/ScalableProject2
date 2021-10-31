import requests
import shutil
import os
import time

start = time.process_time()
r = requests.get("https://cs7ns1.scss.tcd.ie/index.php?download=noresume_speed&shortname=padiro")

# fetch file csv
with open("padiroImages.csv", "wb") as file:
       file.write(r.content)

captchas_dir = 'main_captchas'

# delete directory if exists, otherwise create a new one
if os.path.exists(captchas_dir):
    shutil.rmtree(captchas_dir)

os.mkdir(captchas_dir)

with open("padiroImages.csv", "r") as file:
    for line in file.readlines():
        filename = line.split(',')[0]
        r = requests.get("https://cs7ns1.scss.tcd.ie/index.php?download=noresume_speed&shortname=padiro&myfilename=" + filename)
        with open(os.path.join(captchas_dir, filename), "wb") as wFile:
            wFile.write(r.content)

print('time elapsed')
print(time.process_time() - start)
