import requests
from datetime import datetime
import time
import shutil
import os

start = time.time()
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

response = requests.get("http://cs7ns1.scss.tcd.ie/index.php?download=noresume_speed&shortname=padiro")
print(response)

with open("response.txt", "w") as f:
    f.write(response.text)

import csv

Dir = "CaptchaTests2"
if os.path.exists(Dir):
    shutil.rmtree(Dir)

os.mkdir(Dir)

with open('response.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        # print(str(row[0]))
        # line_count = line_count+1

        response2 = requests.get("http://cs7ns1.scss.tcd.ie/index.php?download=noresume_speed&shortname=padiro&myfilename="+str(row[0]))

        file = open(Dir+"/"+str(row[0]), "wb")
        file.write(response2.content)
        file.close()


print("End Time =", datetime.now().strftime("%H:%M:%S"))
print("Total Time Taken (mins): ", (time.time() - start)/60)