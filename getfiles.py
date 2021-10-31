import requests
from datetime import datetime
import time


start = time.time()
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

response = requests.get("http://cs7ns1.scss.tcd.ie/index.php?download=noresume_speed&shortname=padiro")
print(response)

with open("response.txt", "w") as f:
    f.write(response.text)
#    print(f)

#with open('response.txt') as resp_file:
#    for line in resp_file:
#        print (line)

import csv

with open('response.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        # print(str(row[0]))
        # line_count = line_count+1

        response2 = requests.get("http://cs7ns1.scss.tcd.ie/index.php?download=noresume_speed&shortname=padiro&myfilename="+str(row[0]))

        file = open("CaptchaTests2/"+str(row[0]), "wb")
        file.write(response2.content)
        file.close()

       #with open("CaptchaTests/"+str(row[0]), "w") as f:
         #   f.write(response2)
    # print("Line Count: "+str(line_count))


print("End Time =", datetime.now().strftime("%H:%M:%S"))
print("Total Time Taken (mins): ", (time.time() - start)/60)