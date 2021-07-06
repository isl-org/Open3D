import os

load1, load5, load15 = os.getloadavg()

print("Load average over the last 1 minute:", load1)
print("Load average over the last 5 minute:", load5)
print("Load average over the last 15 minute:", load15)
