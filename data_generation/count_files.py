import os
import sys

directory = "/datas/sandbox/gmoussa/ground_truth/nyu/train"

file_count = 0

for root, dirs, files in os.walk(directory):
    file_count += len(files)

print(f"Number of files: {file_count}")
print(f"Current indecies: {file_count/2}")
