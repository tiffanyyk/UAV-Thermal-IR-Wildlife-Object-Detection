import json
from collections import Counter

filename = "TrainSimulation.json" #Change as needed

with open(filename) as jsonFile:
    jsonObject = json.load(jsonFile)
    jsonFile.close()
    
ann = jsonObject["annotations"]
c = Counter(img['category_id'] for img in ann)
print(c)