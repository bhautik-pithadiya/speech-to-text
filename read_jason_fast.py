import json

with open('res.json') as f:
    data = json.load(f)

print(data[0])