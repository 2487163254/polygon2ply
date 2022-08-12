import yaml

with open('result/result.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.Loader)

goodlist = data['goodlist']
badlist = data['badlist']
# print(len(goodlist), len(badlist))