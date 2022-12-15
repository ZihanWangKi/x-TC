import yaml
apiData = {
    "page": 1,
    "data": {
      "id": [1,2],
      "name": 5e-6
   },
}
with open('1.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(data=apiData, stream=f, allow_unicode=True, default_flow_style=None)