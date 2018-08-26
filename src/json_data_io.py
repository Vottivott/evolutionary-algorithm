import json



s = json.dumps([1,2,3,{'4': 5, '6': 7}], separators=(',',':'))
print s
d = json.loads(s)

print d[3]['4']