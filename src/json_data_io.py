import json, os, pickle, types


def convert_pkl_to_json():
    for root, subdirs, files in os.walk("../"):
        for f in files:
            splt = f.split(".")
            if len(splt) == 2 and splt[1] == "pkl":
                old = os.path.join(root,f)
                new = os.path.join(root,splt[0]+".json")
                print old
                print new
                # with open(old) as data:
                #     if type(data) is types.InstanceType:
                #         with open(new,"w") as out:
                #             json.dump(data.__dict__, out, separators=(',',':'))
                #     else:
                #         with open(new, "w") as out:
                #             json.dump(data, out, separators=(',', ':'))
                # print pickle.load(data)



convert_pkl_to_json()
# s = json.dumps([1,2,3,{'4': 5, '6': 7}], separators=(',',':'))
# print s
# d = json.loads(s)
#
# print d[3]['4']