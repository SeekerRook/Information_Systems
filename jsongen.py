import random
import json
import argparse


col_range_max = 25
col_range_min  = 4



ds_types = {
"huge" : {
    "probability":0.0,
    "multiplier":1,
    "unit":"GB"
    },
"small" : {
    "probability":0.3,
    "multiplier":1,
    "unit":"MB",
    },
"medium" : {
    "probability":0.5,
    "multiplier":10,
    "unit":"MB"
    },
"big" : {
    "probability":0.2,
    "multiplier":100,
    "unit":"MB"
    },
    
}

column_types = {
"int" : 1.5,
"float" : 1,

"word" : 0.2,
"bool" : 1.2,
"uniform" : 0.5
}

# for type in ds_types:
#     for _ in range(N*type["probability"]):
#         size = {}



# print (f"{coln} columns:")
types = []
_ = [types.extend( [t] * int(column_types[t]*5)) for t in column_types]

types_no_uniform  = list(filter(lambda x: x != "uniform", types))

# print (types_no_uniform)
def getcols():
    coln  = random.randint(col_range_min,col_range_max)

    columns = []
    for _ in range(coln):
        t = random.choice(types) 
        if t == "uniform" :
            columns = [random.choice(types_no_uniform)]*coln
            break
        else : columns.append(t)
    res = {}
    for i in set(columns):
        res[i] = columns.count(i)
        try:
            res["bool"]+=1
        except: 
            res["bool"]=1

    return res  

# print(getcols())
def make(N,out):
    data  = []

    for t in ds_types:
        idx = 0
        prob = ds_types[t]["probability"] 
        mul = ds_types[t]["multiplier"] 
        unit = ds_types[t]["unit"] 
        for ds in range(int(max(N*prob,1))): 
            idx+=1
            datum = getcols()
            datum["size"] = f"{random.randint(1,10)*mul}{unit}"
            datum["out"] = f"gen{t}{idx}.csv"
            data.append(datum)

    json.dump(data,open(out,'w'))        
         

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", "--number", default=10, type = int,required=False, help="# of datasets")
    ap.add_argument("-o", "--out", required=True, help="output file name")
    ap.add_argument("-M", "--max", required=False, help="max # of Columns")
    ap.add_argument("-m", "--min", required=False, help="minimum # of Columns")
    args = vars(ap.parse_args())
    
    if args["max"] is not None:
        col_range_max = args["max"]
    if args["min"] is not None: 

        col_range_max = args["min"] 

    make(N = args["number"],out = args["out"])    