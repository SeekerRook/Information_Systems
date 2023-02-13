import pandas as pd
import numpy as np
# import gc
import datetime


GB = 1000000000
MB = 1000000
strsize=8

def heuristic (gb,y, int_columns = 0, str_columns = 0, float_columns = 0, bool_columns= 0, word_columns = 0):
    y = int_columns+ str_columns*strsize + 
    return (gb -2*y)//(2+2*y)


def dataset(rows, int_columns = 0, str_columns = 0, float_columns = 0, bool_columns= 0, word_columns = 0):
    import numpy as np
    a = []
    rng =   rng = np.random.default_rng()

    if int_columns >0 :
    
        a.append(rng.integers(low=10,high=1000,size = (rows,int_columns)).T.astype(object))
    if bool_columns >0 :
        a.append(rng.integers(low=0,high=2,size = (rows,bool_columns)).T.astype(object))
    res = a[0]

    if float_columns >0 :
    
        a.append(rng.random(size = (rows,int_columns)).T.astype(object))
    if str_columns >0 :
        alphabet = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        np_alphabet = np.array(alphabet)
        np_codes = np.random.choice(np_alphabet, [str_columns*rows, strsize])
        codes = ["".join(np_codes[i]) for i in range(len(np_codes))]
        a.append(np.array(codes).reshape(rows,str_columns).T.astype(object))
    if word_columns >0 :
        import json
        alphabet = json.load(open("wordlist.json"))
        np_alphabet = np.array(alphabet)
        np_codes = np.random.choice(np_alphabet, [rows,word_columns])
        a.append(np_codes.T.astype(object))


    return np.concatenate(a).T

def generate(out, x, int_columns = 0, str_columns = 0, float_columns = 0, bool_columns= 0, word_columns = 0):
    columns = int_columns + float_columns + bool_columns + str_columns + word_columns
    start =datetime.datetime.now()
    print("Generating Data....")
    a = dataset(x, int_columns = int_columns, str_columns = str_columns, float_columns = float_columns, bool_columns= bool_columns, word_columns = word_columns)

    endg =datetime.datetime.now()
    print("Done")
    c = endg-start
    days, hours, minutes, seconds = int(c.days), int(c.seconds // 3600), int(c.seconds % 3600 / 60.0), int(c.seconds % 60.0)
    print (f"Generating time :{days}d:{hours}h:{minutes}m:{seconds}s")

    print("Saving Data....")

    pd.DataFrame(a).to_csv(out,index=False)



    ends =datetime.datetime.now()
    print("Done....")
    c = ends-endg
    days, hours, minutes, seconds = int(c.days), int(c.seconds // 3600), int(c.seconds % 3600 / 60.0), int(c.seconds % 60.0)
    print (f"Saving time :{days}d:{hours}h:{minutes}m:{seconds}s")


    c = ends-start
    days, hours, minutes, seconds = int(c.days), int(c.seconds // 3600), int(c.seconds % 3600 / 60.0), int(c.seconds % 60.0)
    print (f"Total time :{days}d:{hours}h:{minutes}m:{seconds}s")

generate("gen1.csv",500*MB,int_columns=2,bool_columns=2,str_columns=1,word_columns=1)

def merge(csvlist):
    import pandas as pd
    print(f"Merging {','.join(csvlist)}")
  
    # merging two csv files
    df = pd.concat(
        map(pd.read_csv, csvlist), ignore_index=True)
    print(df) 
    df.to_csv('gen1234.csv',index=False)
merge(['gen12.csv','gen3.csv','gen3.csv'])

#autosplitter
