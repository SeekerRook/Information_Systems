import pandas as pd
import numpy as np
# import gc
import datetime
from os import system , path


GB = 1000000000
MB = 1000000
B = 1
KB = 1000
TB = 1000000000000
rename = True
strsize=8
fls = 50
flr = 25
threshold = 1*GB

def printtime(c):
    days, hours, minutes, seconds = int(c.days), int(c.seconds // 3600), int(c.seconds % 3600 / 60.0), int(c.seconds % 60.0)
    print (f"{str(days)+' days' if days != 0 else ''}{str(hours)+' hours' if hours != 0 else ''}{str(minutes)+' minutes' if minutes != 0 else ''}{seconds} seconds")

def heuristic (gb, int_columns = 0, str_columns = 0, float_columns = 0, bool_columns= 0, word_columns = 0):
    # ys = int_columns+ str_columns*strsize + word_columns*10 + bool_columns + float_columns
    columns = int_columns+ str_columns + word_columns + bool_columns + float_columns
    y =  2*int_columns + str_columns*strsize + word_columns*6 + bool_columns + float_columns*16 + columns # +1 from \n cancels with -1 from one less comma

    return gb//y 


def dataset(rows, int_columns = 0, str_columns = 0, float_columns = 0, bool_columns= 0, word_columns = 0):
    import numpy as np
    a = []
    rng =   rng = np.random.default_rng()

    if int_columns >0 :
    
        a.append(rng.integers(low=10,high=1000,size = (rows,int_columns)).T.astype(object))
    if bool_columns >0 :
        a.append(rng.integers(low=0,high=2,size = (rows,bool_columns)).T.astype(object))


    if float_columns >0 :
        center = np.random.randint(-1*fls,fls)
        a.append(rng.uniform(center-flr,center+flr,size = (rows,float_columns)).T.astype(object))

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

def _generate(ds, int_columns = 0, str_columns = 0, float_columns = 0, bool_columns= 0, word_columns = 0,out="gen.csv"):
    if rename:
        while (path.exists(out)):
                out = out.replace('.','_new.')


    # columns = int_columns + float_columns + bool_columns + str_columns + word_columns
    x = heuristic(ds,int_columns , str_columns , float_columns , bool_columns, word_columns )
    
    
    start =datetime.datetime.now()
    print("Generating Data....")
    a = dataset(x, int_columns = int_columns, str_columns = str_columns, float_columns = float_columns, bool_columns= bool_columns, word_columns = word_columns)

    print("Done")


    print("Saving Data.... ")

    pd.DataFrame(a).to_csv(out,index=False,header=False)



    ends =datetime.datetime.now()
    print("Done")




    c = ends-start
    printtime(c)
    return out



def merge(csvlist,out="gen.csv"):
    from os import system
    start =datetime.datetime.now()
    print("Merging temp files....")

    if rename:
        while (path.exists(out)):
            out = out.replace('.','_new.')
    else : system(f" echo > {out}")
    for i in csvlist:
        system(f"cat {i}>>{out}")
    system (f"rm {' '.join(csvlist)}")
    ends =datetime.datetime.now()

    print("Done")
    c = ends-start
    printtime(c)
    
    return out


def generate(ds, int_columns = 0, str_columns = 0, float_columns = 0, bool_columns= 0, word_columns = 0,out="gen.csv"):
    start =datetime.datetime.now()

    if ds < threshold:
        out = _generate(ds, int_columns = int_columns , str_columns = str_columns , float_columns = float_columns , bool_columns= bool_columns, word_columns = word_columns ,out= out)
    else: 
        print ("Size too big. Using Temp files.")
        files = []
        for i in range(ds//threshold):
            print(f"\nfile {i} of {ds//threshold+1}")
            files.append(_generate(threshold, int_columns = int_columns , str_columns = str_columns , float_columns = float_columns , bool_columns= bool_columns, word_columns = word_columns ,out= f"tmp{i}.csv"))
            
        print(f"\nfile {ds//threshold+1} of {ds//threshold+1}")

        files.append(_generate(ds%threshold, int_columns = int_columns , str_columns = str_columns , float_columns = float_columns , bool_columns= bool_columns, word_columns = word_columns ,out= f"tmp.csv"))

        out = merge(files,out=out)
    
    ends =datetime.datetime.now()
    
    c = ends-start
    print("\nSuccess")
    print("\n\nTotal Time: ",end= ' ')
    printtime(c)
    return out
    
# merge(['gen12.csv','gen3.csv','gen3.csv'])

#autosplitter

if __name__ == "__main__":
    import argparse

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=False, help="output file name")
    ap.add_argument("-i", "--int","--integer", default=0, required=False, help="output file name")
    ap.add_argument("-f", "--float", default=0, required=False, help="output file name")
    ap.add_argument("-s", "--str","--string", default=0, required=False, help="output file name")
    ap.add_argument("-b", "--bool", default=0, required=False, help="output file name")
    ap.add_argument("-w", "--word", default=0, required=False, help="output file name")
    ap.add_argument("-fs", "--size", "--filesize",  required=True, help="output file name")
    ap.add_argument("-d", "--debug", "--debugging",   help="output file name")
    
    args = vars(ap.parse_args())

    size =  int(''.join(x for x in args['size'] if x.isdigit()))

    if args['out'] is None:
        out = 'gen.csv'
    else:
        out = args['out']
        rename = False
    
    
    units = args['size'].replace(f'{size}','')
    if (units == ""):
        unit = B   
    elif (units == "MB"):
        unit = MB
    elif (units == "GB"):
        unit = GB
    elif (units == "B"):
        unit = B
    elif (units == "KB"):
        unit = KB
    elif (units == "TB"):
        unit = TB
    else:
        print(f"ERROR : Unknown size format '{units}'")
        exit(1)
    for i in args:
        try:
            args[i] = int(args[i])
        except:
            pass
    if ((args['word'] + args['str'] + args['bool'] + args['float'] + args['int'] )== 0):
        args["int"] = 1

    print (f" \n DATASET FOMAT : {size}{units} * [|{'int|'*args['int']}{'bool|'*args['bool']}{'float|'*args['float']}{'str|'*args['str']}{'word|'*args['word']}]\n")

    # generate(500*MB,int_columns= 1,float_columns=2,bool_columns=2,str_columns=1,word_columns=1)
    print ("\n")

    out = generate(size*unit,int_columns=args['int'],float_columns=args['float'],bool_columns=args['bool'],str_columns=args['str'],word_columns=args['word'],out = out)

    print("\n\nResult:  ")
    system(f"ls -lh {out} "+"| awk '{print $9 , $5}'")
    print("")
