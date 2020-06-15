import random
import os
from itertools import chain

def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

nCT_list = os.listdir("/home/qian/Desktop/projects/iMP/ictcf/image/nCT")
train_nCT = list(chain(*partition(nCT_list,10)[0:8]))
f= open("/home/qian/Desktop/projects/iMP/ictcf/Data-split/nCT/train_nCT.txt","w+")
for i in train_nCT:
    f.write("%s\n" % i)
    
f.close()

val_nCT = partition(nCT_list,10)[8]
f= open("/home/qian/Desktop/projects/iMP/ictcf/Data-split/nCT/val_nCT.txt","w+")
for i in val_nCT:
    print(i)
    f.write("%s\n" % i)
    
f.close()

test_nCT = partition(nCT_list,10)[9]

f= open("/home/qian/Desktop/projects/iMP/ictcf/Data-split/nCT/test_nCT.txt","w+")
for i in test_nCT:
    print(i)
    f.write("%s\n" % i)

f.close()



pCT_list = os.listdir("/home/qian/Desktop/projects/iMP/ictcf/image/pCT")
train_pCT = list(chain(*partition(pCT_list,10)[0:8]))
f= open("/home/qian/Desktop/projects/iMP/ictcf/Data-split/pCT/train_pCT.txt","w+")
for i in train_pCT:
    print(i)
    f.write("%s\n" % i)

f.close()

val_pCT = partition(pCT_list,10)[8]

f= open("/home/qian/Desktop/projects/iMP/ictcf/Data-split/pCT/val_pCT.txt","w+")
for i in val_pCT:
    print(i)
    f.write("%s\n" % i)
    
f.close()

test_pCT = partition(pCT_list,10)[9]
f= open("/home/qian/Desktop/projects/iMP/ictcf/Data-split/pCT/test_pCT.txt","w+")
for i in test_pCT:
    print(i)
    f.write("%s\n" % i)

f.close()