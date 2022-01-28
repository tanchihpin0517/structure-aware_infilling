from math import inf
import numpy as np

from main import load_vocab
def melody_sim_DP(a,b,w):

    dp = np.zeros([len(a),len(b)])

    a = [float(token[6:-1]) for token in a if token[:5]=='Pitch']
    b = [float(token[6:-1]) for token in b if token[:5]=='Pitch']

    for i in range(len(a)):
        dp[i,0]=i

    for i in range(len(b)):
        dp[0,i]=i

    def get_value(i,j):
        if i<=-2 or j<=-2:
            return inf
        if i>=0 and j>=0:
            return dp[i,j]
        if i==-1:
            return j+1
        if j==-1:
            return i+1
        

    for i in range(len(a)):
        for j in range(len(b)):
            dp[i,j] = w(a[i],b[j])+min(
                get_value(i-1,j-1),
                get_value(i-2,j-1)+ w(a[i-1],b[j]),
                get_value(i-1,j-2)+ w(a[i],b[j-1]),
                )
    print(dp)
    return dp[-1,-1]


def linear_distance(a,b):
    delta = (a-b)
    return min(delta%12,12-(delta%12))

a = ['Pitch(2)','Pitch(23)','Pitch(4)']
b=['Pitch(1)','Pitch(2)','Pitch(23)','Pitch(4)']
print(melody_sim_DP(a,b,linear_distance))

import json
with open('./gen_tokens/0.2/001.json') as f:
    ids = json.load(f)
vocab = load_vocab('./dataset/vocab.txt')
for id in ids[:100]:
    print(vocab.id_to_token[id])