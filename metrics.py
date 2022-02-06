from math import inf
import numpy as np

from main import load_vocab

def linear_distance(a,b):
    delta = (a-b)
    return min(delta%12,12-(delta%12))

def melody_sim_DP(a,b,w=linear_distance):

    dp = np.zeros([len(a),len(b)])

    a = [float(token[6:-1]) for token in a if token[:5]=='Pitch']
    b = [float(token[6:-1]) for token in b if token[:5]=='Pitch']

    length_penalty = lambda x : x*3

    def get_value(i,j):
        '''
        Handle negative index
        '''
        if i<=-2 or j<=-2:
            return inf
        if i==-1:
            return length_penalty(j+1)
        if j==-1:
            return length_penalty(i+1)
        if i>=0 and j>=0:
            return dp[i,j]


    for i in range(len(a)):
        for j in range(len(b)):
            dp[i,j] = w(a[i],b[j])+min(
                get_value(i-1,j-1),
                get_value(i-2,j-1)+ w(a[i-1],b[j]),
                get_value(i-1,j-2)+ w(a[i],b[j-1]),
                )
    print(dp)

    candidates = []
    for i in range(len(a)):
        candidates.append(dp[i,-1] + length_penalty(len(a)-1-i))

    for j in range(len(b)):
        candidates.append(dp[-1,j] + length_penalty(len(b)-1j))
    result = min(candidates)

    return result

if __name__ == '__main__':
    a = ['Pitch(7)','Pitch(2)','Pitch(3)','Pitch(4)','Pitch(7)','Pitch(7)']
    b = ['Pitch(2)','Pitch(3)','Pitch(4)']
    print('original:',a,'\nquery:',b)
    print(melody_sim_DP(a,b,linear_distance))

    a = ['Pitch(1)','Pitch(2)','Pitch(3)','Pitch(4)']
    b = ['Pitch(2)','Pitch(3)','Pitch(4)']
    print('original:',a,'\nquery:',b)
    print(melody_sim_DP(a,b,linear_distance))

