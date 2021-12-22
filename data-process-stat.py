import numpy as np
#import sys



fname = "./data/email-node-comm.txt"
fout = "./data/email.txt"
ffeat = "./data/email.csv"


#fname = "./data/dblp/dblp-node-comm.txt"
##fout = "./data/dblp/dblp.content"

#fname = "./data/fb/fb-node-comm.txt"
#fout = "./data/fb/fb.content"

#fname = "./data/enron/enron-node-comm.txt"
#fout = "./data/enron/enron.content"

#fname = "./data/ca/ca-node-comm.txt"
#fout = "./data/ca/ca.content"


#fname = "./data/p2p-g04/p2p-g04-node-comm.txt"
#fout = "./data/p2p-g04/p2p-g04.content"




#fname = "./data/p2p-g30/p2p-g30-node-comm.txt"
#fout = "./data/p2p-g30/p2p-g30.content"



comm = np.loadtxt(fname)
feat = np.loadtxt(ffeat,delimiter=",")
x=comm.shape
print(x[1])
#ii=np.identity(x[0],dtype=np.int8)
#print(ii[1][1])
#print(ii[1][2])
node=comm[:,0]
label=comm[:,1]
#node.reshape(x[0],1)
#label.reshape(x[0],1)
#new=np.hstack((node,ii,label))
#new=np.concatenate((node,ii, label), axis=1)

print(node.shape)
#print(ii.shape)
#col=x[0]+2
#new=np.empty((x[0],col),dtype=np.int8)

#for i in range(x[0]): 
#	for j in range(col):
#		new[i][0]=node[i]
#		new[i][col-1]=label[i]

#for i in range(x[0]): 
#	for j in range(x[0]): 
#		new[i][j+1]=ii[i][j]




#	new[i]=np.column_stack((node[i],ii[i], label[i]))

new=np.column_stack((node,ffeat, label))
print(new[1,:])
np.savetxt(fout, new, fmt='%d', delimiter=' ', newline='\n')
