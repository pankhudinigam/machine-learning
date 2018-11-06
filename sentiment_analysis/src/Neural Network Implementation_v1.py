#3 layer Neural Network: Input layer=1; Output layer:1; Hidden layers=1;

import numpy as np

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))#Computing g(x).

def J(theta,n,m,lamda,y,a3):
    Sigma1=0
    Sigma2=0
    test=0
    for i in range(m):
        for k in range(K):
            Sigma1=Sigma1+y[i][k]*np.log(a3[i][k])+(1-y[i][k])*np.log(1-a3[i][k])
    if(n==1):
        for i in range(neuron1):
            for j in range(neuron2):
                Sigma2=Sigma2+pow(theta[i][j],2)
                test=test+pow(theta1[i][j],2)
    if(n==2):
        for i in range(neuron2):
            for j in range(3):
                Sigma2=Sigma2+pow(theta[i][j],2)
    #print("Sigma2: ",Sigma2)
    #print("test: ",test)
    cost=-(1.0/m)*(Sigma1)+(lamda/(2*m))*(Sigma2)
    return(cost)


def gradient_checking(x,y,m,Epsilon,lamda,K,a3,theta1,theta2):
    gradApprox1 = np.zeros((neuron1,neuron2))
    gradApprox2 = np.zeros((neuron2,3))
    print("Calculating gradApprox")
    print(len(theta1))
    
    for i in range(neuron1):
        for j in range(neuron2):
            thetaPlus = np.array(theta1)
            #print("thetaPlus[i] ",thetaPlus[i][j])
            thetaPlus[i][j] = thetaPlus[i][j] + Epsilon
            #print("thetaPlus[i] ",thetaPlus[i][j])
            #print("theta1[i][j] ",theta1[i][j])
            thetaMinus = np.array(theta1)
            thetaMinus[i][j] = thetaMinus[i][j] - Epsilon
            gradApprox1[i][j] = (J(thetaPlus,1,m,lamda,y,a3) - J(thetaMinus,1,m,lamda,y,a3))/(2*Epsilon)#Gives Approximation of J(theta) w.r.t. theta i
        
            #print("\nJ(theta+Epsilon)= ",J(thetaPlus,1,m,lamda,y,a3)," J(theta-Epsilon)= ",J(thetaMinus,1,m,lamda,y,a3))

    for i in range(neuron2):
        for j in range(3):
            thetaPlus = np.array(theta2)
            thetaPlus[i][j] = thetaPlus[i][j] + Epsilon
            thetaMinus = np.array(theta2)
            thetaMinus[i][j] = thetaMinus[i][j] - Epsilon
            gradApprox2[i][j] = (J(thetaPlus,2,m,lamda,y,a3) - J(thetaMinus,2,m,lamda,y,a3))/(2*Epsilon)#Gives Approximation of J(theta) w.r.t. theta i
    print("gradApprox1= ",gradApprox1)
    print("gradApprox2= ",gradApprox2)
    return(gradApprox1,gradApprox2)
    
def create_vocab(sentences):
    vocab=[]
    for sentence in sentences:
        for word in sentence.split(" "):
            if word not in vocab:
                vocab.append(word)
    vocab.remove('')
    vocab.remove('.')
    vocab.remove('....')
    return(vocab)

def unrolled(m1,m2):
    lst=list()
    for i in range(neuron1):
        for j in range(neuron2):
            lst.append(m1[i][j])
    for i in range(neuron2):
        for j in range(3):
            lst.append(m2[i][j])
    return(lst)

def compare(Dvec,gradApprox,total):
    Sum=0
    Sum2=0
    Sum3=0
    for i in range(total):
        #Sum=Sum+Dvec[i]-gradApprox[i]
        #Sum2=Sum2+abs(Dvec[i])-abs(gradApprox[i])
        Sum3=Sum3+abs(Dvec[i]-gradApprox[i])
    #print("\nThe Average Value of difference between Dvec and gradApprox is: ",(Sum/total))
    #print("\nThe Average Value of difference between Dvec and gradApprox is: ",(Sum2/total))
    print("\nThe Average Value of difference between Dvec and gradApprox is: ",(Sum3/total))

#main:
fp=open("review1.txt","r")
text=fp.read()
fp1=open("stopwords.txt","r")
stop_words=(fp1.read()).split("\n")
text=text.lower()
text=text.replace("n't"," not")
sentences=text.split("\n")
for sentence in sentences:
    if(sentence is ' ' or sentence is ''):
        sentences.remove(sentence)
        
i=0
for sentence in sentences:
    content=""
    for word in sentence.split(" "):
        if word not in stop_words:
            content=content+" "+word
    sentences[i]=content
    i=i+1

vocab=create_vocab(sentences)
print("Vocab Done")
list_x=[]
for i in range(len(sentences)):
    sentence=sentences[i]
    temp_list=[]
    for j in range(len(vocab)):
        word=vocab[j]
        if word in sentence:
            temp_list.append(1)
        else:
            temp_list.append(0)
    list_x.append(temp_list)
   
X = np.array(list_x)
inp_dim=len(list_x)
print("X: ",X)

y = np.array([[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[0,0,1],[1,0,0],[1,0,0],[1,0,0],[0,0,1],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[0,0,1],[0,0,1],[0,1,0],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,0,1],[1,0,0],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,0,1],[0,0,1],[0,0,1],[1,0,0]])#[pos,neu,neg]
out_dim=len(y)
print("Each element in x of size: ",len(list_x[0]))
m=out_dim
print("\nInput Matrix: ",inp_dim," Output Matrix: ",out_dim)

np.random.seed(1)
neuron1=len(vocab)
neuron3=3
neuron2=int(pow(neuron1*neuron3,0.5))


print("\nNeurons in layer 2: ",neuron2)
l0 = X
print("lo: ",l0)

theta1 = 2*np.random.random((neuron1,neuron2))- 1#syn0
theta2 = 2*np.random.random((neuron2,3))- 1#syn1
capital_delta1 = np.zeros((neuron1,neuron2))
capital_delta2 = np.zeros((neuron2,3))
print("Weights Initialized")
#print("theta1:\n",theta1,"\ntheta2:\n",theta2)
lamda=0.01
alpha=-0.1#values of alpha which performed well=>-0.1 and -0.01, values smaller and greater than these didnt quite work so well.
for j in range(60000):
    l0 = X
    l1 = nonlin(np.dot(l0,theta1))
    l2 = nonlin(np.dot(l1,theta2))
    l2_delta = y - l2#l2_error
    #l2_delta = l2_error*nonlin(l2,deriv=True)
    if (j% 100) == 0:
        print("Error:" , str(np.mean(np.abs(l2_delta))))
    l1_error = l2_delta.dot(theta2.T)
    l1_delta = l1_error * nonlin(l1,deriv=True)

    capital_delta2=capital_delta2+(l1.T).dot(l2_delta)
    capital_delta1=capital_delta1+(l0.T).dot(l1_delta)
    D2=(1.0/m)*capital_delta2+lamda*theta2#=der(J(theta)) w.r.t. theta2 for layer2
    D1=(1.0/m)*capital_delta1+lamda*theta1#=der(J(theta)) w.r.t. theta1 for layer1
    
    theta2 = theta2 - alpha*(l1.T.dot(l2_delta))
    theta1 = theta1 - alpha*(l0.T.dot(l1_delta))
    #theta2 += l1.T.dot(l2_delta)
    #theta1 += l0.T.dot(l1_delta)
    #if(j%10000==0):
    #    print("theta1:\n",theta1,"\ntheta2:\n",theta2)
    
print("Training Done")
m=76#Number of training examples

Epsilon=0.0001
K=3#number of output nodes
print("Going For Gradient checking")
gradApprox1,gradApprox2=gradient_checking(X,y,m,Epsilon,lamda,K,l2,theta1,theta2)
Dvec=unrolled(D1,D2)
gradApprox=unrolled(gradApprox1,gradApprox2)
#print("\nDvec: ",Dvec,"\ngradApprox: ",gradApprox)
total=neuron1*neuron2+neuron2*3
compare(Dvec,gradApprox,total)


print("Output After Training:")
output=[]
for i in range(len(l2)):
    tuple_i=l2[i]
    temp=[]
    for j in range(len(tuple_i)):
        temp.append(int(l2[i][j]*10000)/10000)
    output.append(temp)   

#Testing using the Neural Network
print("Testing the Net")
fp_test=open("testing_data.txt","r")
text=fp_test.read()
text=text.lower()
text=text.replace("n't"," not")
sentences=text.split("\n")

for sentence in sentences:
    if(sentence is ' ' or sentence is ''):
        sentences.remove(sentence)
        
i=0
for sentence in sentences:
    content=""
    for word in sentence.split(" "):#Removing stop words
        if word not in stop_words:
            content=content+" "+word
    sentences[i]=content
    i=i+1

list_x=[]
for i in range(len(sentences)):
    sentence=sentences[i]
    temp_list=[]
    for j in range(len(vocab)):
        word=vocab[j]
        if word in sentence:
            temp_list.append(1)
        else:
            temp_list.append(0)
    list_x.append(temp_list)
   
X1 = np.array(list_x)
inp_dim=len(list_x)
print("X1: ",X1)

#Neural Network Code:
a11 = X1#FP1
z21 = np.dot(a11,theta1)#FP2
a21 = nonlin(z21)#l1
z31 = np.dot(a21,theta2)
a31 = nonlin(z31)#l2
output=[]
for i in range(len(a31)):
    tuple_i=a31[i]
    temp=[]
    for j in range(len(tuple_i)):
        temp.append(int(a31[i][j]*10000)/10000)
    output.append(temp)   

print("\nTest Results:\n ")
for element in output:
    print(element)

print(len(output))
#print(l2)
