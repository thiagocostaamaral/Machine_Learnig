
##    Thiago Costa Amaral
##
##    Neural Network construction
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def sigmoid(x):
    'Function that returns sigmoid function os value'
    return 1 / (1 + math.exp(-x))

def der_cost(self,d_output):
    'Return derivate of function cost'
    d_cost=[]
    for i in range(len(self.output)):
        d_cost.append([2*(self.output[i][0]-d_output[i][0])])
    d_cost=np.array(d_cost)
    return d_cost

def der_sigmoide(self):
    'Return derivate of function sigmoide'
    d_sigm=[]
    for i in range(len(self.output)):
        fx=self.output[i][0]
        d_sigm.append([fx*(1-fx)])
    d_sigm=np.array(d_sigm)
    return d_sigm

def copy(A):
    B=[]
    for i in range(len(A)):
        line=[]
        for j in range(len(A[0])):
            line.append(A[i][j])
        B.append(line)
    return B
    

def output2(a):                    #Function that reverse valeus of colunm matrix a
    lines=len(a)
    x1,x2,x3=0,0,0
    fator=1
    for i in range(lines):
        if i>lines/2-1:
            x2=x2+a[i][0]
        else:
            x2=x2-a[i][0]
        x1=x1+fator*a[i][0]
        fator=fator*-1
        
    if x1>0.5:
        x1=1
    else:
        x1=0
        
    if x2>0.5:
        x2=1
    else:
        x2=0

    # Other output to be recognized by the neural network    
    #if a[lines-3][0]==1 or a[lines-2][0]==1 or a[lines-1][0]==1:
    #    x3=1
    #else:
    #    x3=0
    b=[[x1],[x2]]
    return b

class layer():
    #!!!!!!!!!!input_val MUST be a column matriz !!!!!!!!!!!!!!
    def __init__(self,n_neurons, n_neurons_pre_layer,input_val=[[None]]):
        #-------initiation of values of weights-------------------------------
        self.weights=[]
        self.bias=[]
        for i in range(n_neurons):
            linha=[]
            for j in range(n_neurons_pre_layer):
                linha.append(random.randint(0,600)/100 - 3)
            self.weights.append(linha)
            self.bias.append([random.randint(0,600)/100 - 3])
        self.weights=np.array(self.weights)
        ##-------Definition of input values--------------------------------
        if input_val[0][0]==None:
            input_val=[]
            for i in range(n_neurons_pre_layer):
                input_val.append([random.randint(0,200)/100 - 1])
            self.input=input_val
        else:
            self.input=input_val
        self.input=np.array(self.input)
        #-------Definition of output-----------------------------------------
        self.output=np.dot(self.weights,self.input)+self.bias#without sigmoid function
        self.dif_output=[]                                   #necessary to see "changes" in output ->calculate error
        for i in range(n_neurons):
            self.output[i][0]=sigmoid(self.output[i][0])
            self.dif_output.append([0])
        self.output=np.array(self.output)
        self.dif_output=np.array(self.dif_output)
        #------Definition of matrix of "change" in weights-----------
        self.dCdW=np.zeros((n_neurons,n_neurons_pre_layer))
        #------Definition of matrix that propagate desired changes in pre-layer-----------
        self.dCda=np.zeros((n_neurons_pre_layer,1))
         #------Definition of matrix that change bias-----------
        self.dCdB=np.zeros((n_neurons,1))
        self.contagem=0
        self.learning_rate=1
    def update_out(self,input_val=[[None]]):
        'Function that update output value of layer'
        n_neurons=len(self.weights)
        pre_output=copy(self.output)
        if input_val[0][0]!=None:
            self.input=input_val 
        self.output=np.dot(self.weights,self.input)
        for i in range(n_neurons):
            self.output[i][0]=sigmoid(self.output[i][0])
            self.dif_output[i][0]=self.output[i][0]-pre_output[i][0]
            
        #print(self.dif_output)

    def cost(self,d_output):
        'Return cost of the Neural Network'
        cost=0
        for i in range(len(d_output)):
            cost=cost+(d_output[i][0]-self.output[i][0])**2
        return cost
              
    def backpropagation(self,d_cost=[[None]],d_output=[[None]]):
        'Gives the vector of changes in the w vector'
        if d_cost[0][0]==None:               #Last layer--->takes d_cost from aoutput 
            d_cost=der_cost(self,d_output)
        elif d_cost[0][0]==None and d_output[0][0]==None:
            print("ERRO ---->BACKPROPAGATION NEEDS TO RECIVE DESIRED OUTPUT OR DERIVATE OF COST")
            return
        d_sigm= der_sigmoide(self)
        #Fiding dC/dW matrix and dCdC
        for i in range(len(self.dCdW)):
            control1=d_sigm[i][0]*d_cost[i][0]
            for j in range(len(self.dCdW[0])):
                self.dCdW[i][j]=( self.dCdW[i][j]+self.input[j][0]*control1 )*self.learning_rate
            self.dCdB[i][0]= (self.dCdB[i][0]+  d_sigm[i][0]*d_cost[i][0] )*self.learning_rate
        #Fiding dC/da matrix, where a is the output of previous layer--->input of this layer
        control2=[]
        n_neurons=len(self.output)
        for i in range(n_neurons):
            control2.append(d_sigm[i][0]*d_cost[i][0])    #i would like to explain, but is too much math
        for i in range(len(self.input)):
            soma=0
            for j in range(n_neurons):
                soma=soma+control2[j]*self.weights[j][i]
            self.dCda[i][0]=soma
        self.contagem=self.contagem+1
        
    def apply_backpropagation(self):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[0])):
                self.weights[i][j]=self.weights[i][j]-self.dCdW[i][j]/self.contagem
                self.dCdW[i][j]=0
            self.bias[i][0]=self.bias[i][0]-self.dCdB[i][0]/self.contagem
            self.dCdB[i][0]=0
        self.contagem=0
def main():
    print("\n\n----NEURAL NETWORK V10------",end="\n\n")
    n_of_inputs=20                            #Number of inputs in the sistem
    n_of_layers=3                            #Number of layers of the sistem
    layers=[]                                 #Matrix that contains layers of NN
    n_neurons=[6,4,2]
    for i in range(n_of_layers):
        if i==0:                              #Input of first layer is input values
            layers.append(layer(n_neurons[i],n_of_inputs))
        else:                                 #Input of n layer is output of previous layer
            layers.append(layer(n_neurons[i],len(layers[i-1].output),layers[i-1].output))
    plt.ion()
    fig = plt.figure(facecolor='black',figsize=(4, 6))
    #deve ser um while que para depois de certa condição ser satisfeita
    n_interações=15000
    img_control=1

    control_graph=n_interações/10
    media=0
    interacoes=[]
    custo=[]
    contagem=1
    for a in range(n_interações):
        img_control=img_control-1
        contagem=contagem-1
        input_v=[]
        for i in range(n_of_inputs):
            input_v.append([random.randint(0,10)/10])
        d_output=output2(input_v)             #Desired Output
        ########Defining output from Neural network
        for i in range(n_of_layers):
            if i==0:
                layers[i].update_out(input_v)
            else:
                layers[i].update_out(layers[i-1].output)
        '''
        print("Interação N %d"%a,end="\n\n")
        print(d_output)
        print(layers[i].output)
        '''
        ########Calculus of backpropagation
        for i in range(n_of_layers):
            if i==0:
                layers[n_of_layers-1-i].backpropagation([[None]],d_output)
                #print(layers[n_of_layers-1].dCdW)
            else:
                layers[n_of_layers-1-i].backpropagation(layers[n_of_layers-i].dCda)
        
####################################################################'       
#########################PRINT RESULT ON SCREEN######################
        if img_control==0:
            img_control=n_interações/20
            plt.clf()
            ax = plt.gca()
            ax.cla()
            plt.axis('off')
            dist=1/(n_of_layers+2)
            r_layer=0.1/n_of_inputs
            ##############Imputs
            height=1/(n_of_inputs+2)
            for j in range(n_of_inputs):
                if input_v[j][0]>0.5:
                    circle = plt.Circle((dist,height+height*j), r_layer, color='white')
                else:
                    circle = plt.Circle((dist,height+height*j), r_layer, color='white',fill=False)
                ax.add_artist(circle)
            ##############Layers
            n_layer_before=n_of_inputs                   #Number of neurons of previous layer
            p_height=height
            for i in range(n_of_layers):
                height=1/(len(layers[i].output)+2)
                for j in range(len(layers[i].output)):
                    if layers[i].output[j][0]>0.5:
                        circle = plt.Circle((2*dist+dist*i,height+height*j), r_layer, color='white')
                    else:
                        circle = plt.Circle((2*dist+dist*i,height+height*j), r_layer, color='white',fill=False)
                    ax.add_patch(circle)
                    for k in range(n_layer_before):
                        width=layers[i].weights[j][k]/3
                        if layers[i].weights[j][k]>0:
                            plt.plot([dist+dist*i,2*dist+dist*i],[p_height+p_height*k,height+height*j],linewidth=width,color='green')
                        else:
                            plt.plot([dist+dist*i,2*dist+dist*i],[p_height+p_height*k,height+height*j],linewidth=width,color='red')
                n_layer_before=(len(layers[i].output))
                p_height=height
            ###############Desired Output
            for j in range(len(d_output)):
                if (d_output[j][0]>0.5 and layers[i].output[j][0]>0.5) or (d_output[j][0]<0.5 and layers[i].output[j][0]<0.5):
                    rect = patches.Rectangle((1-dist+r_layer,height+height*j-r_layer),2*r_layer,2*r_layer,color='green')
                else:
                    rect = patches.Rectangle((1-dist+r_layer,height+height*j-r_layer),2*r_layer,2*r_layer,color='red')
                ax.add_patch(rect)
            plt.pause(0.0001)
########################################################################################
###################GRAPH##################################################
        if control_graph==0:
            interacoes.append(a)
            custo.append(media/n_interações*20)
            control_graph=n_interações/20
            media=0
        control_graph=control_graph-1
        media=media+layers[n_of_layers-1].cost(d_output)
        #Executing back propagation:
        if contagem==0:
            contagem=1
            for i in range(n_of_layers):
                layers[i].apply_backpropagation()
        #print("Cost of the Neural Network = %3f"%layers[i].cost(d_output))
                if a>n_interações/2:
                    layers[i].learning_rate=0.075
    fig=plt.figure(2)
    plt.plot(interacoes,custo)
    plt.xlabel('Interações')
    plt.ylabel('Erro')
    plt.show()

main()

