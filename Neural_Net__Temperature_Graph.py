# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# %% [markdown]
# USANDO UMA REDE NEURAL PARA APRENDER A RELAÇÂO A TEMPERATURA X TEMPO EM FLUIDOS, BASEADO NA LEI DE RESFRIAMENTO DE NEWTON

# %%
#Geração dos dados experimentais
T0 = 37.5 #ºC
k = 0.3
Tamb = 25 #ºC

def T(t,k,T0,Tamb):

    T =  Tamb + (T0 - Tamb) * np.exp(-k*t)
    return T

tempo = np.linspace(0,60,100)

dados_experimentais = T(tempo,k,T0,Tamb)

# %%
#Funções de ativação
def sigmoid(x):
    return 1 / (1+np.exp(-x))

def ReLU(x):
    return np.maximum(0,x)

def ReLU_deriv(x):
    return (x > 0).astype(float)
    
def sigmoid_deriv(x):
    return sigmoid(x) * (1-sigmoid(x))


# %%
#Normalização dos dados:

def MaxMin(X:np.ndarray):
    return (X - np.min(X)) / (np.max(X) - np.min(X))
def Desnorm_MaxMin(X_norm:np.ndarray, X:np.ndarray):
    return X_norm *(np.max(X) - np.min(X)) + np.min(X)

X = tempo.reshape(-1,1)
Y = dados_experimentais.reshape(-1,1)

X_norm = MaxMin(X)
Y_norm = MaxMin(Y)

fig, ax = plt.subplots(figsize=(10,6))

ax.scatter(tempo, dados_experimentais, color="blue", alpha=0.3, label = "Dados da Simulação")
line, = ax.plot([],[], color="red", label="Aprendizado da Rede")
texto_epoca = ax.text(0.05, 0.9, '', transform=ax.transAxes)

ax.set_ylim(20, 40)
ax.set_xlim(-1, 60)
ax.legend()



# %%

learning_rate = 0.1



W1 = np.random.randn(1,4)
b1 = np.zeros((1,4))
W2 = np.random.randn(4,1)
b2 = np.zeros((1,1))

def update(frame):
    global W1,b1,W2,b2, learning_rate
    for epoch in range(5):
        # FORWARD PASS 
        Z1 = np.dot(X_norm, W1) + b1
        A1 = ReLU(Z1)
        
        Z2 = np.dot(A1, W2) + b2
        A2 = sigmoid(Z2)

        # LOSS 
        loss = np.mean((A2 - Y_norm)**2)

        # BACKPROPAGATION
        # Camada de Saída
        dZ2 = 2 * (A2 - Y_norm) * sigmoid_deriv(Z2)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Camada Oculta
        dZ1 = np.dot(dZ2, W2.T) * ReLU_deriv(Z1) 
        dW1 = np.dot(X_norm.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        #  ATUALIZAÇÃO
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    prev = Desnorm_MaxMin(A2, dados_experimentais)
    line.set_data(tempo, prev)
    texto_epoca.set_text(f"Epoca: {frame * 5}")

    return line, texto_epoca


# %%

ani = FuncAnimation(fig,update,frames=50, interval=5,blit=True)
plt.show()



# %%
