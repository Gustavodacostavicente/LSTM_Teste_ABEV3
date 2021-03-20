import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler

Defasagem = True
np.random.seed(7)# Gerar resultado simulável

Dados_Ativo =  pd.read_csv('ABEV3_Daily.csv',  sep=';', engine='python')# Abre Planilha MT5
Abertura = Dados_Ativo.iloc[:,1:2].values
Maxima = Dados_Ativo.iloc[:,2:3].values
Minima = Dados_Ativo.iloc[:,3:4].values
Fechamento = Dados_Ativo.iloc[:,4:5].values
Volume = Dados_Ativo.iloc[:,5:6].values 

Dados_Ativo_Vista =  pd.read_csv('ABEV3_Daily.csv',  sep=';', engine='python',index_col=0,parse_dates=True)# Abre Planilha MT5
OHLCV = Dados_Ativo.drop('Data',1)
Entrada = OHLCV # Entrada 1 Rede Neural Recorrente

if(Defasagem == True):
    Abertura_Defasada = Entrada['Open'].shift(-1).values
    Entrada_2 = pd.DataFrame()
    Entrada_2.insert(0,"Open",Abertura_Defasada,True)
    Entrada_2.insert(1,"High",Maxima,True)
    Entrada_2.insert(2,"Low",Minima,True)
    Entrada_2.insert(3,"Close",Fechamento,True)
    Entrada_2.insert(4,"Volume",Volume,True)
    Entrada_2 = Entrada_2.dropna()
    Entrada = Entrada_2
    
Separador = int(100)# Separação de dados em relação a Valores
Periodos = 200 # Relação móvel de preços passados para prever o próximo preço
Dados_Ativo_Vista_Teste = Dados_Ativo_Vista[-Separador:]

#%%%
Dados_Ativo_Vista_Teste = Dados_Ativo_Vista[-Separador:]

Normaliza = MinMaxScaler(feature_range=(0, 1))# Função de Normalização entre 0 e 1

Entrada = Normaliza.fit_transform(Entrada) # Normaliza os valores de entrada
Saida = Normaliza.fit_transform(Fechamento)

Preco_Treinamento = Entrada[:-Separador] # Separa Treinamento em 80%
Preco_Teste = Entrada[-Separador-Periodos:] # Separa Teste em 20%

Preco_Entrada = []
Preco_Seguinte=[]

for i in range(Periodos,len(Preco_Treinamento)): 
    Preco_Entrada.append(Preco_Treinamento[i-Periodos:i])# Cria matriz (Dados Treinamento - Periodos, Periodos)
    Preco_Seguinte.append(Saida[i])# Matriz de previsores do próximo valor buscado pela RNN
    
Preco_Entrada, Preco_Seguinte = np.array(Preco_Entrada),np.array(Preco_Seguinte)# Transforma numpy
Preco_Entrada = Preco_Entrada.reshape(Preco_Entrada.shape[0], Preco_Entrada.shape[1], Preco_Entrada.shape[2])# Transforma no formato de leitura da RN em (inputs,timesteps,1)
#%%
model = Sequential()
model.add(LSTM(units=30, activation="sigmoid", return_sequences=True, input_shape=(Preco_Entrada.shape[1],Preco_Entrada.shape[2]))) # Return sequences avança para a próxima camada conectada a rede
model.add(Dropout(0))# Função Forget Gate
model.add(LSTM(units=50, activation="sigmoid", return_sequences=True))
model.add(Dropout(0))# Função Forget Gate
model.add(LSTM(units=50,activation="sigmoid", return_sequences=True)) 
model.add(Dropout(0))# Função Forget Gate
model.add(LSTM(units=10,activation="sigmoid"))
model.add(Dropout(0.05))
model.add(Dense(units=1,activation="sigmoid"))# Bloco neural final
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
history = model.fit(Preco_Entrada, Preco_Seguinte, epochs = 1000, batch_size = 32 ,verbose=1)
model.save('model.h5')
#%%
    
mpf.plot(Dados_Ativo_Vista,type='candle',volume=True,mav=(9,21))
plt.savefig(str('ABEV3_SH_Completa'))
plt.close()

plt.figure(2,figsize=(14,8), dpi=80, facecolor='w', edgecolor='k')
plt.ylabel('Perda'); plt.xlabel('Épocas')
plt.semilogy(history.history['loss'])
plt.grid(True)
plt.title('Gráfico Função de Custo')
plt.savefig(str('ABEV3_Funcao_Custo'))
plt.close()


#%% 
Preco_Entrada = []
Preco_Seguinte_Open = []
Preco_Seguinte_High = []
Preco_Seguinte_Low = []
Preco_Seguinte_Close = []

for i in range(Periodos,len(Preco_Teste)):
    Preco_Entrada.append(Preco_Teste[i-Periodos:i])
    Preco_Seguinte_Open.append(Preco_Teste[i][0])
    Preco_Seguinte_High.append(Preco_Teste[i][1])
    Preco_Seguinte_Low.append(Preco_Teste[i][2])
    Preco_Seguinte_Close.append(Preco_Teste[i][3])

Preco_Seguinte_Open = np.array(Preco_Seguinte_Open)
Preco_Seguinte_Open = Preco_Seguinte_Open.reshape(len(Preco_Seguinte_Open),1) 
Preco_Seguinte_Open = Normaliza.inverse_transform(Preco_Seguinte_Open)

Preco_Entrada = np.array(Preco_Entrada)
Preco_Seguinte_Close = np.array(Preco_Seguinte_Close)
Preco_Entrada = Preco_Entrada.reshape((Preco_Entrada.shape[0], Preco_Entrada.shape[1], Preco_Entrada.shape[2]))
Preco_Previsao_Ponto_Movel = model.predict(Preco_Entrada)

#%%
Preco_Previsao_Ponto_Movel_Inv = Normaliza.inverse_transform(Preco_Previsao_Ponto_Movel)
Preco_Seguinte_Close_R = Preco_Seguinte_Close.reshape(len(Preco_Seguinte_Close),1) 
Preco_Seguinte_Close_Inv = Normaliza.inverse_transform(Preco_Seguinte_Close_R)
 
#%%

Matriz_Velas = []
for i in range(0,len(Dados_Ativo_Vista_Teste)):
    if(Dados_Ativo_Vista_Teste.iloc[i,0]>Dados_Ativo_Vista_Teste.iloc[i,3]):
        Matriz_Velas.append('Vermelha')
    if(Dados_Ativo_Vista_Teste.iloc[i,0]<Dados_Ativo_Vista_Teste.iloc[i,3]):
       Matriz_Velas.append('Verde')
    
plt.figure(3,figsize=(14,8), dpi=80, facecolor='w', edgecolor='k')
plt.grid(True)
plt.plot(Preco_Previsao_Ponto_Movel_Inv,'.', color = 'red', label='Previsão Lstm Fechamento')
plt.plot(Preco_Previsao_Ponto_Movel_Inv,'k:', color = 'red')

plt.plot(Preco_Seguinte_Close_Inv,'.', color = 'black', label='Atual Fechamento')
plt.plot(Preco_Seguinte_Close_Inv,'k:', color = 'black')

plt.title('Gráfico Previsão Ponto Móvel')# Utiliza apenas a base de teste para entrada 
plt.legend()
plt.savefig(str('ABEV3_Movel'))
plt.close()
#%%
# Dados_Ativo_Vista_Teste.iloc[:,3] = Preco_Previsao_Ponto_Movel_Inv

# for i in range(0,len(Dados_Ativo_Vista_Teste)):
#     if(Dados_Ativo_Vista_Teste.iloc[i,3] > Dados_Ativo_Vista_Teste.iloc[i,1]):
#         Dados_Ativo_Vista_Teste.iloc[i,1] = Dados_Ativo_Vista_Teste.iloc[i,3]
#     if(Dados_Ativo_Vista_Teste.iloc[i,3] < Dados_Ativo_Vista_Teste.iloc[i,2]):
#         Dados_Ativo_Vista_Teste.iloc[i,2] = Dados_Ativo_Vista_Teste.iloc[i,3]

# mpf.plot(Dados_Ativo_Vista_Teste,type='candle',volume=True,mav=(9,21))

#%%
Preco_Previsao_Realimentado = Preco_Teste.copy()

for i in range(Periodos,len(Preco_Teste)):
    Preco_Entrada_Aux = Preco_Previsao_Realimentado[i-Periodos:i].reshape(1, Periodos, 5)
    Preco_Previsao_Realimentado[i][3] = model.predict(Preco_Entrada_Aux)
    
Preco_Previsao_Realimentado = Preco_Previsao_Realimentado[Periodos:]
Preco_Previsao_Realimentado_Inv = Normaliza.inverse_transform(Preco_Previsao_Realimentado[:,3].reshape(len(Preco_Previsao_Realimentado),1))
Matriz_Previsao_Realimentada = [] 
Preco_Previsao_Realimentado = Preco_Previsao_Realimentado[Periodos:]

plt.figure(4,figsize=(14,8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(Preco_Previsao_Realimentado_Inv,'.', label='Previsão Lstm Fechamento', color = 'red')
plt.plot(Preco_Previsao_Realimentado_Inv,'k:', color = 'red')
plt.plot(Preco_Seguinte_Close_Inv,'.',label='Atual Fechamento', color = 'black')
plt.plot(Preco_Seguinte_Close_Inv,'k:', color = 'black')
plt.grid(True)
plt.title('Gráfico Previsão Realimentado')# Utiliza a própria previsão como dado de entrada
plt.legend()
plt.savefig(str('ABEV3_Realimentado'))
plt.close()