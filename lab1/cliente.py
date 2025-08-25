# 1. Importações e preparação dos dados
import grpc
import treinamento_pb2 as pb2
import treinamento_pb2_grpc as pb2_grpc
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Carrega os dados e divide em conjuntos de treino e teste
iris = load_iris()
atributos = iris.data
rotulos = iris.target
x_treino, x_teste, y_treino, y_teste = train_test_split(atributos, rotulos, test_size=0.2, random_state=42)

# 2. Classe do Cliente para gerenciar a comunicação
class ClienteGRPC:
    def __init__(self):
        self.host = 'localhost'
        self.server_port = 50051
        self.channel = grpc.insecure_channel(f"{self.host}:{self.server_port}")
        self.stub = pb2_grpc.ModeloServiceStub(self.channel)
        print("Cliente conectado ao servidor.")

    def treinar_modelo(self):
        print("\nEnviando dados para treinamento...")
        # Cria a requisição de treino vazia
        requisicao = pb2.FitRequest()

        # Preenche a requisição com os dados de treino
        for atributos_flor, rotulo_flor in zip(x_treino, y_treino):
            amostra = pb2.Amostra(atributos=atributos_flor.tolist(), rotulo=rotulo_flor)
            requisicao.dados_treino.append(amostra)

        # Envia para o servidor e espera a resposta
        resposta = self.stub.Fit(requisicao)
        
        # Retorna a acurácia recebida
        return resposta.acuracia

    def prever_rotulo(self, atributos_para_prever):
        # Cria a requisição de predição vazia
        requisicao = pb2.PredictRequest()

        # Preenche a requisição com os atributos da flor a ser testada
        requisicao.atributos.extend(atributos_para_prever.tolist())

        # Envia para o servidor e espera a resposta
        resposta = self.stub.Predict(requisicao)
        
        # Retorna o rótulo previsto
        return resposta.result

# 3. Ponto de entrada do script
if __name__ == '__main__':
    # Cria o cliente
    cliente = ClienteGRPC()
    
    # --- Ação 1: Treinar o modelo ---
    acuracia_servidor = cliente.treinar_modelo()
    print(f"Treinamento concluído! Acurácia informada pelo servidor: {acuracia_servidor}")
    
    # --- Ação 2: Fazer uma predição ---
    # Vamos pegar a primeira flor do nosso conjunto de teste
    flor_para_teste = x_teste[1]
    rotulo_real = y_teste[1]
    
    print(f"\nEnviando uma flor para predição com atributos: {flor_para_teste.tolist()}")
    
    # Pede ao servidor para prever o rótulo
    rotulo_previsto = cliente.prever_rotulo(flor_para_teste)
    
    print(f"Previsão recebida do servidor: {rotulo_previsto}")
    print(f"O rótulo real era: {rotulo_real}")
