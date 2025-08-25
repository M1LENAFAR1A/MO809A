# 1. Importando as bibliotecas necessárias
import grpc
from concurrent import futures
import time

# Importando o modelo de Machine Learning
from sklearn.neighbors import KNeighborsClassifier

# Importando os arquivos gerados pelo compilador gRPC
import treinamento_pb2
import treinamento_pb2_grpc

# 2. A classe do Servidor com a lógica de negócio
class Servidor(treinamento_pb2_grpc.ModeloServiceServicer):

    # O método __init__ é executado quando a classe é criada.
    # Usamos para inicializar nosso modelo.
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=3)
        print("Modelo KNeighborsClassifier inicializado.")

    # Implementação do método 'Fit' com lógica real
    def Fit(self, request, context):
        print(f"Recebi {len(request.dados_treino)} amostras para treinar.")
        
        # Prepara as listas para o formato que o sklearn espera
        X_treino = []
        y_treino = []
        
        # Extrai os dados da requisição gRPC
        for amostra in request.dados_treino:
            X_treino.append(list(amostra.atributos))
            y_treino.append(amostra.rotulo)
            
        # Treina o modelo com os dados recebidos
        self.model.fit(X_treino, y_treino)
        print("Treinamento do modelo concluído.")
        
        # Calcula a acurácia real do modelo nos dados de treino
        acuracia_calculada = self.model.score(X_treino, y_treino)
        print(f"Acurácia calculada: {acuracia_calculada}")

        # Cria o objeto de resposta
        resposta = treinamento_pb2.FitResponse()
        
        # Preenche o campo da resposta com a acurácia real
        resposta.acuracia = acuracia_calculada
        
        return resposta

    # Implementação do método 'Predict' com lógica real
    def Predict(self, request, context):
        print(f"Recebi os atributos para predição: {list(request.atributos)}")
        
        # O sklearn espera uma lista de listas para predição, 
        # mesmo que seja para uma única amostra.
        dados_para_prever = [list(request.atributos)]
        
        # Usa o modelo treinado para fazer a predição
        predicao = self.model.predict(dados_para_prever)
        
        # Pega o primeiro (e único) resultado da predição
        rotulo_previsto = predicao[0]
        print(f"Rótulo previsto: {rotulo_previsto}")

        # Cria o objeto de resposta
        resposta = treinamento_pb2.PredictResponse()
        
        # Preenche o campo da resposta com o rótulo previsto
        resposta.result = rotulo_previsto
        
        return resposta

# 3. Função para iniciar e rodar o servidor
def serve():
    servidor = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    treinamento_pb2_grpc.add_ModeloServiceServicer_to_server(Servidor(), servidor)
    servidor.add_insecure_port('[::]:50051')
    servidor.start()
    print("Servidor iniciado na porta 50051.")
    servidor.wait_for_termination()

# 4. Ponto de entrada do script
if __name__ == '__main__':
    serve()
