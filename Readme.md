# Classificador de AVC 

A intenção desse mini projeto é fazer um classificador de AVC
 * Vocês vão receber como entrada dados de vários pacientes
 * E o modelo deve retornar se ele teve AVC ou não

## Os dados estão disponíveis no Kaggle

https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

Texto fornecido pelo autor do dataset citado acima em português : 

Este conjunto de dados é usado para prever se um paciente tem probabilidade de sofrer um AVC com base em parâmetros de entrada como sexo, idade, várias doenças e tabagismo. Cada linha nos dados fornece informações relevantes sobre o paciente.

 **Informações de Atributo**
1) id: identificador único
2) gênero: "Masculino", "Feminino" ou "Outro"
3) idade: idade do paciente
4) hipertensão: 0 se o paciente não tiver hipertensão, 1 se o paciente tiver hipertensão
5) doença cardíaca: 0 se o paciente não tiver nenhuma doença cardíaca, 1 se o paciente tiver uma doença cardíaca
6) sempre casado: "Não" ou "Sim"
7) tipo de trabalho: "crianças", "emprego no governo", "nunca trabalhou", "privado" ou "autônomo"
8) tipo de residência: "rural" ou "urbano"
9) nível médio de glicose: nível médio de glicose no sangue
10) IMC: índice de massa corporal
11) status de tabagismo: "fumou anteriormente", "nunca fumou", "fuma" ou "desconhecido"*
12) derrame: 1 se o paciente teve um derrame ou 0 se não teve

*Observação: "Desconhecido" em smoking_status significa que a informação não está disponível para este paciente

## O que é interessante fazer:
- Separar os dados
- Normalizar
- Utilizar um classificador 
- Criar um pipeline
- Utilizar validação cruzada
