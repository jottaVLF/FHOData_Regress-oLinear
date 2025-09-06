# FHOData_RegressãoLinear
Este projeto em Python realiza uma análise completa de um conjunto de dados de carros, com o objetivo de limpar os dados, explorar a relação entre as características dos veículos e construir um modelo de regressão linear

O fluxo de trabalho inclui as seguintes etapas:

Limpeza de Dados Robusta: O código lê o arquivo Cars Datasets 2025.csv e limpa as colunas numéricas, lidando com valores complexos como faixas de preço ($12,000-$15,000), unidades de medida (cc, hp, kwh, Nm) e formatos de divisão (1600/13.8), garantindo que os dados estejam prontos para a análise.

Análise de Correlação: Uma matriz de correlação é gerada e visualizada em um mapa de calor para identificar a força e a direção da relação entre as variáveis, como HorsePower e Cars Prices.

Modelagem de Regressão Linear: Um modelo de regressão linear é construído para prever os preços dos carros com base em características como HorsePower, Total Speed, Performance e Torque. O modelo é avaliado usando métricas como o R-quadrado e o Erro Quadrático Médio (MSE).

#Como Executar
Clone este repositório para sua máquina local.

Certifique-se de que o arquivo de dados Cars Datasets 2025.csv esteja na mesma pasta que o script Python.

Execute o script em um ambiente Python. Você pode usar um editor de código como o VS Code, PyCharm, ou um Jupyter Notebook.

Estrutura do Código
O código está dividido em três seções principais, cada uma responsável por uma parte do fluxo de trabalho:

1. Limpeza dos Dados
Esta seção carrega o arquivo .csv e aplica uma função de limpeza personalizada para pré-processar as colunas, removendo unidades, pontuações e convertendo valores de texto em números. A função clean_and_convert_value() é o coração desta etapa, lidando com as inconsistências dos dados de forma eficiente.

2. Matriz de Correlação
Aqui, um mapa de calor é gerado usando a biblioteca seaborn. Ele visualiza a matriz de correlação das colunas numéricas, fornecendo um entendimento rápido das relações entre as características dos carros.

3. Regressão Linear
Nesta etapa, o modelo de machine learning é construído. Os dados são divididos em conjuntos de treino e teste, o modelo é treinado, as previsões são feitas e, finalmente, o desempenho do modelo é avaliado. As métricas de desempenho e os coeficientes do modelo são exibidos no console.

Resultados
Ao executar o código, você verá as seguintes saídas:

Amostras dos Dados: O console mostrará as primeiras linhas dos dados antes e depois da limpeza, permitindo que você verifique o sucesso do pré-processamento.

Mapa de Calor: Um gráfico de mapa de calor será gerado, mostrando a correlação entre as variáveis.

Métricas de Desempenho: As métricas do modelo de regressão linear, como o Coeficiente de Determinação. Um R² de 0.56, por exemplo, indica que 56% da variação nos preços é explicada pelas variáveis do modelo.

Previsões: As primeiras previsões do modelo são mostradas lado a lado com os preços reais para uma comparação direta.
