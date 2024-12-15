
O código ainda está em desenvolvimento.

# Supervised Learning Framework Inspired by Reinforcement Learning

This framework provides a structured approach to supervised learning, inspired by patterns commonly used in reinforcement learning. It leverages Gymnasium environments and integrates seamlessly with neural network models for efficient training and prediction.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Creating the Environment](#creating-the-environment)
  - [Initializing the Model](#initializing-the-model)
  - [Training the Model](#training-the-model)
  - [Making Predictions](#making-predictions)
- [Components](#components)
  - [ReplayBuffer](#replaybuffer)
  - [EnvDataCollector](#envdatacollector)
  - [LearningManager](#learningmanager)
  - [Factory Function](#factory-function)
- [Design Patterns Applied](#design-patterns-applied)
- [Handling Observations and Masks](#handling-observations-and-masks)
- [Code Quality Tools](#code-quality-tools)
  - [mypy](#mypy)
  - [flake8](#flake8)
  - [black](#black)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Modular Design**: Separates data collection, storage, and training into distinct components.
- **Flexible Replay Buffer**: Efficiently stores and samples transitions for training.
- **Neural Network Integration**: Easily integrate custom neural networks for prediction.
- **Factory Pattern**: Simplifies model and manager instantiation based on environment configurations.
- **Comprehensive Documentation**: Clear docstrings and comments for ease of use and maintenance.

## Installation

Ensure you have Python 3.7 or higher installed. Install the required packages using `pip`:

```bash
pip install torch gymnasium numpy wildfire_pyro







### **Estrutura Proposta para `deep_set_attention_net_wrapper.py`**

1. **Importações**
   - **Descrição:** Importar todas as bibliotecas e módulos necessários para o treinamento.
   - **Componentes:**
     - Bibliotecas padrão (e.g., `os`, `sys`)
     - Bibliotecas de ciência de dados (e.g., `numpy`, `pandas`)
     - Bibliotecas de deep learning (e.g., `torch`, `torch.nn`, `torch.optim`)
     - Módulos personalizados (e.g., `dataset`, `model`, `train`)

2. **Definição de Hiperparâmetros**
   - **Descrição:** Centralizar todos os hiperparâmetros utilizados no treinamento.
   - **Componentes:**
     - Taxa de aprendizado
     - Número de épocas
     - Tamanho do batch
     - Parâmetros de regularização (e.g., `weight_decay`, `lambda_l1`)
     - Configurações do dispositivo (CPU/GPU)
     - Caminhos para salvar modelos e plots

3. **Carregamento de Dados**
   - **Descrição:** Preparar e carregar os conjuntos de dados de treinamento e validação.
   - **Componentes:**
     - Funções para ler os dados (e.g., `read_data`)
     - Criação de datasets personalizados (e.g., `PointNeighborhood`)
     - Geração de batches (e.g., `generate_batches`)

4. **Instanciação do Modelo**
   - **Descrição:** Criar e configurar a arquitetura do modelo de machine learning.
   - **Componentes:**
     - Definição do modelo (e.g., `SpatialRectifiedRegressor`)
     - Transferência para o dispositivo apropriado (CPU/GPU)
     - Ajuste inicial dos pesos, se necessário

5. **Configuração da Função de Perda e Otimizador**
   - **Descrição:** Definir a função de perda e o otimizador para o treinamento.
   - **Componentes:**
     - Função de perda (e.g., `nn.MSELoss`)
     - Otimizador (e.g., `optim.Adam`, `optim.SGD`)
     - Configurações do otimizador (taxa de aprendizado, momentum)

6. **Gerenciamento do Estado de Treinamento**
   - **Descrição:** Manter e atualizar o histórico de treinamento e validação.
   - **Componentes:**
     - Classe `TrainingHistory` para armazenar perdas e melhores métricas
     - Métodos para salvar e carregar checkpoints
     - Atualização das taxas de aprendizado, se necessário

7. **Loop de Treinamento**
   - **Descrição:** Executar o processo de treinamento ao longo das épocas definidas.
   - **Componentes:**
     - Iteração sobre as épocas
     - Iteração sobre os batches de treinamento
     - Cálculo da perda, backpropagation e atualização dos pesos
     - Retenção e monitoramento dos gradientes e ativações, se necessário

8. **Validação do Modelo**
   - **Descrição:** Avaliar o desempenho do modelo em dados de validação após cada época.
   - **Componentes:**
     - Iteração sobre os batches de validação
     - Cálculo da perda de validação
     - Atualização do histórico de validação

9. **Monitoramento e Log**
   - **Descrição:** Registrar e visualizar o progresso do treinamento.
   - **Componentes:**
     - Impressão de logs periódicos (perda de treinamento, perda de validação, etc.)
     - Plotagem de gráficos de perdas (usando `matplotlib`)
     - Debugging periódico (análise de ativações e gradientes)

10. **Salvamento de Checkpoints e Modelos**
    - **Descrição:** Salvar o estado do modelo e do treinamento para uso futuro.
    - **Componentes:**
      - Salvamento de checkpoints periódicos
      - Salvamento do melhor modelo baseado na métrica de validação
      - Salvamento final do modelo após o término do treinamento

11. **Função Principal e Execução do Script**
    - **Descrição:** Encapsular a lógica de treinamento dentro de uma função principal.
    - **Componentes:**
      - Função `main()` que coordena todas as etapas acima
      - Condição `if __name__ == "__main__":` para executar a função principal

### **Diagrama Simplificado da Estrutura**

```
train.py
│
├── Importações
│
├── Definição de Hiperparâmetros
│
├── Carregamento de Dados
│
├── Instanciação do Modelo
│
├── Configuração da Função de Perda e Otimizador
│
├── Gerenciamento do Estado de Treinamento
│
├── Loop de Treinamento
│   ├── Iteração sobre Épocas
│   ├── Iteração sobre Batches de Treinamento
│   ├── Cálculo da Perda e Backpropagation
│   └── Atualização dos Pesos
│
├── Validação do Modelo
│
├── Monitoramento e Log
│
├── Salvamento de Checkpoints e Modelos
│
└── Função Principal e Execução do Script
```
