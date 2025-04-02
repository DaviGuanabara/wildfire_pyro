# Wildfire Pyro Framework Documentation

## 🔥 Overview
Wildfire Pyro é um framework de simulação e aprendizado supervisionado para **fenômenos espaço-temporais**, com forte inspiração em arquiteturas de aprendizado por reforço. Ele permite treinar modelos de deep learning com mecanismos de atenção sobre conjuntos de vizinhos aleatórios (**deep sets**) para prever variáveis físicas a partir de sensores fixos ou móveis.

A arquitetura é modular, orientada a experimentos e compatível com **Farama Gymnasium**, permitindo integração futura com agentes RL ou sistemas multiagentes.

---

## 📁 Estrutura Principal do Projeto

```
wildfire_pyro/
├── environments/           # Ambientes Gym compatíveis com dados espaço-temporais
│   └── sensor_environment.py
├── common/                 # Componentes reutilizáveis
│   ├── callbacks.py        # Avaliação do modelo durante o treinamento
│   ├── tensorboard.py      # Logger simplificado com fallback
├── models/
│   ├── deep_set_attention_net.py           # Modelo DeepSet com atenção
│   └── deep_set_attention_net_wrapper.py   # Wrapper estilo SB3
├── factories/              # Criação de agentes a partir de parâmetros
│   └── learner_factory.py
├── examples/               # Exemplos de uso (scripts executáveis)
│   ├── fixed_sensor_environment_example.py
│   └── learning_example.py
└── utils/                  # Utilitários (ainda em construção)
```

---

## 🎮 Environment API (SensorEnvironment)

Baseado no Gymnasium, o ambiente simula leituras sensoriais ao longo do tempo.

### Principais métodos:
- `reset()`: Inicializa o ambiente e retorna a primeira observação.
- `step(action)`: Envia uma predição do modelo e avança para o próximo sensor.
- `get_bootstrap_observations(n)`: Retorna `n` conjuntos de vizinhos aleatórios para bootstrap.
- `baseline()`: Estima a variável de interesse usando um método de referência (ex: kNN). Pode ser sobrescrito.

Este design permite separar a lógica de dados da lógica de avaliação/modelo.

---

## 🧠 Modelo: Deep Set Attention Net

O modelo implementa a arquitetura de **Deep Sets** com mecanismo de **atenção**.

### Características:
- Suporta entrada com número variável de vizinhos.
- Pode ser treinado com qualquer função de erro (ex: MSE, MAE).
- Utiliza dropout e camadas densas com ativação ReLU.

A interação com o ambiente é mediada por um **wrapper estilo Stable-Baselines3**, permitindo chamadas como `.train()` e `.predict()`.

---

## 🎯 Avaliação: EvalCallback

`EvalCallback` é um **callback de avaliação supervisionada** compatível com o laço de aprendizado.

### Principais responsabilidades:
- Avaliar o modelo a cada `eval_freq` passos.
- Utilizar **bootstrap** para estimar média e desvio padrão do erro.
- Comparar o desempenho do modelo com um **baseline** definido pelo ambiente.
- Registrar métricas em **TensorBoard** e salvar o **melhor modelo**.

### Métrica Especial: `model_over_baseline`
```python
(abs(baseline_error) - abs(model_error)) / (abs(baseline_error) + epsilon)
```
Esta métrica foi normalizada para:
- Ser robusta a baselines perfeitos (evita divisão por zero com epsilon).
- Permitir comparação entre múltiplas execuções (ex: tuning de hiperparâmetros).
- Mostrar valores positivos quando o modelo é melhor que o baseline.
- Ser clipada em [-1, 1] para estabilidade e melhor visualização.

---

## 🧩 Decisões Arquiteturais

### Estilo agente-ambiente (mesmo para supervisão)
Adotamos a semântica de laço interativo para facilitar avaliação incremental, integração com RL, e estruturação de simulações.

### Baseline acoplado ao ambiente
Cada ambiente define seu método `baseline()`, para comparação justa e contextualizada ao domínio do problema.

### Avaliação por bootstrap
Usamos amostras aleatórias dos vizinhos para gerar predições variadas e estimar incerteza.

### Logging estruturado e modular
- Suporte a TensorBoard com fallback via `NoOpWriter`.
- Dados salvos em `.npz`, com previsão de exportação futura para CSV.

---

## 🚧 Status Atual
- Avaliação com bootstrap funcional.
- Callback com logging e comparação com baseline.
- Modelo com arquitetura Deep Set + Attention.
- Treinamento supervisionado em laço interativo.

A documentação futura incluirá instruções para:
- Otimização de hiperparâmetros.
- Exportação estruturada de dados para análise offline.
- Suporte a novos ambientes e sensores móveis.

---

## 📚 Referências Técnicas
- Zaheer et al., *Deep Sets*, 2017.
- Vaswani et al., *Attention is All You Need*, 2017.
- Stable-Baselines3: Arquitetura de agentes RL moderna e modular.

---

## 👨‍💻 Exemplos Rápidos
```bash
# Rodar ambiente com sensores fixos
python examples/fixed_sensor_environment_example.py

# Treinar modelo supervisionado (em desenvolvimento)
python examples/learning_example.py
```

---

## 📌 Observação Final
Este projeto foi concebido com foco na clareza, modularidade e extensibilidade, equilibrando práticas modernas de engenharia de software com necessidades reais de pesquisa em IA espaço-temporal.

