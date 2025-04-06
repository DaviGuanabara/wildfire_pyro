# Model Architecture: Flows and Modules

This document describes the model architecture with a focus on data flow and modular organization. The model is designed to process spatial neighborhoods and capture local dependencies using Multi-Layer Perceptrons (MLPs). It is divided into three main modules:

1. **\( \text{MLP}_\phi \):** Generates latent representations for neighborhood data.
2. **\( \text{MLP}_\theta \):** Computes attention weights for the neighborhoods.
3. **\( \text{MLP}_\omega \):** Produces the final prediction by aggregating the processed information.

The interaction between these modules follows the flow described in the paper and is illustrated in the diagram below.

## **Reference Diagrams**

![Model Flow](figures/neural_network_architecture.png)

---

## **General Model Flow**

The data flow in the model unfolds in three main stages:

1. **Latent Representation:**  
   The \( \text{MLP}_\phi \) processes neighborhood vectors to generate rich latent representations, capturing spatial and temporal relationships.

2. **Attention Calculation:**  
   The \( \text{MLP}_\theta \) applies importance weights to neighborhood elements using an attention mechanism. This mechanism prioritizes more relevant information, in line with the idea that closer elements (in time and space) have greater influence.

3. **Final Prediction:**  
   The \( \text{MLP}_\omega \) combines the weighted representations and produces the final prediction.

---

## **Background: Attention Mechanism**

The attention mechanism, widely used in machine learning, assigns different weights to inputs based on their relevance. It is inspired by the human ability to focus on important aspects of a task while ignoring less significant details. In the context of this model:

- Elements that are closer in space and time are considered more important, in accordance with **Tobler’s First Law of Geography**:  
  “Everything is related to everything else, but near things are more related than distant things.”
- Attention is computed by \( \text{MLP}_\theta \), which produces normalized weights using a Softmax function, highlighting the most relevant elements.

---

## **Module Descriptions**

### **\( \text{MLP}_\phi \): Latent Representation**
The \( \text{MLP}_\phi \) transforms neighborhood inputs into latent representations. It is composed of **three sequential blocks**, each containing the following steps:
- **Batch Normalization:** Normalizes input values.
- **Linear Transformation:** Projects the data into a higher-dimensional space.
- **Tanh Activation:** Introduces non-linearity to capture complex relationships.
- **Dropout:** Prevents overfitting by randomly zeroing out neurons.

These blocks are connected using residual sums, enabling the preservation of critical information and efficient gradient propagation. The final output of \( \text{MLP}_\phi \) is the sum of all intermediate block outputs.

---

### **\( \text{MLP}_\theta \): Attention Calculation**
The \( \text{MLP}_\theta \) uses the same modular structure as \( \text{MLP}_\phi \) to calculate attention weights. It receives the same input and returns a set of weights representing the relevance of each neighborhood element.

#### **How does attention work?**
1. Each neighborhood element is processed to produce a score.
2. These scores are normalized using Softmax so that their sum equals 1.
3. The final weights indicate the relative importance of each element.

This process ensures that the most relevant data has a stronger impact on the model's output, while less significant elements are attenuated.

---

### **\( \text{MLP}_\omega \): Final Prediction**
The \( \text{MLP}_\omega \) combines the outputs of \( \text{MLP}_\phi \) (latent representations) and the weights from \( \text{MLP}_\theta \) to generate the final prediction. It applies the weights computed by \( \text{MLP}_\theta \) to the representations and performs the following operations:

1. **Aggregation:** Combines the weighted information from each neighborhood.
2. **Linear Transformation:** Projects the aggregated output into the final space.
3. **Regression:** Outputs the target value, such as position or intensity of an event.

