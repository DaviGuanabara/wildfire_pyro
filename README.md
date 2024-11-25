
### README: Spatio-Temporal Data Analysis and Simulation Framework

#### **Overview**
This project focuses on spatio-temporal machine learning using deep sets of random neighbors with attention mechanisms. The main purpose is to simulate, analyze, and predict spatio-temporal phenomena based on sparse data collected through fixed or mobile sensors. This framework adheres to the conventions of **Farama Gymnasium**.

#### **Purpose and Context**
As described in the accompanying article, the primary aim is to predict spatial and temporal variables efficiently while incorporating measures of uncertainty. The project demonstrates its utility in various datasets, including simulated data and real-world taxi flow data, through the integration of deep learning with attention mechanisms.

#### **Current Status**
The project is under active development and is not yet functional.

#### **Data Flow**

data.csv -> sensor manager -> gymnasium environment -> model wrapper
---

### **System Components**

#### **1. Data Source**
- Input: A CSV file containing sensor data, with fields for latitude, longitude, timestamp, and observed values.
- Each sensor is uniquely identified by a combination of its latitude and longitude.

#### **2. Data Flow**
**SensorManager** is the core component that:
- **Loads Data**: Reads the CSV file and preprocesses it.
- **Simulates Real-Time**: Manipulates the data to emulate real-time sensor readings.
- **Sensor Identification**: Assigns unique IDs to sensors based on spatial coordinates.
- **Data Access**: Allows selection of specific sensors and their neighbors based on a time window.

#### **3. Simulation Environment**
**Environment** class:
- Adapts data processed by `SensorManager` into the **Farama Gymnasium** format.
- Provides interfaces for sensor interaction, time-step transitions, and data aggregation.

#### **4. AI Model Wrapper**
- Wraps the environment with reinforcement learning models using **Stable Baselines 3**.
- Handles both training (`train`) and prediction (`predict`) phases.
- Integrates seamlessly with neural network models.

#### **5. Deep Learning Model**
Implemented in `model.py`:
- Encodes the architecture using deep sets with an attention mechanism for random neighbor selection.
- Provides uncertainty measures for predictions.

---

### **Features**

1. **Random Neighbor Selection**:
   - Randomly selects neighbors for prediction, ensuring a diverse range of spatial and temporal coverage.
2. **Uncertainty Estimation**:
   - Measures prediction confidence, enabling informed decision-making.
3. **Flexibility**:
   - Supports both fixed and mobile sensor configurations.

---

### **Usage**
#### **Examples**
- Located in the `examples/` directory, demonstrating:
  - Data loading and preprocessing.
  - Simulation setup for training and evaluation.
  - Model training and inference workflows.


---

### **Future Work**


#### **Contact**
