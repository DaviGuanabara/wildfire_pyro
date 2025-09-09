## Learning Manager System Architecture

The Learning Manager abstracts the core mechanics of supervised learning into a clean, modular interface, enabling efficient experimentation, reuse, and extensibility across training pipelines. By encapsulating essential functions within a modular and flexible architecture, it serves as a robust foundation for building and scaling sophisticated machine learning models tailored to specific needs and data characteristics. Features:

- 🎯 **Purpose**: Integrates raw environmental data with neural network inputs, aligning format and semantics.
- 🛠️ **Functionality**: Initializes the neural network with specified parameters and binds it with the data environment for streamlined data flow.
- 🔄 **Batch Processing**: Manages data batches for efficient training using a custom replay buffer that stores and retrieves training samples.
- ⚙️ **Optimization**: Utilizes the `torch.optim.Adam` optimizer for adjusting model weights based on the calculated gradients, enhancing the learning process with efficient backpropagation and weight updates.
- 📉 **Loss Function**: Employs `torch.nn.MSELoss` for the quantification of training performance, guiding the optimization strategy by providing a measure of prediction accuracy.
- 🔄 **Training Loop**: Conducts the training sessions over specified epochs and batches, adjusting learning rates and other parameters dynamically based on the training state.
- 📈 **Performance Monitoring**: Integrates performance metrics tracking within training loops, allowing for real-time monitoring and adjustments.
- 🗃️ **Data Collection**: Data is sourced from the environment, preprocessed, and stored in a replay buffer.
- 🔄 **Batch Sampling**: Data batches are sampled from the buffer for training, ensuring diverse exposure to training samples.

### ⚙️ **Execution Flow: Step by Step**
1. `BaseLearningManager` is initialized with the neural network model under training and the environment.
2. `collect_rollouts()` performs the rollout:
   - Calls `action_provider.get_action(obs)` — by default, the neural network takes the action.
   - Sends `action` to `env.step(action)`.
   - Receives `obs, reward, info`, and extracts the label using `target_provider.get_target(info)`.
3. Stores `(obs, action, target)` into the buffer.
4. `_train()` minimizes the loss between `neural_network(obs)` and the `target`.

### **Supervised Learning Manager Architecture **

The `SupervisedLearningManager` implements the `_train()` function, which processes observations, feeds them through the neural network, and handles outputs for both training feedback and standalone predictions, as show:

```mermaid
graph LR
    A[Observation] --> B[Neural Network]
    B --> C[Training]
    B --> D[Environment]
    D --> A
    E[Ground Truth] --> C
    C --> B
```

#### 🧪 Use Example

```python
manager = SupervisedLearningManager(
    neural_network=student_model,
    environment=env,
    logging_parameters=log_params,
    runtime_parameters=runtime_params,
    model_parameters=model_params,
)
```


### **Distillation Learning Manager Architecture**

The `DistillationLearningManager` is an extension of the `SupervisedLearningManager`, created specifically to support supervised knowledge distillation, in a Teacher-Student design, as shown:

```mermaid
graph LR
    A[Student Observation] --> B[Student (Neural Network)]
    B --> C[Training]
    G --> C

    G --> D[Environment]
    D --> A
    D --> F

    C --> B
    F[Teacher Observation] --> G[Teacher (Neural Network)]
```


- **Teacher-Guided Supervision**: Targets are generated using the teacher network instead of extracted from `info["ground_truth"]`.
- **Modular Targeting**: The `TargetProvider` abstraction allows seamless switching between target generation strategies.
- **No Environment Coupling**: Teacher logic remains decoupled from the environment, improving testability and reusability.
- **Optional RL Fine-Tuning**: The system can be extended with reinforcement learning for further fine-tuning after the supervised phase.


### ✅ **Role Review: Who Acts and Who Learns**

| Component                                       | Role in the Pipeline                        | Notes                                                                                                                            |
| ----------------------------------------------- | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `neural_network` in `SupervisedLearningManager` | Is the the model under training             | ✅ Passed to `BaseLearningManager` and assigned as the default `.provider` of the `action_provider`.                              |
| `action_provider`                               | Determines who acts in the environment      | ✅ Default: `BaseActionProvider` with the `neural_network`. Can be replaced via `self.action_provider = BaseActionProvider(...)`. |
| `target_provider`                               | Supplies the **supervised learning target** | ✅ Default: `"ground_truth"` from the `info` dict. Can be replaced via `self.target_provider = BaseTargetProvider(...)`.          |
| `BaseLearningManager.collect_rollouts()`        | Handles the rollout process                 | Uses `obs → action_provider → action → env.step(action)`, then uses `target_provider` to obtain the training label from `info`.  |
| `SupervisedLearningManager._train()`            | Trains the `neural_network`                 | Trains using `(obs, action, target)` from the replay buffer, with the `target` fetched through the configured `target_provider`. |

#### 🧪 Use Example

```python
manager = DistillationLearningManager(
    student=student_model,
    environment=env,
    logging_parameters=log_params,
    runtime_parameters=runtime_params,
    model_parameters=model_params,
    teacher_nn=teacher_model
)
```
