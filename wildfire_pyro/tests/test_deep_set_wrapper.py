from wildfire_pyro.wrappers.deep_set_attention_net_wrapper import ReplayBuffer, EnvDataCollector, LearningManager
import numpy as np
import pytest
import torch
import pytest
from unittest.mock import Mock, MagicMock, call
from gymnasium import spaces


def test_replay_buffer_initialization():
    batch_size = 4
    observation_shape = (3,)
    action_shape = (2,)
    device = "cpu"

    buffer = ReplayBuffer(
        max_size=batch_size,
        observation_shape=observation_shape,
        action_shape=action_shape,
        device=device,
    )

    assert buffer.max_size == batch_size
    assert buffer.observation_shape == observation_shape
    assert buffer.action_shape == action_shape
    assert buffer.device == device
    assert buffer.size() == 0
    assert not buffer.is_full()
    assert buffer.observations.shape == (batch_size,) + observation_shape
    assert buffer.actions.shape == (batch_size,) + action_shape
    assert buffer.ground_truth.shape == (batch_size, 1)


def test_replay_buffer_add_and_size():
    batch_size = 3
    observation_shape = (4,)
    action_shape = (1,)
    buffer = ReplayBuffer(
        max_size=batch_size,
        observation_shape=observation_shape,
        action_shape=action_shape,
    )

    # Add transitions
    for i in range(batch_size):
        obs = np.array([i, i + 1, i + 2, i + 3])
        action = np.array([i])
        ground_truth = float(i)

        assert not buffer.is_full()

        buffer.add(obs, action, ground_truth)
        assert buffer.size() == i + 1
        

    # After adding max_size transitions
    assert buffer.size() == batch_size
    assert buffer.is_full()


def test_replay_buffer_sample_batch():
    batch_size = 2
    observation_shape = (2,)
    action_shape = (1,)
    buffer = ReplayBuffer(
        max_size=batch_size,
        observation_shape=observation_shape,
        action_shape=action_shape,
    )

    # Attempt to sample before buffer is full
    with pytest.raises(ValueError) as excinfo:
        buffer.sample_batch()
    assert "Buffer is not full yet." in str(excinfo.value)

    # Add transitions
    for i in range(batch_size):
        obs = np.array([i, i + 1])
        action = np.array([i])
        ground_truth = float(i)
        buffer.add(obs, action, ground_truth)

    # Sample batch
    observations, actions, ground_truth_batch = buffer.sample_batch()

    assert observations.shape == (batch_size,) + observation_shape
    assert actions.shape == (batch_size,) + action_shape
    assert ground_truth_batch.shape == (batch_size, 1)

    # Verify contents
    for i in range(batch_size):
        assert torch.equal(
            observations[i], torch.tensor([i, i + 1], dtype=torch.float32)
        )
        assert torch.equal(actions[i], torch.tensor([i], dtype=torch.float32))
        assert torch.equal(
            ground_truth_batch[i], torch.tensor(
                [float(i)], dtype=torch.float32)
        )


def test_replay_buffer_reset():
    batch_size = 2
    observation_shape = (2,)
    action_shape = (1,)
    buffer = ReplayBuffer(
        max_size=batch_size,
        observation_shape=observation_shape,
        action_shape=action_shape,
    )

    # Add transitions
    for i in range(batch_size):
        obs = np.array([i, i + 1])
        action = np.array([i])
        ground_truth = float(i)
        buffer.add(obs, action, ground_truth)

    assert buffer.is_full()

    # Reset the buffer
    buffer.reset()
    assert buffer.size() == 0
    assert not buffer.is_full()

    # Verify that data has been cleared
    for tensor in [buffer.observations, buffer.actions, buffer.ground_truth]:
        assert torch.all(tensor == 0)


def test_replay_buffer_overflow():
    batch_size = 2
    observation_shape = (2,)
    action_shape = (1,)
    buffer = ReplayBuffer(
        max_size=batch_size,
        observation_shape=observation_shape,
        action_shape=action_shape,
    )

    # Add transitions up to max_size
    for i in range(batch_size):
        obs = np.array([i, i + 1])
        action = np.array([i])
        ground_truth = float(i)
        buffer.add(obs, action, ground_truth)

    assert buffer.is_full()

    # Attempt to add another transition should raise RuntimeError
    with pytest.raises(RuntimeError) as excinfo:
        buffer.add(
            np.array([batch_size, batch_size + 1]),
            np.array([batch_size]),
            float(batch_size),
        )
    assert "ReplayBuffer is full. Call `reset` before adding more transitions." in str(
        excinfo.value
    )


def test_replay_buffer_partial_fill():
    batch_size = 5
    observation_shape = (3,)
    action_shape = (2,)
    buffer = ReplayBuffer(
        max_size=batch_size,
        observation_shape=observation_shape,
        action_shape=action_shape,
    )

    # Add fewer transitions than max_size
    for i in range(batch_size - 2):
        obs = np.array([i, i + 1, i + 2])
        action = np.array([i, i + 1])
        ground_truth = float(i)
        buffer.add(obs, action, ground_truth)
        assert buffer.size() == i + 1
        assert not buffer.is_full()

    # Check size and full flag
    assert buffer.size() == batch_size - 2
    assert not buffer.is_full()


def test_replay_buffer_multiple_cycles():
    batch_size = 3
    observation_shape = (2,)
    action_shape = (1,)
    buffer = ReplayBuffer(
        max_size=batch_size,
        observation_shape=observation_shape,
        action_shape=action_shape,
    )

    # Add first batch
    for i in range(batch_size):
        obs = np.array([i, i + 1])
        action = np.array([i])
        ground_truth = float(i)
        buffer.add(obs, action, ground_truth)

    assert buffer.is_full()

    # Sample first batch
    observations1, actions1, ground_truth_batch1 = buffer.sample_batch()

    # Reset buffer
    buffer.reset()

    # Add second batch
    for i in range(batch_size, 2 * batch_size):
        obs = np.array([i, i + 1])
        action = np.array([i])
        ground_truth = float(i)
        buffer.add(obs, action, ground_truth)

    assert buffer.is_full()

    # Sample second batch
    observations2, actions2, ground_truth_batch2 = buffer.sample_batch()

    # Verify that samples are from the second batch
    for i in range(batch_size):
        assert torch.equal(
            observations2[i],
            torch.tensor([i + batch_size, i + batch_size + 1],
                         dtype=torch.float32),
        )
        assert torch.equal(
            actions2[i], torch.tensor([i + batch_size], dtype=torch.float32)
        )
        assert torch.equal(
            ground_truth_batch2[i],
            torch.tensor([float(i + batch_size)], dtype=torch.float32),
        )


def test_env_data_collector_initialization():
    """
    Test that EnvDataCollector initializes correctly with given environment, buffer, and device.
    """
    mock_env = Mock()
    mock_buffer = Mock(spec=ReplayBuffer)
    device = "cuda"

    collector = EnvDataCollector(
        environment=mock_env, buffer=mock_buffer, device=device
    )

    assert collector.environment == mock_env
    assert collector.buffer == mock_buffer
    assert collector.device == device




def test_learning_manager_initialization():
    """
    Test that LearningManager initializes correctly with given neural network, environment, parameters, and batch size.
    """
    # Setup
    mock_env = Mock()
    mock_env.observation_space.shape = (3,)
    mock_env.action_space.shape = (2,)
    mock_env.action_space.__class__ = spaces.Box  # Mocking as Box space

    mock_neural_network = Mock()
    mock_neural_network.parameters.return_value = []

    parameters = {
        "device": "cpu",
        "lr": 1e-3,
        "batch_size": 4,
    }

    # Instantiate LearningManager
    manager = LearningManager(
        neural_network=mock_neural_network,
        environment=mock_env,
        parameters=parameters,
        batch_size=parameters["batch_size"],
    )

    # Assertions
    assert manager.neural_network == mock_neural_network
    assert manager.environment == mock_env
    assert manager.parameters == parameters
    assert manager.batch_size == parameters["batch_size"]
    assert manager.device == parameters["device"]

    # Check ReplayBuffer initialization
    assert manager.buffer.max_size == parameters["batch_size"]
    assert manager.buffer.observation_shape == mock_env.observation_space.shape
    assert manager.buffer.action_shape == mock_env.action_space.shape
    assert manager.buffer.device == parameters["device"]

    # Check EnvDataCollector initialization
    assert manager.data_collector.environment == mock_env
    assert manager.data_collector.buffer == manager.buffer
    assert manager.data_collector.device == parameters["device"]

    # Check optimizer and loss function initialization
    assert isinstance(manager.optimizer, torch.optim.Adam)
    assert isinstance(manager.loss_func, torch.nn.MSELoss)


def test_learning_manager_train_with_full_buffer():
    """
    Test that LearningManager's train method correctly samples data from the buffer and updates the network when buffer is full.
    """
    # Setup
    mock_env = Mock()
    mock_env.observation_space.shape = (3,)
    mock_env.action_space.shape = (2,)
    mock_env.action_space.__class__ = spaces.Box  # Mocking as Box space

    mock_neural_network = Mock()
    mock_neural_network.parameters.return_value = [Mock()]
    mock_neural_network.return_value = torch.tensor(
        [1.0, 2.0], dtype=torch.float32)

    parameters = {
        "device": "cpu",
        "lr": 1e-3,
        "batch_size": 2,
    }

    manager = LearningManager(
        neural_network=mock_neural_network,
        environment=mock_env,
        parameters=parameters,
        batch_size=parameters["batch_size"],
    )

    # Mock ReplayBuffer to be full and return predefined data
    manager.buffer.size = Mock(return_value=2)
    manager.buffer.sample_batch = Mock(
        return_value=(
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                         dtype=torch.float32),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            torch.tensor([[0.5], [1.5]], dtype=torch.float32),
        )
    )

    # Mock neural network forward pass
    manager.neural_network.return_value = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
    )

    # Mock loss function
    manager.loss_func = torch.nn.MSELoss()
    manager.loss_func.return_value = torch.tensor(0.1, requires_grad=True)

    # Mock optimizer step
    manager.optimizer = Mock()

    # Execute
    loss = manager.train()

    # Assertions
    manager.buffer.sample_batch.assert_called_once_with(manager.batch_size)
    manager.optimizer.zero_grad.assert_called_once()
    manager.neural_network.assert_called_once_with(manager.buffer.observations)
    manager.loss_func.assert_called_once_with(
        manager.neural_network.return_value, manager.buffer.ground_truth
    )
    manager.optimizer.step.assert_called_once()
    assert loss == 0.1


def test_learning_manager_train_with_insufficient_data(capfd):
    """
    Test that LearningManager's train method skips training when buffer has insufficient data.
    """
    # Setup
    mock_env = Mock()
    mock_env.observation_space.shape = (3,)
    mock_env.action_space.shape = (2,)
    mock_env.action_space.__class__ = spaces.Box  # Mocking as Box space

    mock_neural_network = Mock()
    mock_neural_network.parameters.return_value = [Mock()]

    parameters = {
        "device": "cpu",
        "lr": 1e-3,
        "batch_size": 4,
    }

    manager = LearningManager(
        neural_network=mock_neural_network,
        environment=mock_env,
        parameters=parameters,
        batch_size=parameters["batch_size"],
    )

    # Mock ReplayBuffer to have insufficient data
    manager.buffer.size = Mock(return_value=2)

    # Execute
    loss = manager.train()

    # Capture printed output
    captured = capfd.readouterr()

    # Assertions
    assert loss == 0.0
    assert (
        "[Warning] Not enough data in buffer to train. Skipping training."
        in captured.out
    )

    # Ensure sample_batch is not called
    manager.buffer.sample_batch.assert_not_called()


def test_learning_manager_learn():
    """
    Test that LearningManager's learn method correctly orchestrates data collection and training over multiple steps.
    """
    # Setup
    mock_env = Mock()
    mock_env.observation_space.shape = (3,)
    mock_env.action_space.shape = (2,)
    mock_env.action_space.__class__ = spaces.Box  # Mocking as Box space

    mock_neural_network = Mock()
    mock_neural_network.parameters.return_value = [Mock()]
    mock_neural_network.predict.return_value = torch.tensor(
        [1.0, 2.0], dtype=torch.float32
    )

    parameters = {
        "device": "cpu",
        "lr": 1e-3,
        "batch_size": 2,
    }

    manager = LearningManager(
        neural_network=mock_neural_network,
        environment=mock_env,
        parameters=parameters,
        batch_size=parameters["batch_size"],
    )

    # Mock EnvDataCollector
    manager.data_collector.collect_rollouts = Mock()

    # Mock ReplayBuffer to be full and return predefined data
    manager.buffer.size = Mock(return_value=2)
    manager.buffer.sample_batch = Mock(
        return_value=(
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                         dtype=torch.float32),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            torch.tensor([[0.5], [1.5]], dtype=torch.float32),
        )
    )

    # Mock neural network forward pass
    manager.neural_network.return_value = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
    )

    # Mock loss function
    manager.loss_func = torch.nn.MSELoss()
    manager.loss_func.return_value = torch.tensor(0.1, requires_grad=True)

    # Mock optimizer step
    manager.optimizer = Mock()

    # Execute
    manager.learn(total_steps=4, rollout_steps=2)

    # Assertions
    # Ensure collect_rollouts is called twice
    assert manager.data_collector.collect_rollouts.call_count == 2
    manager.data_collector.collect_rollouts.assert_has_calls(
        [
            call(neural_network=manager.neural_network, n_rollout_steps=2),
            call(neural_network=manager.neural_network, n_rollout_steps=2),
        ],
        any_order=False,
    )

    # Ensure train is called twice
    assert manager.train.call_count == 2


def test_learning_manager_predict():
    """
    Test that LearningManager's predict method returns the correct action.
    """
    # Setup
    mock_env = Mock()
    mock_env.observation_space.shape = (3,)
    mock_env.action_space.shape = (2,)
    mock_env.action_space.__class__ = spaces.Box  # Mocking as Box space

    mock_neural_network = Mock()
    mock_neural_network.eval = Mock()
    mock_neural_network.return_value = torch.tensor(
        [0.5, 1.5], dtype=torch.float32)

    parameters = {
        "device": "cpu",
        "lr": 1e-3,
        "batch_size": 2,
    }

    manager = LearningManager(
        neural_network=mock_neural_network,
        environment=mock_env,
        parameters=parameters,
        batch_size=parameters["batch_size"],
    )

    # Execute
    obs = np.array([0.1, 0.2, 0.3])
    action, info = manager.predict(obs, deterministic=True)

    # Assertions
    mock_neural_network.eval.assert_called_once()
    mock_neural_network.assert_called_once_with(
        torch.tensor(obs, dtype=torch.float32).to("cpu")
    )
    assert action == [0.5, 1.5]
    assert info is None
