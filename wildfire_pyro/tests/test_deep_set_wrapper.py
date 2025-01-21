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
        buffer.add(obs, action, ground_truth)
        assert buffer.size() == i + 1
        assert not buffer.is_full()

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


def test_collect_rollouts_success():
    """
    Test that collect_rollouts successfully collects transitions and adds them to the buffer.
    """
    # Setup
    mock_env = Mock()
    mock_buffer = Mock(spec=ReplayBuffer)
    mock_nn = Mock()
    device = "cpu"

    # Mock environment.reset() to return initial observation and info with
    # ground_truth
    initial_obs = np.array([0.0, 1.0, 2.0])
    mock_env.reset.return_value = (initial_obs, {"ground_truth": 0.5})

    # Mock neural_network.predict to return a valid action
    predicted_action = np.array([1.0])
    mock_nn.predict.return_value = torch.tensor(
        predicted_action, dtype=torch.float32)

    # Mock environment.step to return new observation, reward, done,
    # truncated, and info
    new_obs = np.array([1.0, 2.0, 3.0])
    mock_env.step.return_value = (
        new_obs, 1.0, False, False, {"ground_truth": 1.0})

    collector = EnvDataCollector(
        environment=mock_env, buffer=mock_buffer, device=device
    )

    # Execute
    n_rollout_steps = 2
    collector.collect_rollouts(mock_nn, n_rollout_steps)

    # Verify
    expected_calls = [
        call.reset(),
        call.predict(torch.tensor(
            initial_obs, device=device, dtype=torch.float32)),
        call.add(initial_obs, predicted_action, 0.5),
        call.step(predicted_action),
        call.reset(),
    ]
    # Since n_rollout_steps=2 and after first step, done=False, so second step uses new_obs
    # However, the mock_env.step returns done=False, so it should not reset after first step
    # Therefore, the expected calls should be:
    # reset, predict, add, step, predict, add, step
    # But according to the mock, step always returns done=False, so no reset

    # Adjusting expected_calls accordingly
    expected_calls = [
        call.reset(),
        call.predict(torch.tensor(
            initial_obs, device=device, dtype=torch.float32)),
        call.add(initial_obs, predicted_action, 0.5),
        call.step(predicted_action),
        call.predict(torch.tensor(
            new_obs, device=device, dtype=torch.float32)),
        call.add(new_obs, predicted_action, 1.0),
        call.step(predicted_action),
    ]

    assert mock_env.method_calls == expected_calls

    # Verify buffer.add called twice
    assert mock_buffer.add.call_count == 2
    mock_buffer.add.assert_has_calls(
        [call(initial_obs, predicted_action, 0.5),
         call(new_obs, predicted_action, 1.0)]
    )


def test_collect_rollouts_missing_ground_truth():
    """
    Test that collect_rollouts handles missing ground_truth by ending the rollout.
    """
    # Setup
    mock_env = Mock()
    mock_buffer = Mock(spec=ReplayBuffer)
    mock_nn = Mock()
    device = "cpu"

    # Mock environment.reset() to return initial observation and info without
    # ground_truth
    initial_obs = np.array([0.0, 1.0, 2.0])
    mock_env.reset.return_value = (initial_obs, {})

    # Execute
    collector = EnvDataCollector(
        environment=mock_env, buffer=mock_buffer, device=device
    )

    # Capture printed warnings
    from io import StringIO
    import sys

    captured_output = StringIO()
    sys.stdout = captured_output

    n_rollout_steps = 3
    collector.collect_rollouts(mock_nn, n_rollout_steps)

    # Restore stdout
    sys.stdout = sys.__stdout__

    # Verify
    # Since ground_truth is None, it should print a warning and stop
    assert (
        "[Warning] Missing ground_truth. Ending rollout." in captured_output.getvalue()
    )

    # Verify that buffer.add was never called
    mock_buffer.add.assert_not_called()


def test_collect_rollouts_episode_termination():
    """
    Test that collect_rollouts resets the environment when an episode terminates.
    """
    # Setup
    mock_env = Mock()
    mock_buffer = Mock(spec=ReplayBuffer)
    mock_nn = Mock()
    device = "cpu"

    # Mock environment.reset() to return initial observation and info with
    # ground_truth
    initial_obs = np.array([0.0, 1.0, 2.0])
    mock_env.reset.return_value = (initial_obs, {"ground_truth": 0.5})

    # Mock neural_network.predict to return a valid action
    predicted_action = np.array([1.0])
    mock_nn.predict.return_value = torch.tensor(
        predicted_action, dtype=torch.float32)

    # Define side effects for environment.step
    # First step: not done
    # Second step: done=True
    new_obs = np.array([1.0, 2.0, 3.0])
    mock_env.step.side_effect = [
        (new_obs, 1.0, False, False, {"ground_truth": 1.0}),  # First step
        (new_obs, 1.0, True, False, {"ground_truth": 1.0}),  # Second step
    ]

    collector = EnvDataCollector(
        environment=mock_env, buffer=mock_buffer, device=device
    )

    # Execute
    n_rollout_steps = 3
    collector.collect_rollouts(mock_nn, n_rollout_steps)

    # Verify
    # After second step, done=True, so it should reset
    expected_calls = [
        call.reset(),
        call.predict(torch.tensor(
            initial_obs, device=device, dtype=torch.float32)),
        call.add(initial_obs, predicted_action, 0.5),
        call.step(predicted_action),
        call.predict(torch.tensor(
            new_obs, device=device, dtype=torch.float32)),
        call.add(new_obs, predicted_action, 1.0),
        call.step(predicted_action),
        call.reset(),
    ]

    assert mock_env.method_calls == expected_calls

    # Verify buffer.add called twice
    assert mock_buffer.add.call_count == 2
    mock_buffer.add.assert_has_calls(
        [call(initial_obs, predicted_action, 0.5),
         call(new_obs, predicted_action, 1.0)]
    )


def test_collect_rollouts_buffer_overflow():
    """
    Test that collect_rollouts raises an error when buffer is full.
    """
    # Setup
    mock_env = Mock()
    mock_buffer = Mock(spec=ReplayBuffer)
    mock_nn = Mock()
    device = "cpu"

    # Configure buffer to raise RuntimeError when add is called beyond capacity
    mock_buffer.add.side_effect = [
        None,
        None,
        RuntimeError(
            "ReplayBuffer is full. Call `reset` before adding more transitions."
        ),
    ]

    # Mock environment.reset() to return initial observation and info with
    # ground_truth
    initial_obs = np.array([0.0, 1.0, 2.0])
    mock_env.reset.return_value = (initial_obs, {"ground_truth": 0.5})

    # Mock neural_network.predict to return a valid action
    predicted_action = np.array([1.0])
    mock_nn.predict.return_value = torch.tensor(
        predicted_action, dtype=torch.float32)

    # Mock environment.step to always return done=False
    new_obs = np.array([1.0, 2.0, 3.0])
    mock_env.step.return_value = (
        new_obs, 1.0, False, False, {"ground_truth": 1.0})

    collector = EnvDataCollector(
        environment=mock_env, buffer=mock_buffer, device=device
    )

    # Execute
    n_rollout_steps = 3

    with pytest.raises(RuntimeError) as excinfo:
        collector.collect_rollouts(mock_nn, n_rollout_steps)

    assert "ReplayBuffer is full. Call `reset` before adding more transitions." in str(
        excinfo.value
    )

    # Verify that buffer.add was called twice before raising
    assert mock_buffer.add.call_count == 3  # Two successful adds and one failure


def test_collect_rollouts_multiple_rollouts():
    """
    Test that collect_rollouts can handle multiple consecutive rollouts without issues.
    """
    # Setup
    mock_env = Mock()
    mock_buffer = Mock(spec=ReplayBuffer)
    mock_nn = Mock()
    device = "cpu"

    # Mock environment.reset() to return different observations and
    # ground_truth
    initial_obs_1 = np.array([0.0, 1.0, 2.0])
    initial_info_1 = {"ground_truth": 0.5}
    initial_obs_2 = np.array([10.0, 11.0, 12.0])
    initial_info_2 = {"ground_truth": 5.0}

    # Define side effects for reset and step
    mock_env.reset.side_effect = [
        (initial_obs_1, initial_info_1),
        (initial_obs_2, initial_info_2),
    ]

    # Mock neural_network.predict to return valid actions
    predicted_action_1 = np.array([1.0])
    predicted_action_2 = np.array([2.0])
    mock_nn.predict.side_effect = [
        torch.tensor(predicted_action_1, dtype=torch.float32),
        torch.tensor(predicted_action_2, dtype=torch.float32),
    ]

    # Mock environment.step to always return done=False
    new_obs_1 = np.array([1.0, 2.0, 3.0])
    new_obs_2 = np.array([11.0, 12.0, 13.0])
    mock_env.step.side_effect = [
        (new_obs_1, 1.0, False, False, {
         "ground_truth": 1.0}),  # First rollout step
        # Second rollout step
        (new_obs_2, 2.0, False, False, {"ground_truth": 2.0}),
    ]

    collector = EnvDataCollector(
        environment=mock_env, buffer=mock_buffer, device=device
    )

    # Execute first rollout
    n_rollout_steps = 2
    collector.collect_rollouts(mock_nn, n_rollout_steps)

    # Verify first rollout
    expected_calls_first = [
        call.reset(),
        call.predict(torch.tensor(
            initial_obs_1, device=device, dtype=torch.float32)),
        call.add(initial_obs_1, predicted_action_1, 0.5),
        call.step(predicted_action_1),
        call.predict(torch.tensor(
            new_obs_1, device=device, dtype=torch.float32)),
        call.add(new_obs_1, predicted_action_1, 1.0),
        call.step(predicted_action_1),
    ]
    assert mock_env.method_calls[:7] == expected_calls_first

    # Execute second rollout
    collector.collect_rollouts(mock_nn, n_rollout_steps)

    # Verify second rollout
    expected_calls_second = [
        call.reset(),
        call.predict(torch.tensor(
            initial_obs_2, device=device, dtype=torch.float32)),
        call.add(initial_obs_2, predicted_action_2, 5.0),
        call.step(predicted_action_2),
        call.predict(torch.tensor(
            new_obs_2, device=device, dtype=torch.float32)),
        call.add(new_obs_2, predicted_action_2, 2.0),
        call.step(predicted_action_2),
    ]
    assert mock_env.method_calls[7:] == expected_calls_second

    # Verify buffer.add called four times in total
    assert mock_buffer.add.call_count == 4
    mock_buffer.add.assert_has_calls(
        [
            call(initial_obs_1, predicted_action_1, 0.5),
            call(new_obs_1, predicted_action_1, 1.0),
            call(initial_obs_2, predicted_action_2, 5.0),
            call(new_obs_2, predicted_action_2, 2.0),
        ],
        any_order=False,
    )


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
