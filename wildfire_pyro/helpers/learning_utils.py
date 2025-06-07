import os



# ==================================================================================================
# Funções adicionais
# ==================================================================================================


def get_path(file_name):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join(
        "data", "synthetic", "fixed_sensor", file_name)
    data_path = os.path.join(SCRIPT_DIR, relative_path)
    return data_path


def log_evaluation(metrics, info, step):
    print(f"\n--- Step {step} ---")
    print(f">> Evaluating Sensor (ID: {info["sensor"]['sensor_id']})")
    print(
        f"   Location: Latitude {info["sensor"]['lat']:.4f}, Longitude: {info["sensor"]['lon']:.4f}, Ground Truth: {info['ground_truth']:.4f}"
    )
    print(">> Bootstrap Model:")
    print(
        f"   Prediction: {metrics['mean_prediction']:.4f} ± {metrics['std_prediction']:.4f} | Error: {metrics['error']:.4f}"
    )
    print(">> Baseline:")
    print(
        f"   Prediction: {metrics['baseline_prediction']:.4f} ± {metrics['baseline_std']:.4f} | Error: {metrics['baseline_error']:.4f}"
    )
