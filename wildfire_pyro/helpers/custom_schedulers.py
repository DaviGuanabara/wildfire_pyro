class DebugScheduler:
    """
    DebugScheduler is a simple learning rate scheduler that prints all received parameters
    and returns a fixed learning rate. Useful for debugging learning rate scheduling pipelines.

    Example usage:
        scheduler = DebugScheduler(lr=1e-3, verbose=True)
        lr = scheduler(step=100, loss=0.04)
    """

    def __init__(self, lr: float = 1e-3, verbose: bool = False):
        self.lr = lr
        self.verbose = verbose

    def __call__(
        self,
        step=None,
        progress_remaining=None,
        loss=None,
        evaluation_metrics=None,
        **kwargs,
    ) -> float:
        if self.verbose:
            print("\n[DEBUG LR SCHEDULER]")
            print(f"  Step: {step}")
            print(f"  Progress Remaining: {progress_remaining}")
            print(f"  Loss: {loss}")
            print(f"  Evaluation Metrics: {evaluation_metrics}")
            for k, v in kwargs.items():
                print(f"  {k}: {v}")

        return self.lr
