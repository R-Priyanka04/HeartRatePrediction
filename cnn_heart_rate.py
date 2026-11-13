import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------
# 1. REPRODUCIBILITY AND CONFIGURATION
# --------------------------------------------------

np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cpu")
SAMPLE_RATE = 30
DURATION = 10
NUM_SAMPLES = SAMPLE_RATE * DURATION

# --------------------------------------------------
# 2. GENERATE SYNTHETIC TRAINING SIGNALS
# --------------------------------------------------
# These signals are artificial examples for testing
# the CNN architecture, NOT real patient measurements.

def generate_signal(bpm):
    t = np.arange(NUM_SAMPLES) / SAMPLE_RATE
    frequency = bpm / 60.0

    phase = np.random.uniform(0, 2 * np.pi)

    # Simulated pulse waveform with harmonics.
    signal = (
        np.sin(2 * np.pi * frequency * t + phase)
        + 0.25 * np.sin(
            4 * np.pi * frequency * t + phase
        )
    )

    # Simulate noise, lighting variation and drift.
    signal += np.random.normal(0, 0.25, NUM_SAMPLES)
    signal += 0.15 * np.sin(2 * np.pi * 0.15 * t)

    # Normalize each signal.
    signal = (signal - signal.mean()) / (
        signal.std() + 1e-8
    )

    return signal.astype(np.float32)


def create_dataset(count):
    signals = []
    labels = []

    for _ in range(count):
        bpm = np.random.uniform(45, 180)
        signals.append(generate_signal(bpm))
        labels.append(bpm)

    x = torch.tensor(
        np.array(signals), dtype=torch.float32
    ).unsqueeze(1)

    y = torch.tensor(
        np.array(labels), dtype=torch.float32
    ).unsqueeze(1)

    return x, y


# --------------------------------------------------
# 3. DEFINE THE 1D CNN MODEL
# --------------------------------------------------

class HeartRateCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=7,
                padding=3
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)


# --------------------------------------------------
# 4. TRAIN THE CNN
# --------------------------------------------------

def main():
    print("Generating synthetic training data...")

    x_train, y_train = create_dataset(4000)
    x_test, y_test = create_dataset(500)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=64,
        shuffle=True
    )

    model = HeartRateCNN().to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )

    epochs = 20

    print("\nTraining 1D CNN...")
    print(f"Device: {DEVICE}")
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for signals, labels in train_loader:
            signals = signals.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(signals)
            loss = criterion(predictions, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(signals)

        average_loss = total_loss / len(x_train)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"| Training MSE: {average_loss:.2f}"
        )

    # --------------------------------------------------
    # 5. EVALUATE ON HELD-OUT SYNTHETIC SIGNALS
    # --------------------------------------------------

    model.eval()

    with torch.no_grad():
        predictions = model(x_test).squeeze(1)
        actual = y_test.squeeze(1)

        mae = torch.mean(
            torch.abs(predictions - actual)
        ).item()

        print("\nSynthetic test results")
        print(f"Mean Absolute Error: {mae:.2f} BPM")

        for i in range(5):
            print(
                f"Actual: {actual[i].item():.1f} BPM | "
                f"Predicted: {predictions[i].item():.1f} BPM"
            )

    # --------------------------------------------------
    # 6. SAVE THE TRAINED MODEL
    # --------------------------------------------------

    torch.save(
        model.state_dict(),
        "heart_rate_cnn_synthetic.pth"
    )

    print("\nCNN training completed!")
    print("Model saved: heart_rate_cnn_synthetic.pth")
    print(
        "WARNING: This model was trained on synthetic "
        "signals only. Do not use its predictions as "
        "real heart-rate measurements."
    )


if __name__ == "__main__":
    main()
