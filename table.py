%matplotlib inline
import matplotlib.pyplot as plt


logs = trainer.state.log_history  # тут те же записи, что были бы в trainer_state.json

steps = []
loss = []
lr = []
epochs = []

for log in logs:
    if "loss" in log:
        steps.append(log["step"])
        loss.append(log["loss"])
        lr.append(log.get("learning_rate", 0.0))
        epochs.append(log.get("epoch", 0.0))

# если epoch нет, нормализуем по шагам
if not epochs or all(e == 0.0 for e in epochs):
    max_step = max(steps) if steps else 1
    epochs = [s / max_step for s in steps]

plt.style.use("default")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# 1) Training Loss
ax1.plot(epochs, loss, linewidth=2)
ax1.set_title("Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True, alpha=0.3)

# 2) Learning Rate
ax2.plot(epochs, lr, linewidth=2)
ax2.set_title("Learning Rate")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Learning rate")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
