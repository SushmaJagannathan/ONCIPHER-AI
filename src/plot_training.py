import pickle
import matplotlib.pyplot as plt


history_file="../models/history.pkl"


with open(history_file,"rb") as f:
    history=pickle.load(f)


plt.figure(figsize=(8,5))

plt.plot(
    history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history["val_accuracy"],
    label="Validation Accuracy"
)


plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "EfficientNetB3 Training Accuracy"
)

plt.legend()

plt.savefig(
    "../results/accuracy_curve.png",
    dpi=300
)


plt.close()



plt.figure(figsize=(8,5))


plt.plot(
    history["loss"],
    label="Training Loss"
)

plt.plot(
    history["val_loss"],
    label="Validation Loss"
)


plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "EfficientNetB3 Training Loss"
)

plt.legend()


plt.savefig(
    "../results/loss_curve.png",
    dpi=300
)


print("Graphs saved")