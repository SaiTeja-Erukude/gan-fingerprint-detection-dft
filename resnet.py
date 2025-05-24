import os
import numpy                    as np
import matplotlib.pyplot        as plt
from keras                      import layers, models
from keras.applications         import ResNet50
from keras.preprocessing.image  import ImageDataGenerator
from keras.optimizers           import Adam
from keras.callbacks            import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


######################
# Configuration
######################
HEIGHT, WIDTH = 256, 256
BATCH_SIZE = 32
EPOCHS = 50
DATA_PATH = "data_path"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Create output directory for model artifacts
OUTPUT_DIR = "output_dir_path"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load ResNet50 model with pre-trained ImageNet weights
base_model = ResNet50(
    weights="imagenet", 
    include_top=False, 
    input_shape=(HEIGHT, WIDTH, 3)
)

# Fine-tuning approach: freeze early layers but make later layers trainable
# Freeze the first 80% of the layers
for layer in base_model.layers[:int(len(base_model.layers)*0.8)]:
    layer.trainable = False
for layer in base_model.layers[int(len(base_model.layers)*0.8):]:
    layer.trainable = True

# Create a custom model on top of ResNet50 with dropout for regularization
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(2, activation="softmax")
])

# Compile the model with a lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy", 
    metrics=["accuracy", "AUC"]
)

# Print the model summary
model.summary()

# Set up ImageDataGenerators with data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,         # Rotation
    width_shift_range=0.1,     # Shifts
    height_shift_range=0.1,    # Shifts
    zoom_range=0.1,            # Gentler zoom
    horizontal_flip=True,      # Flipping
    fill_mode="reflect",
    validation_split=0.18
)

# Validation data should only be rescaled, not augmented
test_datagen = ImageDataGenerator(
    rescale=1./255
)

# Flow from directory with split
train_generator = train_datagen.flow_from_directory(
    f"{DATA_PATH}/train",
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=RANDOM_SEED
)

validation_generator = train_datagen.flow_from_directory(
    f"{DATA_PATH}/train",
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=RANDOM_SEED
)

# Independent test set for final evaluation
test_generator = test_datagen.flow_from_directory(
    f"{DATA_PATH}/test",
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Calculate steps_per_epoch based on dataset size
steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = validation_generator.samples // BATCH_SIZE

# Set up callbacks
callbacks = [
    # Save best model based on validation accuracy
    ModelCheckpoint(
        filepath=os.path.join(OUTPUT_DIR, "best_model.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    # Stop training when improvement stops
    EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when plateau is reached
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_acc, test_auc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Save the final model
model_name = f"resnet50"
model.save(os.path.join(OUTPUT_DIR, f"{model_name}.h5"))

# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"))
plt.show()

# Function to create a confusion matrix
def plot_confusion_matrix(model, test_generator):
    import sklearn.metrics as metrics
    import seaborn as sns
    
    # Get predictions
    test_generator.reset()
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Calculate confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=test_generator.class_indices.keys(),
                yticklabels=test_generator.class_indices.keys())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.show()
    
    # Calculate precision, recall, and F1 score
    precision = metrics.precision_score(y_true, y_pred_classes, average="weighted")
    recall = metrics.recall_score(y_true, y_pred_classes, average="weighted")
    f1 = metrics.f1_score(y_true, y_pred_classes, average="weighted")
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

plot_confusion_matrix(model, test_generator)