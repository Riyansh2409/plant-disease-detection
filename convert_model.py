import tensorflow as tf

# load old model
model = tf.keras.models.load_model(
    "model/plant_disease_model.h5",
    compile=False
)

# save in new format
model.save("model/model.keras")

print("✅ Model converted successfully!")