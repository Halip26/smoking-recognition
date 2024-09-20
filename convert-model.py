from keras.models import load_model
from keras.saving import save_model

# Load your model
model = load_model("weights/smoking.h5")

# Save the model in .keras format
save_model(model, "smoking.h5")
