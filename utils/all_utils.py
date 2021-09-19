import os

def prepare_data(df):
  X = df.drop("y", axis=1)

  y = df["y"]

  return X, y

 def save_model(model,filename):
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    filePath = os.path.join(model_dir,filename)
    joblib.dump(model,filename)