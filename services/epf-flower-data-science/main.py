from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi import APIRouter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import pandas as pd
import pandas as pd
import joblib 
from firebase_admin import credentials, firestore, initialize_app
import uvicorn
cred = credentials.Certificate(r'C:\Users\omars\OneDrive\Bureau\EPF 5A\DATA_SOURCES\TP2_API\EPF-API-TP\services\epf-flower-data-science\credentials.json')
initialize_app(cred)


db = firestore.client()
app = FastAPI()

@app.get("/")
async def redirect_to_swagger():
    return RedirectResponse(url='/docs')

@app.get("/hello")
def read_hello():
    return {"message": "Hello World"}

@app.get("/load-iris")
def load_iris_dataset():
    df = pd.read_csv('src/data/iris.csv')  # Assurez-vous que le chemin vers le fichier est correct
    return df.to_dict(orient='records')

@app.get("/process-iris")
def process_iris_dataset():
    df = pd.read_csv('src/data/iris.csv')
    # Exemple de traitement : normalisation des colonnes numériques
    scaler = StandardScaler()
    df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
    return df.to_dict(orient='records')

@app.get("/train-test-split")
def split_iris_dataset(test_size: float = 0.2):
    df = pd.read_csv('C:/Users/omars/OneDrive/Bureau/EPF 5A/DATA_SOURCES/TP2_API/EPF-API-TP/services/epf-flower-data-science/src/data/IRIS.csv')
    print(df.columns)
    X = df.drop('species', axis=1)  
    y = df['species']  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return {
        "X_train": X_train.to_dict(orient='records'),
        "X_test": X_test.to_dict(orient='records'),
        "y_train": y_train.tolist(),
        "y_test": y_test.tolist()
    }
@app.get("/load-model-params")
def load_model_params():
    with open('C:/Users/omars/OneDrive/Bureau/EPF 5A/DATA_SOURCES/TP2_API/EPF-API-TP/services/epf-flower-data-science/src/config/model_parameters.json', 'r') as file:
        params = json.load(file)
    return params

@app.post("/train-model")
def train_model():
    
    df_train = pd.read_csv('C:/Users/omars/OneDrive/Bureau/EPF 5A/DATA_SOURCES/TP2_API/EPF-API-TP/services/epf-flower-data-science/src/data/IRIS.csv')
    X_train = df_train.drop('species', axis=1)  
    y_train = df_train['species']

    
    with open('C:/Users/omars/OneDrive/Bureau/EPF 5A/DATA_SOURCES/TP2_API/EPF-API-TP/services/epf-flower-data-science/src/config/model_parameters.json', 'r') as file:
        model_params = json.load(file)['LogisticRegression']

    
    model = LogisticRegression(**model_params)
    model.fit(X_train, y_train)
    #with open('C:/Users/omars/OneDrive/Bureau/EPF 5A/DATA_SOURCES/TP2_API/EPF-API-TP/services/epf-flower-data-science/src/models/test.txt', 'w') as f:
    #f.write('Hello, world!')

    # save model
    joblib.dump(model, 'C:/Users/omars/OneDrive/Bureau/EPF 5A/DATA_SOURCES/TP2_API/EPF-API-TP/services/epf-flower-data-science/src/models/logistic_regression_model.joblib')

    return {"message": "Model trained and saved"}

model = joblib.load('C:/Users/omars/OneDrive/Bureau/EPF 5A/DATA_SOURCES/TP2_API/EPF-API-TP/services/epf-flower-data-science/src/models/logistic_regression_model.joblib')

@app.post("/predict")
def make_prediction(input_data: dict):
    try:
        
        input_df = pd.DataFrame([input_data])

        
        prediction = model.predict(input_df)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error of prediction : {e}")
    
@app.get("/get-parameters")
def get_parameters():
    try:
        doc_ref = db.collection("parameters").document("Xured35um6Uw0NGTDyCY")
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            return {"message": "the doc does not exist."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error of access : {e}")
@app.post("/add-parameters")
def add_parameters(new_params: dict):
    try:
        doc_ref = db.collection("parameters").document("new_parameters")
        doc_ref.set(new_params)
        return {"message": "new parameters added succesfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error : {e}")
@app.put("/update-parameters/{param_id}")
def update_parameters(param_id: str, update_values: dict):
    try:
        doc_ref = db.collection("parameters").document(param_id)
        doc_ref.update(update_values)
        return {"message": "parameters updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error : {e}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")

