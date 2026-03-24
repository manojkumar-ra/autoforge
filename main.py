from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import pandas as pd
import pickle
from dotenv import load_dotenv

load_dotenv()

from database import init_db, save_run, get_history
from data_analyzer import analyze_dataset, detect_task_type
from preprocessor import preprocess_data
from trainer import train_and_compare
from explainer import explain_results

app = FastAPI(title="AutoForge", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


@app.on_event("startup")
def startup():
    print("starting autoforge...")
    init_db()


@app.get("/health")
def health():
    return {"status": "ok", "message": "AutoForge is running!"}


# keep the uploaded data in memory for now
_current_data = {}


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        content = await file.read()

        filepath = os.path.join("uploads", file.filename)
        with open(filepath, 'wb') as f:
            f.write(content)

        from io import BytesIO
        df = pd.read_csv(BytesIO(content))

        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Dataset too small, need at least 10 rows")

        if len(df.columns) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 columns")

        _current_data['df'] = df
        _current_data['filename'] = file.filename

        analysis = analyze_dataset(df)

        return {
            "success": True,
            "filename": file.filename,
            "analysis": analysis
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class TrainRequest(BaseModel):
    target_column: str


@app.post("/train")
def train_model(req: TrainRequest):
    if 'df' not in _current_data:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet. Upload a CSV first.")

    df = _current_data['df']
    filename = _current_data.get('filename', 'unknown.csv')

    if req.target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.target_column}' not found in dataset")

    try:
        task_type = detect_task_type(df, req.target_column)
        print(f"task type: {task_type}")

        X, y, prep_info, error = preprocess_data(df, req.target_column, task_type)

        if error:
            raise HTTPException(status_code=400, detail=error)

        print(f"preprocessed: {X.shape[0]} rows, {X.shape[1]} features")

        training_results = train_and_compare(X, y, task_type)

        dataset_info = {
            "total_rows": len(df),
            "total_columns": len(df.columns)
        }

        explanation = explain_results(training_results, dataset_info, req.target_column)

        best_score = training_results.get("best_score", 0)
        run_id = save_run(
            filename, req.target_column, task_type,
            training_results.get("best_model", "Unknown"),
            best_score, len(df), prep_info["final_features"],
            training_results.get("model_path", "")
        )

        return {
            "success": True,
            "run_id": run_id,
            "task_type": task_type,
            "preprocessing": prep_info,
            "training": training_results,
            "explanation": explanation,
            "filename": filename
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"training error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class PredictRequest(BaseModel):
    data: dict


@app.post("/predict")
def predict(req: PredictRequest):
    # find the latest trained model
    models_dir = os.path.join(os.path.dirname(__file__), "trained_models")
    if not os.path.exists(models_dir):
        raise HTTPException(status_code=400, detail="No trained model found. Train a model first.")

    model_files = sorted(os.listdir(models_dir), reverse=True)
    if not model_files:
        raise HTTPException(status_code=400, detail="No trained model found. Train a model first.")

    try:
        model_path = os.path.join(models_dir, model_files[0])
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        input_df = pd.DataFrame([req.data])

        prediction = model.predict(input_df)

        return {
            "success": True,
            "prediction": prediction[0] if len(prediction) > 0 else None,
            "model_used": model_files[0]
        }

    except Exception as e:
        print(f"prediction error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/history")
def history():
    results = get_history()
    return {"history": results, "count": len(results)}


if __name__ == "__main__":
    import uvicorn
    print("starting autoforge on port 8002...")
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
