from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
import pandas as pd
import time
from io import StringIO

from src.data_loader import load_and_clean_data
from src.embedding import generate_embeddings
from src.similarity import SimilarityEngine
from src.finetuned_model import finetune_model

app = FastAPI(title="Volunteer Description Matcher")

# Hold engine in memory
engine = None

@app.get("/health")
async def health_check():
    return {"status": "ok"}


# Union[Literal["mpnet", "fine-tuned", "bert"], str] = Query("mpnet")
@app.post("/upload/")
async def upload_csv(
    file: UploadFile = File(...),
    model_name: str = Query(default="all-mpnet-base-v2", enum=["all-mpnet-base-v2", "all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2", "all-distilroberta-v1"]),
    backend: str = Query(default="in_memory", enum=["in_memory", "faiss", "chromadb"])
):
    """
    Upload a CSV of volunteer descriptions and initialize the similarity engine.
    """
    if not file.filename.endswith(".csv"):
        return JSONResponse(content={"error": "Only CSV files are supported"}, status_code=400)

    content = await file.read()
    df = pd.read_csv(StringIO(content.decode("utf-8")))

    try:
        start = time.time()
        df = load_and_clean_data(df)
        print(df.head())
        embeddings = generate_embeddings(df['cleaned_description'].tolist(), model_name=model_name)
        # embeddings = generate_embeddings(df['cleaned_description'].tolist())
        global engine
        engine = SimilarityEngine(df, embeddings, backend=backend)
        duration = time.time() - start
        return {"message": f"Loaded and Embedded {len(df)} volunteer profiles using backend '{backend}' in {duration:2f} seconds."}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/finetuned_upload/")
async def finetuned_upload_csv(
    file: UploadFile = File(...),
    finetuned_model_name: str = Query(),
    backend: str = Query(default="in_memory", enum=["in_memory", "faiss", "chromadb"])
):
    """
    Upload a CSV of volunteer descriptions and initialize the similarity engine.
    """
    if not file.filename.endswith(".csv"):
        return JSONResponse(content={"error": "Only CSV files are supported"}, status_code=400)

    content = await file.read()
    df = pd.read_csv(StringIO(content.decode("utf-8")))

    # try:
    start = time.time()
    df = load_and_clean_data(df)
    print(df.head())
    print('finetune model')
    model_name = finetune_model(finetuned_model_name)
    embeddings = generate_embeddings(df['cleaned_description'].tolist(), model_name=finetuned_model_name)
    # embeddings = generate_embeddings(df['cleaned_description'].tolist())
    global engine
    engine = SimilarityEngine(df, embeddings, backend=backend)
    duration = time.time() - start
    return {"message": f"Loaded and Embedded {len(df)} volunteer profiles using backend '{backend}' in {duration:2f} seconds."}
    # except Exception as e:
    #     return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/query/")
async def query_volunteers(
    description: str = Form(...),
    top_k: int = Form(3),
    model_name: str = Query(default="all-mpnet-base-v2", enum=["all-mpnet-base-v2", "all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2", "all-distilroberta-v1"])
):
    """
    Query the most similar volunteer descriptions to the input.
    """
    start = time.time()
    if engine is None:
        return JSONResponse(content={"error": "No volunteer data loaded. Please upload a CSV first."}, status_code=400)

    try:
        results = engine.query(description, top_k=top_k, model_name=model_name)
        duration = time.time() - start
        return {"query": description, "results": results, "duration": {duration}}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)