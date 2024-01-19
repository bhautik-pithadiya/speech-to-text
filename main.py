from fastapi import FastAPI, Request,File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from tempfile import NamedTemporaryFile
from fastapi import HTTPException
from src import diarize_parallel
from src import *
from pathlib import Path


app = FastAPI()
templates = Jinja2Templates(directory="templates/")

whisper_model, punct_model = diarize_parallel.init_models()

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.parent.absolute() / "Speech_to_text_api/static"),
    name="static",
)
@app.get("/")
def form_post(request: Request):
    try : 
        result = "Type a number"
        return templates.TemplateResponse('index.html', context={'request': request, 'result': result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/")
def form_post(audioFile: UploadFile = File(...)):
    try:
        if audioFile:
            with NamedTemporaryFile(delete=True) as temp:
                with open(temp.name, 'wb') as temp_file:
                    temp_file.write(audioFile.file.read())
        
                transcript = diarize_parallel.process(temp.name,whisper_model,punct_model)
            return {'Transcript ': transcript}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
