from fastapi import FastAPI, Request,File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from tempfile import NamedTemporaryFile
from fastapi import HTTPException
from pathlib import Path
import logging
from datetime import datetime
from summary_sentiment import summarize,sentiment
import uuid, os
import shutil
from src import diarize
from src import *
import json
import pytz



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()
templates = Jinja2Templates(directory="templates/")
app.mount("/static", StaticFiles(directory="static"), name="static")


logger.info("            Loading Diarize Models")
whisper_model, msdd_model, punct_model = diarize.init_models()

logger.info("            Loading Summarization Model")
summ_model = summarize.Model(model_dict = "summary_sentiment/MEETING-SUMMARY-BART-LARGE-XSUM-SAMSUM-DIALOGSUM-AMI")
# summ_model = summarize.Model()

logger.info("            Loading Sentiment Model")
sentiToken, sentiModel = sentiment.load_sentiment_model()
logger.info("            Model Loading Complete")



@app.get("/")
def form_post(request: Request):
    try : 
        result = "Type a number"
        return templates.TemplateResponse('index.html', context={'request': request, 'result': result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/")
def form_post(audioFile: UploadFile = File(...)):
    unique_id = str(uuid.uuid4())
    destination_path  = f'static/data/audios/{unique_id}.wav'
    try:
        if audioFile:
            with NamedTemporaryFile(delete=True) as temp:
                with open(temp.name, 'wb') as temp_file:
                    temp_file.write(audioFile.file.read())
                
                transcript = diarize.process(temp.name,whisper_model,msdd_model, punct_model)
                
                # saving to data/audios/....wav
                shutil.copyfile(temp.name, destination_path) 
            
            logger.info("            Now Summarizing Convesations")
            text = summ_model.clean_text(transcript)
            
            generated_summary = summ_model.summary(text)
            
            if generated_summary!="":
                logger.info("            Summary Generated.")
                
            logger.info("            Sentiment Analysis")
            
            generated_sentiment = sentiment.inference(generated_summary,sentiToken,sentiModel)
            logger.info("            Analysis Done.")
            try:
                with open('static/data/results/result.json', 'r') as file:
                    try:
                        chat_history = json.load(file)
                    except json.JSONDecodeError:
                        chat_history = []
            except FileNotFoundError:
                chat_history = []
            
            india_timezone = pytz.timezone('Asia/Kolkata')
            utc_now = datetime.now(pytz.utc)
            current_time = utc_now.astimezone(india_timezone)
            response = {"id" :unique_id,
                        'Transcript': transcript,
                        "Summary":generated_summary,
                        'Sentiment':generated_sentiment,
                       "DateTime": current_time.strftime('%Y-%m-%d %H:%M:%S') }
            
            chat_history.insert(0,response)
            
            with open("static/data/results/result.json", "w") as file:
                json.dump(chat_history, file)
             

            return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app,port=8051,host="0.0.0.0")