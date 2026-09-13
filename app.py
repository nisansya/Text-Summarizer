from fastapi import FastAPI, Request
import requests
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates  
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


app= FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")


model = T5ForConditionalGeneration.from_pretrained("final_model")
tokenizer = T5Tokenizer.from_pretrained("final_model")


if torch.backends.mps.is_available():
    device= torch.device("mps")

elif torch.cuda.is_available():
    device= torch.device("cuda")

else:
    device= torch.device("cpu")

model.to(device)

templates = Jinja2Templates(directory="templates")


class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text) # lines
    text = re.sub(r"\s+", " ", text) # spaces
    text = re.sub(r"<.*?>", " ", text) # html tags <p> <h1>
    text = text.strip().lower()
    return text

def summarize_dialogue(dialogue :str):
    dialogue= clean_data(dialogue) 
    input_text= "summarize: "+ dialogue 
    
  
    inputs= tokenizer(
        input_text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    
    model.to(device)
    targets= model.generate(
        input_ids= inputs["input_ids"],
        attention_mask= inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )

    
    summary= tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary


@app.post("/summarize/")
async def summarize(dialogue_input:DialogueInput):
    summary= summarize_dialogue(dialogue_input.dialogue)
    return {"summary":summary}
    
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})

