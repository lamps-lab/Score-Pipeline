"""
This is the main FastAPI application definition. Endpoints currently defined include:
1. /health: To verify that the API is up and running | returns a JSON response {"Status": "Alive"}
2. /claim: Endpoint used to evaluate a claim/paper through the SCORE application. Note that this endpoint accepts
    a single PDF as input for processing.
3. /claim_meta: Evaluate a claim/paper through the SCORE application while using the supplied metadata file for the DOI,
    ISSN.
4. /train-market: Trains agents on the dataset uploaded in CSV format and saves weights to path specified in
    Market/config.py
"""

from fastapi import Depends, FastAPI, UploadFile, File, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from driver import get_score, train_market
from typing import Optional
import uvicorn
import secrets
import yaml

config_file = open(r'app_config.yaml')
config = yaml.safe_load(config_file)
auth = config['APP']['AUTH']
windows_pdftotext_path = config['PIPELINE']['PDFTOTEXT_PATH']

app = FastAPI()

security = HTTPBasic()


@app.get("/health")
def alive():
    return {"Status": "Alive"}


@app.post("/claim")
async def process_claim(file: UploadFile = File(...), creds: HTTPBasicCredentials = Depends(security)):
    """
    Processes an uploaded PDF through the pipeline and market to generate the evaluation result

    :param file: SBS Paper to be evaluated in PDF format
    :param creds: Basic authentication credentials which should be included in the request payload

    :return: JSON formatted payload with the SCORE and Market interpretability result
    """
    if secrets.compare_digest(creds.username, auth['USER']) \
            and secrets.compare_digest(creds.password, auth['PASSWORD']):
        score = get_score(file)
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return score


@app.post("/claim-meta")
async def process_claim(file: UploadFile = File(...), meta_file: Optional[UploadFile] = File(None),
                        creds: HTTPBasicCredentials = Depends(security)):
    """
    Processes an uploaded PDF through the pipeline and market while using the supplied metadata file
    to initialize the meta features used in the feature pipeline as opposed to using GROBID results.

    :param file: SBS Paper to be evaluated in PDF format
    :param meta_file: Metadata CSV file in the format with identifiers matching the format supplied by OSF

    :return: JSON formatted payload with the SCORE and Market interpretability result
    """
    if secrets.compare_digest(creds.username, auth['USER']) and \
            secrets.compare_digest(creds.password, auth['PASSWORD']):
        if not meta_file:
            score = get_score(file)
        else:
            score = get_score(file, meta_file)
        return score
    else:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )


@app.post("/train-market")
async def train_agents(dataset: UploadFile = File(...), creds: HTTPBasicCredentials = Depends(security)):
    """
    Accepts a CSV upload of a dataset to be used for training. It is assumed that the last column is the label.

    :param dataset: Dataset to be used for training in CSV format
    :param creds: Basic authentication credentials which should be included in the request payload

    :return: JSON response indicating whether market agents were trained successfully
    """
    if secrets.compare_digest(creds.username, auth['USER']) \
            and secrets.compare_digest(creds.password, auth['PASSWORD']):
        training_status = train_market(dataset)
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return training_status

# Uncomment to test in debugger
# uvicorn.run(app, host="127.0.0.1", port=8000)
