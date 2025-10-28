# In: backend/main.py

from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
import json
import os

# Import all your new modules
import crud
import models
import schemas
from database import SessionLocal, engine
import auth

# Import the modules for the analysis part (even though it's commented out)
import cv2
import numpy as np
import pytesseract
from dotenv import load_dotenv
from groq import AsyncGroq

# Load environment variables from .env file
load_dotenv()

# This line creates the database tables defined in models.py
models.Base.metadata.create_all(bind=engine)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

# --- CORS Middleware ---
# This allows your frontend to communicate with the backend
origins = [
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dependency to get a DB session ---
# This function creates a database session for each request and closes it when done
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- User Registration Endpoint ---
@app.post("/users/", response_model=schemas.User)
def create_new_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)

# --- NEW: User Login Endpoint ---
@app.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = crud.get_user_by_email(db, email=form_data.username)
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = crud.get_user_by_email(db, email=email)
    if user is None:
        raise credentials_exception
    return user

# --- NEW: Profile Endpoints ---
@app.put("/profile/", response_model=schemas.Profile)
def update_user_profile(
    profile_update: schemas.ProfileUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    return crud.create_or_update_profile(db=db, profile_data=profile_update, user_id=current_user.id)

@app.get("/profile/", response_model=schemas.Profile)
def read_user_profile(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    profile = crud.get_profile_by_user_id(db, user_id=current_user.id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


# --- Root Endpoint ---
@app.get("/")
def read_root():
    return {"message": "Hello, NutriLens Backend!"}

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

async def analyze_text_with_groq(text: str, profile: models.Profile = None):
    
    # Create the user profile string, or a default message if no profile exists.
    profile_details = "The user has not provided a health profile. Provide a general analysis."
    if profile:
        profile_details = (
            f"USER PROFILE: Goal='{profile.primary_goal}', "
            f"Health Conditions='{profile.health_conditions}', Allergies='{profile.allergies}'."
        )

    # --- THE NEW, SMARTER PROMPT ---
    system_prompt = """
    You are an expert nutrition analyst AI. Your primary function is to receive messy, unstructured text
    extracted from a food nutrition label via Optical Character Recognition (OCR) and a user's health profile.
    Your task is to transform this information into a structured, personalized, and data-driven analysis.

    Your response MUST be a single, valid JSON object and nothing else. Do not include any introductory text,
    explanations, or markdown formatting like json.

    1. Core Analysis Rules:
    - Credibility: Your assessment of nutrient levels (e.g., 'high sodium', 'moderate sugar') MUST be based on
    dietary guidelines from the World Health Organization (WHO) and the Food Safety and Standards Authority of India (FSSAI).
    Explicitly mention the source in the reasoning and references.
    - Objectivity: Use a neutral, factual tone. Avoid words like 'bad', 'unhealthy', or 'terrible'.
    Use phrases like 'This product is high in...' or 'This may not be suitable for...'.
    - Safety: You are not a medical professional. Do not provide medical advice. Use cautious language such as
    'This may not be suitable for individuals with...'.
    - Personalization: The explanation in the summary must directly link nutrient information to the user’s
    specific health goals and conditions (e.g., high sodium → hypertension).
    - Source Transparency: Every claim about a nutrient level must be backed by a clear reference to WHO or FSSAI
    guidelines in the reasoning field and/or references section.

    2. Input Format:
    - User Profile: A string containing the user's health goals, health conditions, and allergies.
    - OCR Text: A string of messy, unformatted text from the food label.

    3. Required JSON Output Structure:
    {
      "health_rating": "A" | "B" | "C" | "D" | "E" | "F",
      "summary": "A fully expanded, structured paragraph with three parts:
           1. Verdict — a clear, one-sentence personalized conclusion explicitly mentioning the user's health conditions and goals.
           2. Explanation — include all relevant nutrients from the OCR text, providing specific values, % of daily recommended limits, and explicitly citing WHO or FSSAI for each.
           3. Positive Notes — mention all beneficial nutrients and their effects. Example: For individuals with hypertension, this product may not be suitable due to high sodium content (650 mg per serving), which exceeds 30% of the WHO daily limit of 2000 mg. Sugar content is low (2g per serving), suitable for blood sugar management. It provides 22g of protein and 5g of fiber, supporting muscle maintenance and digestive health. Saturated fat is low (3g per serving), supporting heart health.",
      "pros": ["Array of positive nutritional aspects"],
      "cons": [
          {"nutrient": "sodium", "level": "High", "source": "WHO", "reasoning": "650 mg per serving is over 30% of the 2000 mg daily limit recommended by WHO.", "value": "650 mg"},
          "... other nutrients ..."
      ],
      "nutrient_levels": {
          "sodium": {"level": "High", "source": "WHO", "reasoning": "650 mg per serving is over 30% of the 2000 mg daily limit recommended by WHO.", "value": "650 mg"},
          "sugar": {"level": "Low", "source": "FSSAI", "reasoning": "2g per serving is below the 25g daily limit recommended by FSSAI.", "value": "2g"},
          "... other nutrients ..."
      },
      "references": [
          "World Health Organization (WHO) — Sodium, Protein, and Saturated Fat Intake Guidelines",
          "Food Safety and Standards Authority of India (FSSAI) — Sugar, Fiber, and Fat Intake Guidelines"
      ]
    }

    4. Additional Instructions:
    - Do not include 'product_name' in the output.
    - The summary MUST include all relevant nutrients from the OCR text with numeric justification and citations (WHO or FSSAI).
    - Maintain the 3-part summary structure: Verdict, Explanation, Positive Notes.
    - The reasoning field under each nutrient in 'nutrient_levels' must include numeric justification (e.g., % of daily limit).
    - The pros and cons arrays must be factual and reference nutrient values.
    - In the 'references' array, group nutrients by organization to avoid repetition. Format: 'Organization — Nutrient1, Nutrient2 Intake Guidelines.'
    - Do not invent numbers. Only use values from OCR text or official guidelines.
    - Always maintain personalization, data-driven logic, and transparent citations.
    """

    try:
        client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
        model_name = os.environ.get("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
        
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the nutrition label text:\n{text}"},
            ],
            model=model_name,
            temperature=0.3, # A little more creative for a good summary
            response_format={"type": "json_object"},
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f'{{"error": f"An error occurred with the AI analysis: {e}"}}'
def preprocess_image(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return processed_img

@app.post("/analyze-label/")
async def analyze_label(file: UploadFile = File(...)):
    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)
    extracted_text = pytesseract.image_to_string(processed_image)
    raw_analysis = await analyze_text_with_groq(extracted_text)
    
    try:
        json_analysis = json.loads(raw_analysis)
        return json_analysis
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned malformed data.")
