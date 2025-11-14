# In: backend/main.py

from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import JWTError, jwt
import json
import os
import asyncio

# Import all your new modules
import crud
import models
import schemas
from database import SessionLocal, engine
import auth

# Import the modules for the analysis part
import cv2
import numpy as np
import pytesseract
from dotenv import load_dotenv
from groq import AsyncGroq
import google.generativeai as genai
from PIL import Image
import io

# Path for local Windows Tesseract installation (Comment out for deployment)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load environment variables from .env file
load_dotenv()

# This line creates the database tables defined in models.py
models.Base.metadata.create_all(bind=engine)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

# --- CORS Middleware ---
origins = [
    "http://localhost:3000",
    "https://foodlens-app-six.vercel.app" # Example production URL
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dependency to get a DB session ---
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

# --- User Login Endpoint ---
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

# --- Get Current User Dependency ---
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

# --- Profile Endpoints ---
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

# --- Helper Functions (Image/AI) ---

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

# Configure the Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

async def get_product_info(image_bytes: bytes):
    """
    Analyzes the product image (not the label) using Gemini
    to identify the product name.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        prompt = "Identify the full product name in this image. Respond with ONLY the product name and nothing else."
        response = await model.generate_content_async([prompt, img])
        product_name = response.text.strip().replace("\n", " ").replace("*", "")
        
        if not product_name:
            return "Unknown Product"
            
        return product_name
        
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "Analysis Error"

# --- NEW FUNCTION 1: Extracts raw data from text ---
async def extract_data_from_text(text: str):
    """
    First AI call: Uses Groq to extract raw data from OCR text.
    """
    system_prompt = """
    You are a data extraction robot. Your only job is to find specific data from the user's text and return it as a valid JSON object. Do not add any explanation.
    
    1.  **Find Nutrients:** Find ALL nutrients mentioned (e.g., Sodium, Sugar, Fiber, Protein, Saturated Fat) and their exact values (e.g., "650mg", "12g").
    2.  **Find Category:** Infer the product category (e.g., "Snack", "Cereal").
    3.  **Find Certifications:** Find ALL certifications (e.g., "FSSAI", "USDA Organic"). If none, return [].

    Return a JSON object with keys: "nutrients" (an object of key/value pairs), "category", "certifications".
    
    Example Output:
    {
      "nutrients": {
        "Sodium": "650mg",
        "Total Sugars": "12g",
        "Dietary Fiber": "4g",
        "Protein": "10g"
      },
      "category": "Snack",
      "certifications": ["FSSAI"]
    }
    """
    
    try:
        client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
        model_name = os.environ.get("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
        
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the text:\n{text}"},
            ],
            model=model_name,
            temperature=0.0, # Low temperature for factual extraction
            response_format={"type": "json_object"},
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"Groq Extraction Error: {e}")
        return {"error": "Failed to extract nutrient data."}


# --- NEW FUNCTION 2: Generates personalized analysis from the extracted data ---
async def get_personalized_analysis(extracted_data: dict, profile: models.Profile):
    """
    Second AI call: Uses Groq to analyze the extracted data and user profile.
    """
    profile_details = "The user has not provided a health profile. Provide a general analysis."
    if profile:
        profile_details = (
            f"USER PROFILE: Goal='{profile.primary_goal}', "
            f"Health Conditions='{profile.health_conditions}', Allergies='{profile.allergies}'."
        )

    system_prompt = f"""
    You are an expert nutrition analyst AI. You will be given a JSON object of nutrient data and a user profile.
    Your task is to write the analysis and return a valid JSON object.

    ---
    CRITICAL ANALYSIS RULES
    ---
    1.  **JSON ONLY:** Your entire response MUST be a single, valid JSON object.
    2.  **CREDIBILITY:** Base assessments on WHO (Sodium < 2000mg, Sat Fat < 20g, Protein 50g) and FSSAI (Added Sugar < 25g, Fiber 25g).
    3.  **IMPACT (PERCENTAGE RULE):** Your reasoning MUST explain the 'per serving' amount as a **percentage of the daily limit**. (e.g., "650mg per serving (32% of 2000mg WHO daily limit).")
    4.  **PERSONALIZATION:** The 'summary' MUST connect findings (e.g., "high sodium") to the user's profile (e.g., "Hypertension").
    5.  **COMPLETENESS:** Analyze ALL nutrients provided in the input JSON.

    ---
    USER PROFILE
    ---
    {profile_details}
    
    ---
    REQUIRED JSON OUTPUT STRUCTURE
    ---
    1.  `health_rating`: (String) A single letter grade (A-F).
    2.  `summary`: (String) A 3-part paragraph: 1. Personalized Verdict. 2. Explanation (with percentages). 3. Positive Notes.
    3.  `pros`: (Array of Strings) ALL positive findings (e.g., "Good source of fiber (4g per serving, 16% of 25g FSSAI daily limit)").
    4.  `cons`: (Array of Objects) ALL negative findings. Each object: `nutrient`, `level`, `source`, `reasoning` (with percentage), `value`.
    5.  `nutrient_levels`: (Object) An entry for EVERY nutrient from the input. Each entry: `level`, `source`, `reasoning` (with percentage), `value`.
    6.  `references`: (Array of Strings) List of the guidelines used.
    """

    try:
        client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
        model_name = os.environ.get("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
        
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the nutrient data:\n{json.dumps(extracted_data)}"},
            ],
            model=model_name,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"Groq Analysis Error: {e}")
        return {"error": "Failed to generate personalized analysis."}


# --- NEW Main Analysis Endpoint ---
@app.post("/analyze-label/", response_model=dict)
async def analyze_label(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
    file_label: UploadFile = File(...),
    file_product: UploadFile = File(...)
):
    
    # 1. Read bytes from both files
    label_bytes = await file_label.read()
    product_bytes = await file_product.read()

    # 2. Fetch the user's profile
    user_profile = crud.get_profile_by_user_id(db, user_id=current_user.id)

    # 3. Process the label image for OCR
    processed_image = preprocess_image(label_bytes)
    extracted_text = pytesseract.image_to_string(processed_image)
    
    # --- NEW: Call AI in a chain ---
    
    # 4. Task 1: Extract raw data from text
    extracted_data = await extract_data_from_text(extracted_text)
    if extracted_data.get("error"):
        raise HTTPException(status_code=500, detail=extracted_data.get("error"))

    # 5. Task 2: Get product name from image (Gemini)
    # 6. Task 3: Get personalized analysis from extracted data (Groq)
    #    Run these two in parallel
    product_name_task = get_product_info(product_bytes)
    analysis_task = get_personalized_analysis(extracted_data, profile=user_profile)
    
    product_name, final_analysis = await asyncio.gather(
        product_name_task,
        analysis_task
    )

    if final_analysis.get("error"):
        raise HTTPException(status_code=500, detail=final_analysis.get("error"))

    # 7. Combine all results into one final JSON
    final_analysis['product_name'] = product_name
    final_analysis['category'] = extracted_data.get('category', 'Unknown')
    final_analysis['certifications'] = extracted_data.get('certifications', [])
    
    # Optional: Print the final combined object for debugging
    # print(f"--- FINAL COMBINED JSON --- \n {final_analysis} \n --- END ---")
    
    return final_analysis