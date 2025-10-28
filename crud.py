# In: backend/crud.py
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import models
import schemas

# Setup for password hashing
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def get_user_by_email(db: Session, email: str):
    """
    Queries the database to find a user by their email address.
    """
    return db.query(models.User).filter(models.User.email == email).first()

def create_user(db: Session, user: schemas.UserCreate):
    """
    Creates a new user in the database with a hashed password.
    """
    hashed_password = pwd_context.hash(user.password)
    db_user = models.User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_profile_by_user_id(db: Session, user_id: int):
    """
    Retrieves a profile by the owner's user ID.
    """
    return db.query(models.Profile).filter(models.Profile.owner_id == user_id).first()

def create_or_update_profile(db: Session, profile_data: schemas.ProfileCreate, user_id: int):
    """
    Creates a new profile if one doesn't exist, or updates an existing one.
    """
    db_profile = get_profile_by_user_id(db, user_id=user_id)
    if db_profile:
        # Update existing profile
        update_data = profile_data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_profile, key, value)
    else:
        # Create new profile
        db_profile = models.Profile(**profile_data.model_dump(), owner_id=user_id)
        db.add(db_profile)
    
    db.commit()
    db.refresh(db_profile)
    return db_profile