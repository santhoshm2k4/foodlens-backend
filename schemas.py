# In: backend/schemas.py

from pydantic import BaseModel
from typing import Optional

# --- User schemas remain the same ---
class UserCreate(BaseModel):
    email: str
    password: str

class User(BaseModel):
    id: int
    email: str
    class Config:
        from_attributes = True

# --- NEW, UPGRADED PROFILE SCHEMAS ---
class ProfileBase(BaseModel):
    age: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    gender: Optional[str] = None
    activity_level: Optional[str] = None
    primary_goal: Optional[str] = None
    health_conditions: Optional[str] = None
    allergies: Optional[str] = None

class ProfileCreate(ProfileBase):
    pass

class ProfileUpdate(ProfileBase):
    pass

class Profile(ProfileBase):
    id: int
    owner_id: int
    class Config:
        from_attributes = True