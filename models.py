from sqlalchemy import Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    profile = relationship("Profile", back_populates="owner", uselist=False)

# --- NEW, UPGRADED PROFILE MODEL ---
class Profile(Base):
    __tablename__ = "profiles"
    id = Column(Integer, primary_key=True, index=True)
    
    # Biometrics
    age = Column(Integer, nullable=True)
    weight = Column(Float, nullable=True) # in kg
    height = Column(Float, nullable=True) # in cm
    gender = Column(String, nullable=True)
    activity_level = Column(String, nullable=True) # e.g., 'Sedentary', 'Active'
    
    # Goals
    primary_goal = Column(String, nullable=True) # e.g., 'Lose Weight', 'Gain Muscle'
    
    # Health Info (stored as comma-separated strings for simplicity)
    health_conditions = Column(String, nullable=True) # e.g., 'Diabetes,Hypertension'
    allergies = Column(String, nullable=True) # e.g., 'Peanuts,Gluten'
    
    # Foreign Key to link to a User
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="profile")