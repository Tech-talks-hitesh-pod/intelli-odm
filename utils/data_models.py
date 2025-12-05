"""Pydantic models for data validation and cleaning."""

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal

class ProductModel(BaseModel):
    """Pydantic model for product data validation."""
    product_id: str = Field(..., description="Unique product identifier")
    name: Optional[str] = Field(None, description="Product name")
    description: Optional[str] = Field(None, description="Product description")
    category: Optional[str] = Field(None, description="Product category")
    material: Optional[str] = Field(None, description="Material type")
    color: Optional[str] = Field(None, description="Color")
    pattern: Optional[str] = Field(None, description="Pattern type")
    sleeve: Optional[str] = Field(None, description="Sleeve type")
    fit: Optional[str] = Field(None, description="Fit type")
    gsm: Optional[float] = Field(None, description="GSM (fabric weight)")
    gender: Optional[str] = Field(None, description="Gender")
    
    @field_validator('product_id')
    @classmethod
    def validate_product_id(cls, v):
        if not v or not v.strip():
            raise ValueError("product_id cannot be empty")
        return v.strip()
    
    @field_validator('gsm')
    @classmethod
    def validate_gsm(cls, v):
        if v is not None and (v < 0 or v > 1000):
            raise ValueError("GSM must be between 0 and 1000")
        return v
    
    class Config:
        extra = "allow"  # Allow additional fields


class SalesRecordModel(BaseModel):
    """Pydantic model for sales data validation."""
    date: datetime = Field(..., description="Sale date")
    store_id: str = Field(..., description="Store identifier")
    sku: str = Field(..., description="Product SKU")
    units_sold: int = Field(..., ge=0, description="Units sold (non-negative)")
    revenue: Decimal = Field(..., ge=0, description="Revenue (non-negative)")
    
    @field_validator('date')
    @classmethod
    def validate_date(cls, v):
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace('Z', '+00:00'))
            except:
                return pd.to_datetime(v)
        return v
    
    @field_validator('units_sold', 'revenue')
    @classmethod
    def validate_non_negative(cls, v):
        if v < 0:
            raise ValueError("Value cannot be negative")
        return v
    
    class Config:
        extra = "allow"


class InventoryRecordModel(BaseModel):
    """Pydantic model for inventory data validation."""
    store_id: str = Field(..., description="Store identifier")
    sku: str = Field(..., description="Product SKU")
    on_hand: int = Field(..., ge=0, description="Units on hand (non-negative)")
    reserved: Optional[int] = Field(0, ge=0, description="Reserved units")
    available: Optional[int] = Field(None, description="Available units")
    
    @model_validator(mode='after')
    def calculate_available(self):
        if self.available is None:
            self.available = self.on_hand - (self.reserved or 0)
        return self
    
    class Config:
        extra = "allow"


class PricingRecordModel(BaseModel):
    """Pydantic model for pricing data validation."""
    store_id: str = Field(..., description="Store identifier")
    sku: str = Field(..., description="Product SKU")
    price: Decimal = Field(..., gt=0, description="Price (must be positive)")
    cost: Optional[Decimal] = Field(None, ge=0, description="Cost")
    margin: Optional[Decimal] = Field(None, description="Margin percentage")
    
    @field_validator('price')
    @classmethod
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError("Price must be positive")
        return v
    
    @model_validator(mode='after')
    def calculate_margin(self):
        if self.cost is not None and self.price > 0:
            if self.margin is None:
                self.margin = ((self.price - self.cost) / self.price) * 100
        return self
    
    class Config:
        extra = "allow"


class StoreModel(BaseModel):
    """Pydantic model for store data validation."""
    store_id: str = Field(..., description="Store identifier")
    name: Optional[str] = Field(None, description="Store name")
    city: Optional[str] = Field(None, description="City")
    state: Optional[str] = Field(None, description="State")
    region: Optional[str] = Field(None, description="Region")
    climate: Optional[str] = Field(None, description="Climate type")
    locality: Optional[str] = Field(None, description="Locality tier")
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Latitude")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Longitude")
    
    class Config:
        extra = "allow"


class ProcessedDataModel(BaseModel):
    """Model for processed/cleaned data output."""
    products: List[ProductModel] = Field(default_factory=list)
    sales: List[SalesRecordModel] = Field(default_factory=list)
    inventory: List[InventoryRecordModel] = Field(default_factory=list)
    pricing: List[PricingRecordModel] = Field(default_factory=list)
    stores: List[StoreModel] = Field(default_factory=list)
    
    # Analysis results
    product_attributes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    validation_summary: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"

