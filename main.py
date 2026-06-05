import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib


app = FastAPI()

data_pkl = joblib.load("model.pkl")
model = data_pkl['model']
ratio = 1.0

class PredictRequest(BaseModel):
    hour: int
    dayofweek: int
    month: int
    day: int
    shopId: int
    cat_id: int
    has_promotion: int
    has_discount: int
    discount_depth: float
    show_discount: int
    item_price_min: int
    item_price_max: int
    priceBeforeDiscount: int
    shop_rating: float
    shop_response_rate: float
    shop_follower_count: int
    is_official_shop: int
    is_verified: int
    review_rating: float
    total_rating_count: int
    cmt_count: int
    is_free_shipping: int
    is_pre_order: int
    model_last_price: int
    model_median_price: float
    model_mean_price: float
    model_count: int

class AnchorItem(BaseModel):
    harga_asli: float
    harga_prediksi: float

class CalibrateRequest(BaseModel):
    anchors: list[AnchorItem]
    
@app.get("/")
def root():
    return {"pesan" : "API prediksi harga siap!"}

@app.post("/predict")
def prediksi(data: PredictRequest):
    X = np.array([[data.hour, data.dayofweek, data.month, data.day,
               data.shopId, data.cat_id, data.has_promotion, data.has_discount,
               data.discount_depth, data.show_discount, data.item_price_min,
               data.item_price_max, data.priceBeforeDiscount, data.shop_rating,
               data.shop_response_rate, data.shop_follower_count, data.is_official_shop,
               data.is_verified, data.review_rating, data.total_rating_count,
               data.cmt_count, data.is_free_shipping, data.is_pre_order,
               data.model_last_price, data.model_median_price,
               data.model_mean_price, data.model_count]])
        
    try:
        if data.model_count < 3:
            pred_log = model.predict(X)
            pred = np.expm1(pred_log)
        else:
            pred = data.model_last_price
    except:
        raise HTTPException(status_code=500, detail="Model gagal melakukan prediksi")
        
    hasil = pred*ratio
    return {"predicted_price": float(hasil) if data.model_count >= 3 else float(hasil[0])}

@app.post("/calibrate")
def kalibrasi(data: CalibrateRequest):
    global ratio
    ratios = [item.harga_asli / item.harga_prediksi for item in data.anchors]
    ratio = np.median(ratios)
    return {"ratio": float(ratio)}

@app.get("/health")
def health():
    if model:
        return {"status": "ok", "model_loaded": True}
    else:
        return {"status": "error", "model_loaded": False}
