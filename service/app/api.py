from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.model.model import HybridPerformerBiLSTM
from app.model.dataset import MultiStepDataset
from app.model.utils import inverse_transform
from app.data.preprocessing import create_multi_step_targets
from app.data.features import select_features_rfe
import torch
import pandas as pd

router = APIRouter()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HORIZON = 6
SEQ_LEN = 120

class PredictRequest(BaseModel):
    coin: str
    raw_data: list[dict]

@router.post("/predict")
def predict_price(req: PredictRequest):
    try:
        df = pd.DataFrame(req.raw_data)
        df = create_multi_step_targets(df, target_col='log_return', horizon=HORIZON)
        target_cols = [f'target_t+{i}' for i in range(1, HORIZON + 1)]

        features = select_features_rfe(df, target_col='log_return', n_features=10)
        if 'close' not in features:
            features.append('close')

        dataset = MultiStepDataset(df, feature_cols=features, target_cols=target_cols, seq_len=SEQ_LEN)
        last_x, _ = dataset[-1]
        model = HybridPerformerBiLSTM(input_dim=len(features), horizon=HORIZON)
        model.load_state_dict(torch.load("models/best_hybrid_model.pth", map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            pred = model(last_x.unsqueeze(0).to(DEVICE)).cpu().numpy()[0]

        scaler = dataset.scaler
        real = inverse_transform(scaler, pred, feature_index=features.index('close'))
        return {"coin": req.coin, "predicted_prices": real.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))