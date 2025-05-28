from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from model.model import HybridPerformerBiLSTM
from model.dataset import MultiStepDataset
from model.utils import inverse_transform
from model.utils import create_multi_step_targets
from model.utils import select_features_rfe
from model.utils import add_technical_indicators
import torch
import pandas as pd

router = APIRouter()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HORIZON = 6
SEQ_LEN = 120

# Модель и параметры инициализируются один раз
model = None
model_input_features = None

class PredictRequest(BaseModel):
    coin: str
    raw_data: list[dict]

@router.post("/predict")
def predict_price(req: PredictRequest):
    global model, model_input_features
    try:
        df = pd.DataFrame(req.raw_data)

        # 1. Добавим тех. индикаторы
        df = add_technical_indicators(df)

        # 2. Создание цели
        df = create_multi_step_targets(df, target_col='log_return', horizon=HORIZON)
        target_cols = [f'target_t+{i}' for i in range(1, HORIZON + 1)]

        # 3. Отбор признаков
        if model_input_features is None:
            features = select_features_rfe(df, target_col='log_return', n_features=10)
            if 'close' not in features:
                features.append('close')
            model_input_features = features
        else:
            features = model_input_features

        dataset = MultiStepDataset(df, feature_cols=features, target_cols=target_cols, seq_len=SEQ_LEN)
        if len(dataset) == 0:
            raise ValueError("Недостаточно данных для предсказания")

        x_last, _ = dataset[-1]

        # 4. Загрузка модели
        if model is None:
            model = HybridPerformerBiLSTM(input_dim=len(features), horizon=HORIZON)
            model.load_state_dict(torch.load("models/best_hybrid_model.pth", map_location=DEVICE))
            model.to(DEVICE)
            model.eval()

        with torch.no_grad():
            x_tensor = x_last.unsqueeze(0).to(DEVICE)
            pred = model(x_tensor).cpu().numpy()[0]

        scaler = dataset.scaler
        predicted_price = inverse_transform(scaler, pred, feature_index=features.index('close'))[0]

        return {
            "coin": req.coin,
            "predicted_price": round(predicted_price, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка в предсказании: {str(e)}")
