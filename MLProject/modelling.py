import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train():
    print("Training College Graduation Rate Model")

    # Load Data
    try:
        url = 'https://drive.google.com/uc?id=1lo9nw60UZf0KiMMZwJHgXN_czIlQgTTS'
        df = pd.read_csv(url)
        print("Dataset berhasil dimuat.")
    except Exception as e:
        print(f"Gagal memuat dataset. {e}")
        return

    # Pemisahan Fitur dan Target
    target = 'Grad.Rate'
    drop_cols = [target, 'Grad_Category_Medium_Grad_Rate', 'Grad_Category_High_Grad_Rate']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model Definition
    best_params = {
        'n_estimators': 150,
        'max_depth': 20,          
        'min_samples_split': 2,   
        'random_state': 42
    }
    
    print(f"Melatih model dengan parameter: {best_params}")
    
    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
        
    print(f"Metrics -> MAE: {mae:.4f}, R2-Score: {r2:.4f}")

    # Manual Logging
    mlflow.log_params(best_params)
    mlflow.log_metrics({"MAE": mae, "R2_Score": r2})
        
    # Log Model dengan Signature
    signature = mlflow.models.infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(model, "model_college_grad_rate", signature=signature)

if __name__ == "__main__":
    train()
