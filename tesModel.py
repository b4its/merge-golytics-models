import os
import json
import torch
import joblib
import numpy as np
import pickle
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Import the model class from train.py
from train import BisnisAssistantModel

# Base directory where models are stored
BASE_DIR = "final"

# ------------------- LOAD MODELS ------------------- #
def load_models():
    print("🔄 Loading models...")
    models = {}
    
    # Load Neural Network model
    try:
        nn_model = BisnisAssistantModel()
        nn_model.load_state_dict(torch.load(f"{BASE_DIR}/model/integrated_neural_network.pth", 
                                           map_location=torch.device('cpu')))
        nn_model.eval()
        models["neural_network"] = nn_model
        print("✅ Neural Network model loaded")
    except Exception as e:
        print(f"⚠️ Error loading Neural Network model: {e}")
    
    # Load scalers
    try:
        scaler_x = joblib.load(f"{BASE_DIR}/data/normData/scaler_x.pkl")
        scaler_y = joblib.load(f"{BASE_DIR}/data/normData/scaler_y.pkl")
        models["scaler_x"] = scaler_x
        models["scaler_y"] = scaler_y
        print("✅ Scalers loaded")
    except Exception as e:
        print(f"⚠️ Error loading scalers: {e}")
    
    # Load SBERT model
    try:
        sbert_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        models["sbert"] = sbert_model
        print("✅ SBERT model loaded")
    except Exception as e:
        print(f"⚠️ Error loading SBERT model: {e}")
    
    # Load GPT-2 model
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(f"{BASE_DIR}/text_models/gpt2_business_qa")
        gpt2_model = GPT2LMHeadModel.from_pretrained(f"{BASE_DIR}/text_models/gpt2_business_qa")
        tokenizer.pad_token = tokenizer.eos_token
        models["gpt2"] = gpt2_model
        models["tokenizer"] = tokenizer
        print("✅ GPT-2 model loaded")
    except Exception as e:
        print(f"⚠️ Error loading GPT-2 model: {e}")
        try:
            # Fallback to standard GPT-2
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            models["gpt2"] = gpt2_model
            models["tokenizer"] = tokenizer
            print("✅ Fallback to standard GPT-2 model")
        except Exception as e2:
            print(f"⚠️ Error loading fallback GPT-2 model: {e2}")
    
    # Load time series models
    time_series_models = {}
    stores = ['A', 'B', 'C', 'D']
    for store in stores:
        try:
            arima_path = f"{BASE_DIR}/models/arima_store_{store}.pkl"
            sarima_path = f"{BASE_DIR}/models/sarima_store_{store}.pkl"
            
            if os.path.exists(arima_path):
                with open(arima_path, 'rb') as f:
                    time_series_models[f"arima_store_{store}"] = pickle.load(f)
            
            if os.path.exists(sarima_path):
                with open(sarima_path, 'rb') as f:
                    time_series_models[f"sarima_store_{store}"] = pickle.load(f)
        except Exception as e:
            print(f"⚠️ Error loading time series model for Store {store}: {e}")
    
    if time_series_models:
        models["time_series"] = time_series_models
        print(f"✅ Time series models loaded: {len(time_series_models)} models")
    
    # Load classification models
    try:
        dt_model = joblib.load(f"{BASE_DIR}/models/decision_tree_model.pkl")
        rf_model = joblib.load(f"{BASE_DIR}/models/random_forest_model.pkl")
        class_labels = joblib.load(f"{BASE_DIR}/models/classification_labels.pkl")
        
        models["decision_tree"] = dt_model
        models["random_forest"] = rf_model
        models["class_labels"] = class_labels
        print("✅ Classification models loaded")
    except Exception as e:
        print(f"⚠️ Error loading classification models: {e}")
    
    return models

# ------------------- ENTITY EXTRACTION ------------------- #
def extract_entities(text):
    text = text.lower()
    
    # Time detection
    if "hari ini" in text or "sekarang" in text:
        waktu = "hari_ini"
    elif "kemarin" in text:
        waktu = "kemarin"
    elif "minggu ini" in text or "sepekan ini" in text:
        waktu = "minggu_ini"
    elif "bulan ini" in text:
        waktu = "bulan_ini"
    elif "tahun ini" in text:
        waktu = "tahun_ini"
    else:
        waktu = "all"  # fallback to all data
    
    # Target detection
    if "modal" in text:
        target = "modal"
    elif "rugi" in text or "kerugian" in text or "defisit" in text:
        target = "rugi"
    elif "untung" in text or "profit" in text or "laba" in text or "keuntungan" in text:
        target = "profit"
    else:
        target = "profit"  # default target
    
    # Store detection
    if "toko a" in text or "store a" in text:
        store = "A"
    elif "toko b" in text or "store b" in text:
        store = "B"
    elif "toko c" in text or "store c" in text:
        store = "C"
    elif "toko d" in text or "store d" in text:
        store = "D"
    else:
        store = "A"  # default store
    
    # Intent detection
    if "prediksi" in text or "forecast" in text or "perkiraan" in text:
        intent = "forecast"
    elif "klasifikasi" in text or "kelompokkan" in text:
        intent = "classify"
    elif "bandingkan" in text or "compare" in text:
        intent = "compare"
    else:
        intent = "general"
    
    return {
        "waktu": waktu,
        "target": target,
        "store": store,
        "intent": intent
    }

# ------------------- GET DATA BY TIME ------------------- #
def get_data_by_time(waktu_target="hari_ini"):
    matched = []
    norm_dir = f"{BASE_DIR}/data/normData"
    now = datetime.now()
    
    # Check if directory exists
    if not os.path.exists(norm_dir):
        # Create sample data
        pemasukan = np.random.uniform(100000, 500000)
        pengeluaran = np.random.uniform(50000, 300000)
        jam = now.hour / 24.0
        return pemasukan, pengeluaran, jam
    
    for file in os.listdir(norm_dir):
        if not file.endswith(".json") or file == "normalization_stats.json":
            continue
        
        try:
            with open(os.path.join(norm_dir, file)) as f:
                data = json.load(f)
            
            for item in data:
                try:
                    waktu = datetime.fromisoformat(item["waktu"])
                    if waktu_target == "hari_ini" and waktu.date() == now.date():
                        matched.append(item)
                    elif waktu_target == "kemarin" and waktu.date() == (now.date() - timedelta(days=1)):
                        matched.append(item)
                    elif waktu_target == "minggu_ini" and waktu.isocalendar()[1] == now.isocalendar()[1]:
                        matched.append(item)
                    elif waktu_target == "bulan_ini" and waktu.month == now.month and waktu.year == now.year:
                        matched.append(item)
                    elif waktu_target == "tahun_ini" and waktu.year == now.year:
                        matched.append(item)
                    elif waktu_target == "all":
                        matched.append(item)
                except:
                    continue
        except:
            continue
    
    if not matched:
        # Create sample data if no matches
        pemasukan = np.random.uniform(100000, 500000)
        pengeluaran = np.random.uniform(50000, 300000)
        jam = now.hour / 24.0
    else:
        pemasukan = np.mean([item["total_pemasukan"] for item in matched])
        pengeluaran = np.mean([item["total_pengeluaran"] for item in matched])
        jam = np.mean([datetime.fromisoformat(item["waktu"]).hour / 24.0 for item in matched])
    
    return pemasukan, pengeluaran, jam

# ------------------- PREDICT FROM NEURAL NETWORK ------------------- #
def predict_from_neural_network(models, pemasukan, pengeluaran, jam_float):
    if "neural_network" not in models or "scaler_x" not in models or "scaler_y" not in models:
        return {
            "modal": 50000,
            "profit": pemasukan - pengeluaran,
            "rugi": 0
        }
    
    input_data = np.array([[pemasukan, pengeluaran, jam_float]], dtype=np.float32)
    input_scaled = models["scaler_x"].transform(input_data)
    
    with torch.no_grad():
        pred_scaled = models["neural_network"](torch.tensor(input_scaled, dtype=torch.float32)).numpy()
    
    pred = models["scaler_y"].inverse_transform(pred_scaled)[0]
    return {
        "modal": pred[0],
        "profit": pred[1],
        "rugi": pred[2]
    }

# ------------------- PREDICT FROM TIME SERIES ------------------- #
def predict_time_series(models, store="A", steps=7):
    if "time_series" not in models:
        return None
    
    time_series_models = models["time_series"]
    
    # Try SARIMA first, then ARIMA
    model_key = f"sarima_store_{store}"
    if model_key not in time_series_models:
        model_key = f"arima_store_{store}"
    
    if model_key not in time_series_models:
        return None
    
    try:
        # Get forecast
        forecast = time_series_models[model_key].forecast(steps=steps)
        return forecast.tolist()
    except Exception as e:
        print(f"Error in time series prediction: {e}")
        return None

# ------------------- GENERATE TEXT RESPONSE ------------------- #
def generate_text_response(models, prompt, max_length=100):
    if "gpt2" not in models or "tokenizer" not in models:
        return "Maaf, model text generation tidak tersedia."
    
    try:
        # Prepare input
        input_text = f"Bisnis Assistant: {prompt}"
        input_ids = models["tokenizer"].encode(input_text, return_tensors="pt")
        
        # Generate response
        output = models["gpt2"].generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=models["tokenizer"].eos_token_id
        )
        
        # Decode and clean up response
        response = models["tokenizer"].decode(output[0], skip_special_tokens=True)
        response = response.replace(input_text, "").strip()
        
        return response
    except Exception as e:
        print(f"Error in text generation: {e}")
        return "Maaf, terjadi kesalahan dalam menghasilkan respons."

# ------------------- HANDLE USER QUERY ------------------- #
def handle_query(models, query):
    # Extract entities
    entities = extract_entities(query)
    
    # Handle different intents
    if entities["intent"] == "forecast":
        # Time series forecast
        forecast = predict_time_series(models, store=entities["store"])
        if forecast:
            forecast_text = ", ".join([f"Hari {i+1}: Rp {val:,.0f}" for i, val in enumerate(forecast)])
            return f"📊 Prediksi penjualan untuk Toko {entities['store']} 7 hari ke depan:\n{forecast_text}"
        else:
            # Fallback to neural network
            pemasukan, pengeluaran, jam = get_data_by_time(entities["waktu"])
            hasil = predict_from_neural_network(models, pemasukan, pengeluaran, jam)
            return f"📈 Prediksi untuk {entities['waktu'].replace('_', ' ')}:\n→ {entities['target'].capitalize()} Anda diperkirakan sebesar Rp {hasil[entities['target']]:,.0f}"
    
    elif entities["intent"] == "classify" or entities["intent"] == "compare":
        # Generate text response for classification or comparison
        response = generate_text_response(models, query)
        return response
    
    else:
        # General query - use neural network
        pemasukan, pengeluaran, jam = get_data_by_time(entities["waktu"])
        hasil = predict_from_neural_network(models, pemasukan, pengeluaran, jam)
        
        # Enhance with text generation
        basic_response = f"📊 Prediksi untuk {entities['waktu'].replace('_', ' ')}:\n→ {entities['target'].capitalize()} Anda diperkirakan sebesar Rp {hasil[entities['target']]:,.0f}"
        
        # Try to generate additional advice
        try:
            advice_prompt = f"Berikan saran bisnis untuk {entities['target']} sebesar {hasil[entities['target']]:,.0f}"
            advice = generate_text_response(models, advice_prompt, max_length=150)
            if advice and len(advice) > 20:  # Only use if we got a meaningful response
                return f"{basic_response}\n\n💡 Saran: {advice}"
        except:
            pass
        
        return basic_response

# ------------------- MAIN INTERACTIVE LOOP ------------------- #
def main():
    print("\n" + "=" * 50)
    print("🤖 BUSINESS ASSISTANT AI - INTEGRATED MODEL")
    print("=" * 50)
    
    # Load all models
    models = load_models()
    
    print("\n✨ Business Assistant siap membantu Anda!")
    print("📝 Contoh pertanyaan:")
    print("  - 'Berapa profit saya hari ini?'")
    print("  - 'Prediksi penjualan toko A untuk minggu depan'")
    print("  - 'Bagaimana cara meningkatkan keuntungan bisnis saya?'")
    print("  - 'Bandingkan performa toko A dan toko B'")
    print("\nKetik 'exit', 'quit', atau 'keluar' untuk mengakhiri.\n")
    
    while True:
        try:
            user_input = input("🧑‍💼 Anda: ").strip()
            if user_input.lower() in ["exit", "quit", "keluar"]:
                print("\n👋 Terima kasih telah menggunakan Business Assistant AI!")
                break
            
            # Process query and get response
            response = handle_query(models, user_input)
            print(f"\n🤖 Assistant: {response}\n")
            
        except Exception as e:
            print(f"\n⚠️ Terjadi kesalahan: {e}\n")

if __name__ == "__main__":
    main()