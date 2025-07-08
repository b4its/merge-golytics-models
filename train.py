import os
import json
import torch
import numpy as np
import pandas as pd
import joblib
import pickle
import warnings
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import shutil
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import traceback
warnings.filterwarnings("ignore")


# ================== SETUP DIRECTORIES ================== #
def setup_directories(base_dir="final"):
    """Setup all required directories with new base directory"""
    dirs = [
        f"{base_dir}/model", 
        f"{base_dir}/dataset", 
        f"{base_dir}/data/normData", 
        f"{base_dir}/output/plot", 
        f"{base_dir}/output/forecast", 
        f"{base_dir}/output/model_summary", 
        f"{base_dir}/models",
        f"{base_dir}/text_models"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✅ All directories created successfully")
    
    # Copy existing dataset files if they exist
    source_dirs = ["dataset", "data"]
    for source_dir in source_dirs:
        if os.path.exists(source_dir):
            for file_name in os.listdir(source_dir):
                source_path = os.path.join(source_dir, file_name)
                if os.path.isfile(source_path):
                    dest_path = os.path.join(f"{base_dir}/{source_dir}", file_name)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy2(source_path, dest_path)
    
    print("✅ Dataset files copied to new directory structure")

# ================== NEURAL NETWORK MODEL ================== #
class BisnisAssistantModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        return self.model(x)

# ================== TEXT GENERATION MODEL ================== #
class TextGenerationModel:
    def __init__(self, base_dir="final"):
        self.base_dir = base_dir
        self.model_dir = f"{base_dir}/text_models"
        self.dataset_dir = f"{base_dir}/dataset"
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(f"{self.model_dir}/gpt2_business_qa", exist_ok=True)
        
    def prepare_text_data(self):
        """Prepare text data for fine-tuning"""
        # Combine all text data sources
        text_data_paths = [
            f"{self.dataset_dir}/questions_augmentedv1.json",
            f"{self.dataset_dir}/questions_augmentedv2.json"
        ]
        
        all_texts = []
        for path in text_data_paths:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        all_texts.append(item["text"])
            else:
                print(f"⚠️ Text data file not found: {path}")
        
        if not all_texts:
            print("⚠️ No text data found for training")
            return None
            
        # Create training text file
        train_file = f"{self.dataset_dir}/text_train.txt"
        with open(train_file, "w", encoding="utf-8") as f:
            f.write("\n".join(all_texts))
            
        print(f"✅ Text training data prepared: {len(all_texts)} examples")
        return train_file
    
    def train_gpt2_model(self, train_file, epochs=3, batch_size=4):
        """Train a small GPT-2 model for text generation"""
        if train_file is None:
            print("⚠️ No training file provided for GPT-2 model")
            return None
            
        try:
            # Check for CUDA
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
            
            # Load tokenizer and model
            model_name = "gpt2"  # Using the smallest GPT-2 model
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            # Create a smaller GPT-2 config for faster training
            config = GPT2Config.from_pretrained(model_name)
            config.n_layer = 4  # Reduce number of layers
            config.n_head = 4   # Reduce number of attention heads
            config.n_embd = 256 # Reduce embedding dimension
            
            model = GPT2LMHeadModel(config=config)
            
            # Prepare dataset
            dataset = TextDataset(
                tokenizer=tokenizer,
                file_path=train_file,
                block_size=128
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"{self.model_dir}/gpt2_business_qa_checkpoints",
                overwrite_output_dir=True,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                save_steps=500,
                save_total_limit=2,
                logging_dir=f"{self.base_dir}/output/logs",
                logging_steps=100,
                fp16=torch.cuda.is_available(),  # Use mixed precision if available
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=tokenizer, mlm=False
                ),
                train_dataset=dataset,
            )
            
            # Train model
            print("🚀 Training GPT-2 text generation model...")
            trainer.train()
            
            # Save model and tokenizer to the specified directory
            model_save_path = f"{self.model_dir}/gpt2_business_qa"
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            
            # Verify the model was saved correctly
            if os.path.exists(f"{model_save_path}/pytorch_model.bin"):
                print(f"✅ Text generation model trained and saved to {model_save_path}")
            else:
                print(f"⚠️ Model files not found at {model_save_path}")
            
            return model
            
        except Exception as e:
            print(f"❌ Error in text generation model training: {e}")
            traceback.print_exc()  # Print full traceback for debugging
            return None

# ================== NVIDIA OPTIMIZATIONS ================== #
def setup_nvidia_optimizations():
    """Setup NVIDIA optimizations if available"""
    if torch.cuda.is_available():
        print(f"🔥 NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
        
        # Set CUDA device
        torch.cuda.set_device(0)
        
        # Enable cuDNN benchmarking for better performance
        torch.backends.cudnn.benchmark = True
        
        # For deterministic results (slightly slower)
        # torch.backends.cudnn.deterministic = True
        
        # Set number of threads for CPU operations
        cpu_count = os.cpu_count()
        torch.set_num_threads(cpu_count // 2)
        
        print(f"✅ NVIDIA optimizations enabled | CPU Threads: {torch.get_num_threads()}")
        return True
    else:
        print("⚠️ No NVIDIA GPU detected, running on CPU")
        return False

# ================== CLASSIFICATION FUNCTIONS ================== #
def load_processed_classification(filepath):
    """Load processed classification data"""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"⚠️ Classification file not found: {filepath}")
        return None

def label_and_onehot_processing(df):
    """Process classification data with label encoding and one-hot encoding"""
    label_features = ['Has Website', 'Social Media Presence', 'Marketplace Usage',
                      'Payment Digital Adoption', 'POS (Point of Sales) Usage',
                      'Online Ads Usage', 'E-Wallet Acceptance']

    onehot_features = ['Title', 'Active Social Media Channels', 'Social Media Posting Frequency',
                       'Year Started Digital Adoption', 'Business Size', 'Monthly Revenue',
                       'Number of Employees', 'Location Type']

    target = 'Willingness to Develop'

    X = df.drop(columns=[target])
    y = df[target]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    real_label = {target: dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))}

    for column in label_features:
        if column in X.columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            real_label[column] = dict(zip(le.transform(le.classes_), le.classes_))

    # Filter onehot_features to only include columns that exist in X
    existing_onehot_features = [col for col in onehot_features if col in X.columns]
    
    if existing_onehot_features:
        column_transformer = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(drop='first', sparse_output=False), existing_onehot_features)
            ],
            remainder='passthrough'
        )

        X_processed = column_transformer.fit_transform(X)
        onehot_feature_names = column_transformer.named_transformers_['onehot'].get_feature_names_out(existing_onehot_features)
        final_feature_names = list(onehot_feature_names) + [col for col in label_features if col in X.columns]
        X_processed_df = pd.DataFrame(X_processed, columns=final_feature_names)
    else:
        X_processed_df = X
        final_feature_names = list(X.columns)

    return X_processed_df, y, final_feature_names, real_label

def train_decision_tree(X_train, X_test, y_train, y_test, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """Train and evaluate decision tree"""
    clf = DecisionTreeClassifier(criterion='entropy',
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Accuracy: {accuracy * 100:.2f}% | max_depth={max_depth}")
    return clf, accuracy

def train_random_forest(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """Train and evaluate random forest"""
    rf = RandomForestClassifier(n_estimators=n_estimators,
                               max_depth=max_depth,
                               min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf,
                               random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy * 100:.2f}% | n_estimators={n_estimators}")
    return rf, accuracy

# ================== TIME SERIES FUNCTIONS ================== #
def load_time_series_data():
    """Load and prepare time series data"""
    try:
        store = pd.read_csv("final/dataset/store.csv")
        train = pd.read_csv("final/dataset/train.csv", parse_dates=True, low_memory=False, index_col='Date')
        
        train['Year'] = train.index.year
        train['Month'] = train.index.month
        train['Day'] = train.index.day
        train['WeekOfYear'] = train.index.isocalendar().week
        train['SalePerCustomer'] = train['Sales']/train['Customers']
        train.fillna(0, inplace=True)
        
        store.fillna(0, inplace=True)
        
        # Clean data
        train = train[(train["Open"] != 0) & (train['Sales'] != 0)]
        train_store = pd.merge(train, store, how='inner', on='Store')
        
        print(f"✅ Time series data loaded: {train_store.shape}")
        return train, train_store
    except FileNotFoundError as e:
        print(f"⚠️ Time series files not found: {e}")
        return None, None

def test_stationarity(timeseries, window=12, cutoff=0.01, series_name="Time Series", base_dir="final"):
    """Test stationarity of time series"""
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    fig = plt.figure(figsize=(12, 8))
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title(f'Rolling Mean & Standard Deviation - {series_name}')
    plt.savefig(f'{base_dir}/output/plot/rolling_stats_{series_name.replace(" ", "_")}.png')
    plt.close()

    dftest = adfuller(timeseries, autolag='AIC', maxlag=20)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    
    pvalue = dftest[1]
    if pvalue < cutoff:
        print(f'p-value = {pvalue:.4f}. The series is likely stationary.')
    else:
        print(f'p-value = {pvalue:.4f}. The series is likely non-stationary.')
    
    return dfoutput

def train_arima_sarima_models(train_data, base_dir="final"):
    """Train ARIMA and SARIMA models for different stores"""
    models = {}
    
    # Select specific stores for analysis
    stores = [2, 85, 1, 13]
    store_names = ['A', 'B', 'C', 'D']
    
    for store_id, store_name in zip(stores, store_names):
        try:
            print(f"\n🔍 Training models for Store {store_name} (ID: {store_id})")
            
            # Extract sales data for the store
            sales_data = train_data[train_data.Store == store_id]['Sales']
            
            if len(sales_data) < 50:  # Skip if insufficient data
                print(f"⚠️ Insufficient data for Store {store_name}")
                continue
            
            # Test stationarity
            print(f"Testing stationarity for Store {store_name}...")
            test_stationarity(sales_data, series_name=f"Store {store_name}", base_dir=base_dir)
            
            # Determine ARIMA parameters based on store
            if store_name == 'A':
                arima_order = (11, 1, 0)
                sarima_order = (11, 1, 0)
                sarima_seasonal = (2, 1, 0, 12)
            else:
                arima_order = (1, 1, 0)
                sarima_order = (1, 1, 0)
                sarima_seasonal = (2, 1, 0, 12)
            
            # Train ARIMA model
            print(f"Training ARIMA{arima_order} for Store {store_name}...")
            arima_model = ARIMA(sales_data, order=arima_order).fit()
            
            # Train SARIMA model with error handling
            print(f"Training SARIMA for Store {store_name}...")
            try:
                sarima_model = SARIMAX(sales_data, trend='n', order=sarima_order, seasonal_order=sarima_seasonal).fit()
            except MemoryError:
                print(f"Memory error for Store {store_name} SARIMA. Using simpler parameters...")
                try:
                    sarima_model = SARIMAX(sales_data, trend='n', order=sarima_order, seasonal_order=(1, 1, 0, 12)).fit()
                except MemoryError:
                    print(f"Still memory error. Using ARIMA instead for Store {store_name}...")
                    sarima_model = ARIMA(sales_data, order=arima_order).fit()
            
            # Save models
            models[f'arima_store_{store_name}'] = arima_model
            models[f'sarima_store_{store_name}'] = sarima_model
            
            # Save to files
            with open(f'{base_dir}/models/arima_store_{store_name}.pkl', 'wb') as f:
                pickle.dump(arima_model, f)
            with open(f'{base_dir}/models/sarima_store_{store_name}.pkl', 'wb') as f:
                pickle.dump(sarima_model, f)
            
            # Save model summaries
            with open(f'{base_dir}/output/model_summary/arima_store_{store_name}_summary.txt', 'w') as f:
                f.write(str(arima_model.summary()))
            with open(f'{base_dir}/output/model_summary/sarima_store_{store_name}_summary.txt', 'w') as f:
                f.write(str(sarima_model.summary()))
            
            print(f"✅ Models for Store {store_name} trained and saved successfully")
            
        except Exception as e:
            print(f"❌ Error training models for Store {store_name}: {e}")
            continue
    
    return models

# ================== NEURAL NETWORK TRAINING ================== #
def prepare_neural_network_data(base_dir="final"):
    """Prepare data for neural network training"""
    X, y = [], []
    
    # Check if normal data directory exists
    normal_dir = f"{base_dir}/dataset/normal"
    if not os.path.exists(normal_dir):
        print(f"⚠️ Neural network data directory not found: {normal_dir}")
        print("Creating sample data for demonstration...")
        
        # Create sample data if directory doesn't exist
        os.makedirs(normal_dir, exist_ok=True)
        sample_dir = os.path.join(normal_dir, "sample_label")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Create sample JSON data
        sample_data = [
            {
                "total_pemasukan": 100000,
                "total_pengeluaran": 80000,
                "waktu": "2024-01-01T10:00:00",
                "modal_awal": 50000,
                "rugi": 0
            },
            {
                "total_pemasukan": 150000,
                "total_pengeluaran": 120000,
                "waktu": "2024-01-01T14:00:00",
                "modal_awal": 60000,
                "rugi": 0
            }
        ]
        
        with open(os.path.join(sample_dir, "sample_data.json"), 'w') as f:
            json.dump(sample_data, f, indent=2)
    
    norm_dir = f"{base_dir}/data/normData"
    
    # Process data
    for label_folder in os.listdir(normal_dir):
        label_path = os.path.join(normal_dir, label_folder)
        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):
            if not file.endswith(".json"):
                continue
                
            try:
                with open(os.path.join(label_path, file)) as f:
                    data = json.load(f)

                out_file = os.path.join(norm_dir, f"dataset_{label_folder}_{file}")
                with open(out_file, "w") as out:
                    json.dump(data, out, indent=2)

                for item in data:
                    pemasukan = item["total_pemasukan"]
                    pengeluaran = item["total_pengeluaran"]
                    waktu = datetime.fromisoformat(item["waktu"])
                    jam = waktu.hour / 24.0

                    X.append([pemasukan, pengeluaran, jam])
                    modal = item["modal_awal"]
                    rugi = item["rugi"]
                    profit = pemasukan - pengeluaran if rugi == 0 else 0
                    y.append([modal, profit, rugi])
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

    if len(X) == 0:
        print("⚠️ No data found for neural network training")
        return None, None, None, None
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    # Normalization
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Save scalers
    joblib.dump(scaler_x, os.path.join(norm_dir, "scaler_x.pkl"))
    joblib.dump(scaler_y, os.path.join(norm_dir, "scaler_y.pkl"))
    
    return X_scaled, y_scaled, scaler_x, scaler_y

def train_neural_network(X_scaled, y_scaled, base_dir="final", use_gpu=False):
    """Train neural network model with MLflow tracking"""
    if X_scaled is None or y_scaled is None:
        print("⚠️ Skipping neural network training due to insufficient data")
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    model = BisnisAssistantModel(input_size=X_train.shape[1], hidden_size=64, output_size=y_train.shape[1]).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    patience = 20
    min_delta = 1e-4
    best_loss = float("inf")
    wait = 0
    
    # MLflow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment_name = "integrated-business-model-enhanced"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="neural_network_training_enhanced"):
        pbar = tqdm(range(1000), desc="Training Neural Network")
        
        for epoch in pbar:
            model.train()
            epoch_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                output = model(batch_x)
                loss = loss_fn(output, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")
            
            if epoch_loss + min_delta < best_loss:
                best_loss = epoch_loss
                wait = 0
                torch.save(model.state_dict(), f"{base_dir}/model/integrated_neural_network.pth")
            else:
                wait += 1
                if wait >= patience:
                    pbar.close()
                    print(f"\n⏹ Early stopping triggered at epoch {epoch}")
                    break
        
        # MLflow logging
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("patience", patience)
        mlflow.log_param("device", str(device))
        mlflow.log_metric("final_train_loss", best_loss)
        
        # Save model to MLflow
        example_input = np.random.randn(1, X_train.shape[1]).astype(np.float32)
        mlflow.pytorch.log_model(
            model,
            artifact_path="integrated_neural_network_enhanced",
            input_example=example_input
        )
        
        print("✅ Neural network training completed and logged to MLflow")
        
    return model

# ================== TEXT PROCESSING ================== #
def process_text_data(base_dir="final"):
    """Process text data for paraphrasing"""
    INTENT_TEMPLATES = {
        "tanya_profit": {
            "templates": [
                "Berapa keuntungan saya hari ini?",
                "Berapakah profit saya hari ini?",
                "Apa laba saya per hari ini?",
                "Keuntungan saya hari ini berapa?",
                "Apakah saya mendapatkan keuntungan hari ini?"
            ],
            "entities": {
                "waktu": "hari_ini",
                "target": "keuntungan"
            }
        },
        "tanya_rugi": {
            "templates": [
                "Apakah saya mengalami kerugian minggu ini?",
                "Saya rugi minggu ini?",
                "Rugi saya minggu ini berapa?",
                "Apakah saya mengalami rugi sepekan ini?",
                "Apakah saya mengalami defisit sepekan ini?"
            ],
            "entities": {
                "waktu": "minggu_ini",
                "target": "kerugian"
            }
        }
    }
    
    text_data_paths = [
        f"final/dataset/questions_augmentedv1.json",
        f"final/dataset/questions_augmentedv2.json"
    ]
    
    text_data = []
    for path in text_data_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                text_data.extend(json.load(f))
        else:
            print(f"⚠️ Text data file not found: {path}")
    
    if not text_data:
        print("⚠️ No text data found for processing")
        # Create sample data
        print("Creating sample text data...")
        sample_data = []
        for intent, info in INTENT_TEMPLATES.items():
            for template in info["templates"]:
                sample_data.append({
                    "text": template,
                    "intent": intent,
                    "entities": info["entities"]
                })
        text_data = sample_data
        
        # Save sample data
        os.makedirs(f"{base_dir}/dataset", exist_ok=True)
        with open(f"final/dataset/questions_augmentedv1.json", "w", encoding="utf-8") as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    try:
        model_sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        augmented_data = []
        
        for entry in text_data:
            question = entry["text"]
            intent = entry["intent"]
            
            if intent not in INTENT_TEMPLATES:
                continue
            
            templates = INTENT_TEMPLATES[intent]["templates"]
            base_entities = INTENT_TEMPLATES[intent]["entities"]
            
            try:
                question_embedding = model_sbert.encode(question, convert_to_tensor=True)
                template_embeddings = model_sbert.encode(templates, convert_to_tensor=True)
                cos_scores = util.cos_sim(question_embedding, template_embeddings)[0]
                top_indices = cos_scores.argsort(descending=True)[:3]
                
                for idx in top_indices:
                    new_question = templates[idx]
                    augmented_data.append({
                        "text": new_question,
                        "intent": intent,
                        "entities": base_entities
                    })
            except Exception as e:
                print(f"Paraphrase error: {e}")
        
        # Add entities to original data
        for entry in text_data:
            if "entities" not in entry:
                entry["entities"] = INTENT_TEMPLATES.get(entry["intent"], {}).get("entities", {})
        
        text_data.extend(augmented_data)
        
        # Save augmented data
        with open(f"{base_dir}/data/questions_augmented_integrated.json", "w", encoding="utf-8") as f:
            json.dump(text_data, f, indent=2, ensure_ascii=False)
        
        # Create text file for GPT-2 training
        with open(f"{base_dir}/dataset/text_train.txt", "w", encoding="utf-8") as f:
            f.write("\n".join([entry["text"] for entry in text_data]))
        
        print("✅ Text data processing completed")
        
    except Exception as e:
        print(f"❌ Error in text processing: {e}")

# ================== MAIN TRAINING FUNCTION ================== #
def main():
    """Main function to run integrated training"""
    base_dir = "final"
    print("🚀 Starting Enhanced Integrated Model Training...")
    print("=" * 60)
    
    # Setup directories with new base directory
    setup_directories(base_dir)
    
    # Setup NVIDIA optimizations
    has_gpu = setup_nvidia_optimizations()
    
    # 1. Classification Model Training
    print("\n🔍 PHASE 1: Classification Model Training")
    print("-" * 40)
    
    classification_files = [
        f"{base_dir}/dataset/processed_data.csv",
        f"{base_dir}/dataset/business_data.csv",
        f"{base_dir}/dataset/classification_data.csv"
    ]
    
    classification_df = None
    for file_path in classification_files:
        classification_df = load_processed_classification(file_path)
        if classification_df is not None:
            break
    
    if classification_df is not None:
        try:
            X_processed, y, feature_names, real_labels = label_and_onehot_processing(classification_df)
            X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)
            
            # Train Decision Tree
            dt_model, dt_accuracy = train_decision_tree(X_train, X_test, y_train, y_test, max_depth=10)
            
            # Train Random Forest
            rf_model, rf_accuracy = train_random_forest(X_train, X_test, y_train, y_test, 
                                                      n_estimators=100, max_depth=10)
            
            # Save classification models
            joblib.dump(dt_model, f'{base_dir}/models/decision_tree_model.pkl')
            joblib.dump(rf_model, f'{base_dir}/models/random_forest_model.pkl')
            joblib.dump(real_labels, f'{base_dir}/models/classification_labels.pkl')
            
            print("✅ Classification models saved successfully")
            
        except Exception as e:
            print(f"❌ Error in classification training: {e}")
    else:
        print("⚠️ No classification data found, skipping classification training")
    
    # 2. Time Series Model Training
    print("\n📈 PHASE 2: Time Series Model Training")
    print("-" * 40)
    
    train_data, train_store_data = load_time_series_data()
    if train_data is not None:
        try:
            arima_sarima_models = train_arima_sarima_models(train_data, base_dir)
            print("✅ Time series models trained and saved successfully")
        except Exception as e:
            print(f"❌ Error in time series training: {e}")
    else:
        print("⚠️ No time series data found, skipping time series training")
    
    # 3. Neural Network Training
    print("\n🧠 PHASE 3: Neural Network Training")
    print("-" * 40)
    
    try:
        X_scaled, y_scaled, scaler_x, scaler_y = prepare_neural_network_data(base_dir)
        neural_model = train_neural_network(X_scaled, y_scaled, base_dir, has_gpu)
        
        if neural_model is not None:
            print("✅ Neural network trained and saved successfully")
        else:
            print("⚠️ Neural network training skipped due to insufficient data")
    except Exception as e:
        print(f"❌ Error in neural network training: {e}")
    
    # 4. Text Processing and Generation Model
    print("\n📝 PHASE 4: Text Data Processing and Generation Model")
    print("-" * 40)
    
    try:
        process_text_data(base_dir)
        
        # Train text generation model
        text_gen = TextGenerationModel(base_dir)
        train_file = text_gen.prepare_text_data()
        
        if train_file:
            text_model = text_gen.train_gpt2_model(train_file)
            
            # Verify model was saved
            model_path = f"{base_dir}/text_models/gpt2_business_qa"
            if os.path.exists(f"{model_path}/pytorch_model.bin"):
                print(f"✅ Text generation model saved to {model_path}")
            else:
                print(f"⚠️ Text generation model not saved properly")
                
                # Fallback: try to save a minimal model
                try:
                    print("Attempting to save a minimal model as fallback...")
                    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                    model = GPT2LMHeadModel.from_pretrained("gpt2")
                    
                    # Create directory if it doesn't exist
                    os.makedirs(model_path, exist_ok=True)
                    
                    # Save model and tokenizer
                    model.save_pretrained(model_path)
                    tokenizer.save_pretrained(model_path)
                    
                    if os.path.exists(f"{model_path}/pytorch_model.bin"):
                        print(f"✅ Fallback text generation model saved to {model_path}")
                    else:
                        print(f"❌ Failed to save fallback model")
                except Exception as e:
                    print(f"❌ Error saving fallback model: {e}")
        else:
            print("⚠️ No training data available for text generation model")
        
        print("✅ Text data processing and model training completed")
    except Exception as e:
        print(f"❌ Error in text processing and generation: {e}")
        traceback.print_exc()  # Print full traceback for debugging
        
        # Fallback: try to save a minimal model
        try:
            print("Attempting to save a minimal model as fallback...")
            model_path = f"{base_dir}/text_models/gpt2_business_qa"
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            
            # Create directory if it doesn't exist
            os.makedirs(model_path, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            if os.path.exists(f"{model_path}/pytorch_model.bin"):
                print(f"✅ Fallback text generation model saved to {model_path}")
            else:
                print(f"❌ Failed to save fallback model")
        except Exception as e:
            print(f"❌ Error saving fallback model: {e}")
    
    # Summary
    print("\nTRAINING COMPLETE!")
    print("=" * 60)
    print(f"All models saved in '{base_dir}/' directory structure:")
    print(f"  - Decision Tree: {base_dir}/models/decision_tree_model.pkl")
    print(f"  - Random Forest: {base_dir}/models/random_forest_model.pkl")
    print(f"  - ARIMA/SARIMA: {base_dir}/models/arima_store_*.pkl, {base_dir}/models/sarima_store_*.pkl")
    print(f"  - Neural Network: {base_dir}/model/integrated_neural_network.pth")
    print(f"  - Text Generation: {base_dir}/text_models/gpt2_business_qa")
    print(f"  - Scalers: {base_dir}/data/normData/scaler_*.pkl")
    print(f"Plots saved in '{base_dir}/output/plot/' directory")
    print(f"Forecasts saved in '{base_dir}/output/forecast/' directory")
    print(f"Model summaries saved in '{base_dir}/output/model_summary/' directory")
    print("MLflow tracking available at: http://127.0.0.1:5000")

if __name__ == "__main__":
    main()
