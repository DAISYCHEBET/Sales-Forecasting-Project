#  Sales Time Series Forecasting - Production ML Pipeline

> An end-to-end machine learning project for sales forecasting with MLOps best practices, API deployment, and interactive dashboards.

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-enabled-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

##  Project Overview

This project demonstrates a **production-ready machine learning pipeline** for time series forecasting. It includes:

-  Multiple forecasting models (Prophet, SARIMA, XGBoost, LSTM)
-  MLOps practices (MLflow tracking, experiment versioning)
-  REST API (FastAPI)
-  Interactive Dashboard (Streamlit)
-  Docker containerization
-  Comprehensive testing
-  CI/CD pipeline

##  Architecture
```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Raw Data  │─────▶│  Processing  │─────▶│   Models    │
└─────────────┘      └──────────────┘      └─────────────┘
                            │                      │
                            ▼                      ▼
                     ┌──────────────┐      ┌─────────────┐
                     │   Features   │      │  MLflow     │
                     └──────────────┘      └─────────────┘
                            │                      │
                            ▼                      ▼
                     ┌──────────────┐      ┌─────────────┐
                     │   FastAPI    │◀────▶│  Streamlit  │
                     └──────────────┘      └─────────────┘
```

## Project Structure
```
sales_timeseries_project/
├── data/                    # Data storage
│   ├── raw/                # Original datasets
│   ├── processed/          # Cleaned data
│   └── predictions/        # Forecast outputs
├── notebooks/              # Jupyter notebooks for exploration
├── src/                    # Source code
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── evaluation.py
│   └── utils.py
├── api/                    # FastAPI application
├── streamlit_app/          # Dashboard
├── tests/                  # Unit tests
├── config/                 # Configuration files
└── models/                 # Saved models
```

##  Quick Start

### Prerequisites
- Python 3.9+
- Docker (optional)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/DAISYCHEBET/sales_timeseries_project.git
cd sales_timeseries_project
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

#### Run the complete pipeline
```bash
python src/main.py
```

#### Start the API
```bash
uvicorn api.app:app --reload
```

#### Launch the dashboard
```bash
streamlit run streamlit_app/dashboard.py
```

#### Using Docker
```bash
docker-compose up
```

##  Models Implemented

| Model | Use Case | Accuracy |
|-------|----------|----------|
| **Prophet** | Quick baseline, handles seasonality | MAE: TBD |
| **SARIMA** | Statistical approach, interpretable | MAE: TBD |
| **XGBoost** | Feature-rich, high performance | MAE: TBD |
| **LSTM** | Deep learning, complex patterns | MAE: TBD |

##  MLOps Features

- **Experiment Tracking**: MLflow for logging metrics, parameters, and models
- **Model Registry**: Version control for trained models
- **CI/CD**: Automated testing and deployment
- **Monitoring**: Performance tracking and drift detection
- **Reproducibility**: Seeded random states and environment configs

##  API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Generate forecast |
| `/models` | GET | List available models |
| `/metrics` | GET | Model performance metrics |
| `/retrain` | POST | Trigger model retraining |

##  Dashboard Features

-  Interactive time series visualization
-  Real-time predictions
-  Model comparison
-  Data upload capability
-  Export predictions

##  Testing
```bash
# Run all tests
pytest

# With coverage
pytest --cov=src tests/
```

##  Results

*Coming soon - performance metrics and visualizations*

## Roadmap

- [ ] Phase 1: Project setup
- [ ] Phase 2: Data preprocessing module
- [ ] Phase 3: Model development
- [ ] Phase 4: API development
- [ ] Phase 5: Dashboard creation
- [ ] Phase 6: Docker deployment
- [ ] Phase 7: CI/CD pipeline
- [ ] Phase 8: Cloud deployment

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



## Author

**Daisy Chebet**
- GitHub: [@DAISYCHEBET](https://github.com/DAISYCHEBET)
- Email: chebetd003@gmail.com

## Acknowledgments

- Dataset: Superstore Sales Dataset
- Inspiration: Production ML best practices

---

**Star this repo if you find it helpful!**
