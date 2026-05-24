# Demand Forecasting DevOps

Projekt przedstawia system prognozowania tygodniowego popytu rozszerzony o elementy DevOps.  
Początkowo model uczenia maszynowego był obsługiwany przez aplikację webową w Streamlit, a następnie projekt został rozbudowany o REST API, testy automatyczne, konteneryzację w Dockerze oraz pipeline CI w GitHub Actions.

---

## Cel projektu

Celem projektu było pokazanie, jak aplikację AI/ML można przygotować do bardziej produkcyjnego uruchamiania.

Projekt nie kończy się wyłącznie na lokalnej aplikacji Streamlit, ale posiada dodatkową warstwę backendową w postaci REST API. Dzięki temu model predykcyjny może być wywoływany nie tylko przez interfejs użytkownika, ale również przez inne aplikacje, systemy lub pipeline wdrożeniowy.

---

## Zakres wykonanych prac

W ramach rozszerzenia DevOps wykonano:

- dodanie REST API we FastAPI,
- wydzielenie logiki predykcji do osobnego modułu,
- dodanie endpointów `/health`, `/model-info` oraz `/predict`,
- przygotowanie testów automatycznych z użyciem pytest,
- przygotowanie Dockerfile do konteneryzacji aplikacji,
- dodanie docker-compose.yml do prostszego uruchamiania kontenera,
- przygotowanie workflow CI w GitHub Actions,
- wrzucenie projektu do repozytorium GitHub,
- skonfigurowanie `.gitignore` oraz `.dockerignore`.

---

## Technologie

W projekcie wykorzystano:

- Python
- Streamlit
- FastAPI
- Uvicorn
- pandas
- numpy
- scikit-learn
- joblib
- pytest
- Docker
- Docker Compose
- Git
- GitHub
- GitHub Actions

---

## Struktura projektu

```text
demand-forecasting-devops/
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── predictor.py
│
├── tests/
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app_streamlit.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
│
├── encoder_week.pkl
├── scaler_week.pkl
├── feature_cols_week.pkl
└── prediction_log.csv
