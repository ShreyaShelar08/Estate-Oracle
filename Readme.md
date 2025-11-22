# 🏠 Multi-Agent Property Price Predictor

An intelligent property price prediction system powered by 4 specialized AI agents and Google Gemini API.

## 🌟 Features

- 🤖 **4 Specialized AI Agents**: Structural Analysis, Location Analysis, Market Trends, Geographic Pricing
- 🗺️ **Location Intelligence**: Predict prices across 10 major US cities
- 💬 **Natural Language Queries**: Ask questions in plain English using Gemini AI
- 📊 **Market Analysis Dashboard**: Comprehensive market insights and comparisons
- 📈 **Interactive Visualizations**: Beautiful charts and graphs using Plotly

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- Gemini API Key (free from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/property-price-predictor.git
cd property-price-predictor
```

2. Create a virtual environment:
```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

5. Run the application:
```bash
streamlit run app.py
```

## 📖 Usage

### Chat Prediction
Ask natural language questions:
- "How much would a 3 bedroom house cost in Austin?"
- "What's the price for a 2000 sqft property in San Diego?"

### Location Search
- Select a city and property type
- Compare prices across different cities
- View market insights

### Manual Input
- Enter specific property details
- Get detailed price breakdown
- See individual agent predictions

### Market Analysis
- View comprehensive market statistics
- Compare cities
- Export data as CSV

## 🏙️ Supported Cities

New York, Los Angeles, Chicago, Houston, Phoenix, Philadelphia, San Antonio, San Diego, Dallas, Austin

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML Models**: Scikit-learn (Random Forest, Gradient Boosting, Linear Regression)
- **AI**: Google Gemini API
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy

## 📊 System Architecture
```
┌─────────────────────────────────────────┐
│      Orchestrator Agent (Ensemble)       │
└─────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┬──────────┐
        │           │           │          │
┌───────▼─────┐ ┌──▼────┐ ┌────▼────┐ ┌───▼──────┐
│ Structural  │ │Location│ │ Market  │ │Geographic│
│  Analysis   │ │Analysis│ │ Trends  │ │ Pricing  │
│   Agent     │ │ Agent  │ │  Agent  │ │  Agent   │
└─────────────┘ └────────┘ └─────────┘ └──────────┘
```



