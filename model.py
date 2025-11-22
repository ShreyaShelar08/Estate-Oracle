import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Dict, List, Tuple
import warnings
import streamlit as st
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()
warnings.filterwarnings('ignore')

# ==================== DATA GENERATION ====================
def generate_sample_data(n_samples=1000):
    """Generate synthetic property data with location information"""
    np.random.seed(42)
    
    # Define cities with their characteristics
    cities = {
        'New York': {'base_price_mult': 2.0, 'location_score': 9.0, 'crime_rate': 4.5},
        'Los Angeles': {'base_price_mult': 1.8, 'location_score': 8.5, 'crime_rate': 5.0},
        'Chicago': {'base_price_mult': 1.3, 'location_score': 7.5, 'crime_rate': 6.0},
        'Houston': {'base_price_mult': 1.0, 'location_score': 7.0, 'crime_rate': 5.5},
        'Phoenix': {'base_price_mult': 0.9, 'location_score': 6.5, 'crime_rate': 4.0},
        'Philadelphia': {'base_price_mult': 1.2, 'location_score': 7.5, 'crime_rate': 5.5},
        'San Antonio': {'base_price_mult': 0.8, 'location_score': 6.0, 'crime_rate': 4.5},
        'San Diego': {'base_price_mult': 1.7, 'location_score': 8.5, 'crime_rate': 3.5},
        'Dallas': {'base_price_mult': 1.1, 'location_score': 7.0, 'crime_rate': 5.0},
        'Austin': {'base_price_mult': 1.4, 'location_score': 8.0, 'crime_rate': 3.0},
    }
    
    city_names = list(cities.keys())
    selected_cities = np.random.choice(city_names, n_samples)
    
    data = {
        'city': selected_cities,
        'sqft': np.random.randint(500, 5000, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 5, n_samples),
        'age': np.random.randint(0, 50, n_samples),
        'amenities_score': np.random.randint(1, 11, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Add city-specific attributes
    df['location_score'] = df['city'].map(lambda x: cities[x]['location_score'] + np.random.uniform(-0.5, 0.5))
    df['crime_rate'] = df['city'].map(lambda x: cities[x]['crime_rate'] + np.random.uniform(-0.5, 0.5))
    df['base_price_multiplier'] = df['city'].map(lambda x: cities[x]['base_price_mult'])
    
    # Generate price with city-specific multiplier
    df['price'] = (
        (df['sqft'] * 150 +
        df['bedrooms'] * 20000 +
        df['bathrooms'] * 15000 -
        df['age'] * 1000 +
        df['location_score'] * 30000 +
        df['amenities_score'] * 5000 -
        df['crime_rate'] * 8000) * df['base_price_multiplier'] +
        np.random.normal(0, 30000, n_samples)
    )
    
    return df, cities

# ==================== BASE AGENT CLASS ====================
class BaseAgent:
    """Base class for all agents"""
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        
    def preprocess(self, X):
        """Preprocess features"""
        return self.scaler.fit_transform(X)
    
    def train(self, X, y):
        """Train the agent's model"""
        raise NotImplementedError
    
    def predict(self, X):
        """Make predictions"""
        raise NotImplementedError
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        predictions = self.predict(X)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        return {'mae': mae, 'r2': r2, 'predictions': predictions}

# ==================== SPECIALIZED AGENTS ====================
class StructuralAnalysisAgent(BaseAgent):
    """Agent specialized in analyzing structural features"""
    def __init__(self):
        super().__init__("Structural Analysis Agent")
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.feature_cols = ['sqft', 'bedrooms', 'bathrooms', 'age']
    
    def train(self, X, y):
        X_features = X[self.feature_cols]
        X_scaled = self.scaler.fit_transform(X_features)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        X_features = X[self.feature_cols]
        X_scaled = self.scaler.transform(X_features)
        return self.model.predict(X_scaled)

class LocationAnalysisAgent(BaseAgent):
    """Agent specialized in analyzing location-based features"""
    def __init__(self):
        super().__init__("Location Analysis Agent")
        self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.feature_cols = ['location_score', 'crime_rate', 'amenities_score', 'base_price_multiplier']
    
    def train(self, X, y):
        X_features = X[self.feature_cols]
        X_scaled = self.scaler.fit_transform(X_features)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        X_features = X[self.feature_cols]
        X_scaled = self.scaler.transform(X_features)
        return self.model.predict(X_scaled)

class MarketTrendsAgent(BaseAgent):
    """Agent specialized in analyzing overall market trends"""
    def __init__(self):
        super().__init__("Market Trends Agent")
        self.model = LinearRegression()
        self.feature_cols = ['sqft', 'bedrooms', 'location_score', 'age', 'base_price_multiplier']
    
    def train(self, X, y):
        X_features = X[self.feature_cols]
        X_scaled = self.scaler.fit_transform(X_features)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        X_features = X[self.feature_cols]
        X_scaled = self.scaler.transform(X_features)
        return self.model.predict(X_scaled)

class GeographicPricingAgent(BaseAgent):
    """NEW: Agent specialized in city-based pricing patterns"""
    def __init__(self):
        super().__init__("Geographic Pricing Agent")
        self.model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, random_state=42)
        self.feature_cols = ['base_price_multiplier', 'location_score', 'crime_rate', 'sqft']
    
    def train(self, X, y):
        X_features = X[self.feature_cols]
        X_scaled = self.scaler.fit_transform(X_features)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        X_features = X[self.feature_cols]
        X_scaled = self.scaler.transform(X_features)
        return self.model.predict(X_scaled)

# ==================== ORCHESTRATION LAYER ====================
class OrchestratorAgent:
    """Orchestrator that coordinates all agents"""
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
        self.weights = None
        
    def train_agents(self, X_train, y_train):
        """Train all individual agents"""
        for agent in self.agents:
            agent.train(X_train, y_train)
        
        self._optimize_weights(X_train, y_train)
    
    def _optimize_weights(self, X_val, y_val):
        """Optimize ensemble weights based on validation performance"""
        performances = []
        
        for agent in self.agents:
            predictions = agent.predict(X_val)
            mae = mean_absolute_error(y_val, predictions)
            weight = 1 / (mae + 1)
            performances.append(weight)
        
        total = sum(performances)
        self.weights = [w / total for w in performances]
    
    def predict(self, X):
        """Make ensemble prediction using weighted average"""
        if self.weights is None:
            raise ValueError("Orchestrator must be trained before making predictions")
        
        predictions = []
        for agent in self.agents:
            pred = agent.predict(X)
            predictions.append(pred)
        
        ensemble_pred = np.zeros(len(X))
        for i, pred in enumerate(predictions):
            ensemble_pred += self.weights[i] * pred
        
        return ensemble_pred
    
    def detailed_prediction(self, X):
        """Get detailed predictions from all agents"""
        results = {
            'agents': {},
            'ensemble': None
        }
        
        for agent in self.agents:
            predictions = agent.predict(X)
            results['agents'][agent.name] = predictions
        
        results['ensemble'] = self.predict(X)
        return results

# ==================== GEMINI AI ASSISTANT ====================
class GeminiAssistant:
    """AI Assistant using Gemini API for natural language interaction"""
    def __init__(self, api_key: str, cities_data: Dict):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.cities_data = cities_data
    
    def get_location_info(self, city_name: str) -> Dict:
        """Get location information for a city"""
        # Normalize city name
        city_name_normalized = city_name.strip().title()
        
        if city_name_normalized in self.cities_data:
            city_info = self.cities_data[city_name_normalized]
            return {
                'city': city_name_normalized,
                'location_score': city_info['location_score'],
                'crime_rate': city_info['crime_rate'],
                'base_price_multiplier': city_info['base_price_mult']
            }
        
        # Use Gemini to find similar city or provide defaults
        prompt = f"""
        The user mentioned the city: {city_name}
        
        From this list of available cities: {', '.join(self.cities_data.keys())}
        
        Find the most similar city or return 'Houston' as default if no match.
        Return ONLY the city name, nothing else.
        """
        
        try:
            response = self.model.generate_content(prompt)
            suggested_city = response.text.strip()
            
            if suggested_city in self.cities_data:
                city_info = self.cities_data[suggested_city]
                return {
                    'city': suggested_city,
                    'location_score': city_info['location_score'],
                    'crime_rate': city_info['crime_rate'],
                    'base_price_multiplier': city_info['base_price_mult'],
                    'note': f"Using {suggested_city} as similar city"
                }
        except:
            pass
        
        # Default to Houston
        city_info = self.cities_data['Houston']
        return {
            'city': 'Houston',
            'location_score': city_info['location_score'],
            'crime_rate': city_info['crime_rate'],
            'base_price_multiplier': city_info['base_price_mult'],
            'note': "Using Houston as default city"
        }
        
    def extract_property_features(self, user_query: str) -> Dict:
        """Extract property features from natural language query"""
        cities_list = ', '.join(self.cities_data.keys())
        
        prompt = f"""
        Extract property features from the following user query. Return ONLY a valid JSON object with these fields:
        - city (extract city name if mentioned, otherwise use "Houston")
        - sqft (square footage, 500-5000)
        - bedrooms (1-5)
        - bathrooms (1-4)
        - age (0-50 years)
        - amenities_score (1-10, based on amenities mentioned)
        
        Available cities: {cities_list}
        
        If a value is not mentioned, use reasonable defaults based on context.
        
        User query: {user_query}
        
        Return ONLY the JSON object, no additional text.
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            features = json.loads(response_text)
            
            # Get location-specific data
            city_data = self.get_location_info(features.get('city', 'Houston'))
            features.update(city_data)
            
            return features
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            return None
    
    def explain_prediction(self, features: Dict, prediction: float, agent_predictions: Dict) -> str:
        """Generate natural language explanation of the prediction"""
        prompt = f"""
        Explain the following property price prediction in a friendly, conversational way:
        
        Property Features:
        - Location: {features.get('city', 'Unknown')}
        - Square Footage: {features.get('sqft', 0)} sqft
        - Bedrooms: {features.get('bedrooms', 0)}
        - Bathrooms: {features.get('bathrooms', 0)}
        - Age: {features.get('age', 0)} years
        - Location Score: {features.get('location_score', 0):.1f}/10
        - Crime Rate: {features.get('crime_rate', 0):.1f}/10
        
        Final Predicted Price: ${prediction:,.2f}
        
        Individual Agent Predictions:
        {json.dumps({k: f"${v[0]:,.2f}" for k, v in agent_predictions.items()}, indent=2)}
        
        Provide a clear, helpful explanation of:
        1. Why this price makes sense given the location and features
        2. Which factors most influenced the price
        3. Brief comparison with typical prices in this city
        
        Keep it concise (4-5 sentences) and helpful.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Prediction complete. The estimated price is ${prediction:,.2f} for a property in {features.get('city', 'this location')}."
    
    def compare_cities(self, property_features: Dict, cities_to_compare: List[str]) -> str:
        """Compare property prices across different cities"""
        prompt = f"""
        Given these property features:
        {json.dumps(property_features, indent=2)}
        
        Provide insights on how property prices might differ across these cities: {', '.join(cities_to_compare)}
        
        Consider factors like:
        - Market conditions
        - Cost of living
        - Location desirability
        - Typical price ranges
        
        Keep it concise (3-4 sentences) and informative.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return "City comparison analysis is currently unavailable."