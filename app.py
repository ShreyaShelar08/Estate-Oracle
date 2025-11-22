import streamlit as st
from model import generate_sample_data
import pandas as pd 
import os
import plotly.express as px
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from datetime import datetime, timedelta
from model import LocationAnalysisAgent
from model import StructuralAnalysisAgent
from model import MarketTrendsAgent
from model import GeographicPricingAgent
from model import OrchestratorAgent
from model import GeminiAssistant
# ==================== STREAMLIT UI ====================
def initialize_system():
    """Initialize the multi-agent system"""
    if 'orchestrator' not in st.session_state:
        with st.spinner("Training multi-agent system with location data..."):
            # Generate data with cities
            df, cities = generate_sample_data(n_samples=2000)
            
            # Prepare features
            X = df.drop(['price', 'city'], axis=1)
            y = df['price']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize agents (now with 4 agents)
            agents = [
                StructuralAnalysisAgent(),
                LocationAnalysisAgent(),
                MarketTrendsAgent(),
                GeographicPricingAgent()
            ]
            
            # Create and train orchestrator
            orchestrator = OrchestratorAgent(agents)
            orchestrator.train_agents(X_train, y_train)
            
            st.session_state.orchestrator = orchestrator
            st.session_state.cities = cities
            st.session_state.df = df
            st.session_state.trained = True

def main():
    st.set_page_config(page_title="AI Property Price Predictor", page_icon="🏠", layout="wide")
    
    # Header
    st.title("🏠 Multi-Agent Property Price Predictor")
    st.markdown("### Powered by 4 AI Agents & Gemini")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # Get API key from environment
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if gemini_api_key:
            st.success("✅ API Key Loaded from Environment")
        else:
            st.error("❌ GEMINI_API_KEY not found in .env file")
            st.info("Please create a .env file with: GEMINI_API_KEY=your_key_here")
        
        st.markdown("---")
        st.header("📊 System Info")
        st.info("""
        **Active Agents (4):**
        - 🏗️ Structural Analysis Agent
        - 📍 Location Analysis Agent
        - 📈 Market Trends Agent
        - 🌍 Geographic Pricing Agent
        
        **Orchestrator:** Ensemble Predictor
        """)
        
        if st.button("🔄 Retrain System"):
            if 'orchestrator' in st.session_state:
                del st.session_state.orchestrator
            st.rerun()
    
    # Initialize system
    initialize_system()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Prediction", "🗺️ Location Search", "📝 Manual Input", "📊 Market Analysis"])
    
    # Tab 1: Chat-based prediction
    with tab1:
        st.header("Ask me about property prices!")
        st.markdown("*Example: 'How much would a 2000 sqft house with 3 bedrooms cost in Austin?'*")
        
        if not gemini_api_key:
            st.warning("⚠️ Please enter your Gemini API key in the sidebar to use chat-based prediction.")
        else:
            user_query = st.text_area(
                "Describe your property:", 
                height=100, 
                placeholder="E.g., 'I'm looking at a 3 bedroom, 2 bathroom house in San Diego that's 1800 square feet...'"
            )
            
            if st.button("🔮 Get Price Prediction", type="primary", key="chat_predict"):
                if user_query:
                    with st.spinner("Analyzing your property..."):
                        # Initialize Gemini assistant
                        assistant = GeminiAssistant(gemini_api_key, st.session_state.cities)
                        
                        # Extract features
                        features = assistant.extract_property_features(user_query)
                        
                        if features:
                            # Create DataFrame for prediction
                            df_pred = pd.DataFrame([features])
                            
                            # Get prediction
                            orchestrator = st.session_state.orchestrator
                            detailed = orchestrator.detailed_prediction(df_pred)
                            
                            # Display results
                            st.success("### 🎯 Prediction Results")
                            
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.metric("💰 Estimated Price", f"${detailed['ensemble'][0]:,.2f}")
                                st.metric("📍 Location", features.get('city', 'Unknown'))
                                
                                if 'note' in features:
                                    st.info(features['note'])
                                
                                st.markdown("#### 📋 Property Details:")
                                st.text(f"Square Feet: {features['sqft']}")
                                st.text(f"Bedrooms: {features['bedrooms']}")
                                st.text(f"Bathrooms: {features['bathrooms']}")
                                st.text(f"Age: {features['age']} years")
                                st.text(f"Location Score: {features['location_score']:.1f}/10")
                                st.text(f"Crime Rate: {features['crime_rate']:.1f}/10")
                            
                            with col2:
                                st.markdown("#### 🤖 Agent Predictions:")
                                for agent_name, pred in detailed['agents'].items():
                                    st.text(f"{agent_name}:")
                                    st.text(f"  ${pred[0]:,.2f}")
                                
                                # Visualization
                                agent_names = [name.split()[0] for name in detailed['agents'].keys()]
                                agent_values = [pred[0] for pred in detailed['agents'].values()]
                                
                                fig = go.Figure(data=[
                                    go.Bar(x=agent_names, y=agent_values, 
                                          marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
                                ])
                                fig.update_layout(
                                    title="Agent Predictions",
                                    xaxis_title="Agent",
                                    yaxis_title="Price ($)",
                                    height=300
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # AI explanation
                            st.markdown("---")
                            st.markdown("#### 💡 AI Explanation:")
                            explanation = assistant.explain_prediction(features, detailed['ensemble'][0], detailed['agents'])
                            st.info(explanation)
                else:
                    st.warning("Please describe your property first!")
    
    # Tab 2: NEW - Location-based search
    with tab2:
        st.header("🗺️ Search by Location")
        st.markdown("*Compare property prices across different cities*")
        
        if not gemini_api_key:
            st.warning("⚠️ Please configure your API key to use this feature.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                city_name = st.selectbox(
                    "Select City",
                    options=list(st.session_state.cities.keys()),
                    index=0
                )
            
            with col2:
                property_type = st.selectbox(
                    "Property Type",
                    ["Small", "Medium", "Large", "Luxury"]
                )
            
            # Predefined property templates
            property_templates = {
                "Small": {"sqft": 1000, "bedrooms": 2, "bathrooms": 1, "age": 15, "amenities_score": 5},
                "Medium": {"sqft": 2000, "bedrooms": 3, "bathrooms": 2, "age": 10, "amenities_score": 7},
                "Large": {"sqft": 3500, "bedrooms": 4, "bathrooms": 3, "age": 5, "amenities_score": 8},
                "Luxury": {"sqft": 5000, "bedrooms": 5, "bathrooms": 4, "age": 2, "amenities_score": 10}
            }
            
            if st.button("🔍 Search Properties", type="primary", key="location_search"):
                with st.spinner(f"Searching properties in {city_name}..."):
                    assistant = GeminiAssistant(gemini_api_key, st.session_state.cities)
                    
                    # Get city data
                    city_info = st.session_state.cities[city_name]
                    template = property_templates[property_type]
                    
                    # Create feature dict
                    features = {
                        **template,
                        'city': city_name,
                        'location_score': city_info['location_score'],
                        'crime_rate': city_info['crime_rate'],
                        'base_price_multiplier': city_info['base_price_mult']
                    }
                    
                    # Get prediction
                    df_pred = pd.DataFrame([features])
                    orchestrator = st.session_state.orchestrator
                    detailed = orchestrator.detailed_prediction(df_pred)
                    
                    # Display results
                    st.success(f"### 🏘️ {property_type} Property in {city_name}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("💰 Estimated Price", f"${detailed['ensemble'][0]:,.2f}")
                    with col2:
                        st.metric("📊 Location Score", f"{city_info['location_score']:.1f}/10")
                    with col3:
                        st.metric("🔒 Safety Score", f"{10-city_info['crime_rate']:.1f}/10")
                    
                    # Compare with other cities
                    st.markdown("---")
                    st.markdown("#### 🌍 Price Comparison Across Cities")
                    
                    comparison_data = []
                    for city, info in st.session_state.cities.items():
                        city_features = {
                            **template,
                            'location_score': info['location_score'],
                            'crime_rate': info['crime_rate'],
                            'base_price_multiplier': info['base_price_mult']
                        }
                        df_city = pd.DataFrame([city_features])
                        price = orchestrator.predict(df_city)[0]
                        comparison_data.append({
                            'City': city,
                            'Price': price,
                            'Location Score': info['location_score'],
                            'Safety': 10 - info['crime_rate']
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data).sort_values('Price', ascending=False)
                    
                    # Visualization
                    fig = px.bar(
                        comparison_df, 
                        x='City', 
                        y='Price',
                        title=f'{property_type} Property Prices by City',
                        color='Price',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Data table
                    st.dataframe(
                        comparison_df.style.format({'Price': '${:,.0f}', 'Location Score': '{:.1f}', 'Safety': '{:.1f}'}),
                        use_container_width=True
                    )
                    
                    # AI insights
                    st.markdown("#### 💭 Market Insights")
                    insights = assistant.compare_cities(features, list(st.session_state.cities.keys())[:5])
                    st.info(insights)
    
    # Tab 3: Manual input (enhanced)
    with tab3:
        st.header("📝 Enter Property Details Manually")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            city = st.selectbox("City", options=list(st.session_state.cities.keys()), key="manual_city")
            sqft = st.number_input("Square Footage", min_value=500, max_value=5000, value=2000, step=100)
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=5, value=3)
        
        with col2:
            bathrooms = st.number_input("Bathrooms", min_value=1, max_value=4, value=2)
            age = st.number_input("Property Age (years)", min_value=0, max_value=50, value=10)
            amenities_score = st.slider("Amenities Score", min_value=1, max_value=10, value=6)
        
        with col3:
            city_info = st.session_state.cities[city]
            st.metric("Location Score", f"{city_info['location_score']:.1f}/10")
            st.metric("Crime Rate", f"{city_info['crime_rate']:.1f}/10")
            st.metric("Market Multiplier", f"{city_info['base_price_mult']:.1f}x")
        
        if st.button("🚀 Predict Price", type="primary", key="manual_predict"):
            # Create feature dict
            features = {
                'sqft': sqft,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'age': age,
                'location_score': city_info['location_score'],
                'amenities_score': amenities_score,
                'crime_rate': city_info['crime_rate'],
                'base_price_multiplier': city_info['base_price_mult']
            }
            
            # Create DataFrame
            df_pred = pd.DataFrame([features])
            
            # Get prediction
            orchestrator = st.session_state.orchestrator
            detailed = orchestrator.detailed_prediction(df_pred)
            
            # Display results
            st.success("### 🎯 Prediction Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.metric("💰 Estimated Price", f"${detailed['ensemble'][0]:,.2f}")
                st.metric("📍 Location", city)
                
                # Price breakdown
                st.markdown("#### 💵 Price Breakdown:")
                base_structural = sqft * 150 + bedrooms * 20000 + bathrooms * 15000
                location_impact = city_info['location_score'] * 30000 * city_info['base_price_mult']
                age_impact = age * 1000
                
                st.text(f"Base Structural Value: ${base_structural:,.0f}")
                st.text(f"Location Premium: +${location_impact:,.0f}")
                st.text(f"Age Depreciation: -${age_impact:,.0f}")
            
            with col2:
                st.markdown("#### 🤖 Individual Agent Predictions:")
                for agent_name, pred in detailed['agents'].items():
                    st.text(f"{agent_name}:")
                    st.text(f"  ${pred[0]:,.2f}")
                
                # Visualization of agent contributions
                agent_names = [name.split()[0] for name in detailed['agents'].keys()]
                agent_values = [pred[0] for pred in detailed['agents'].values()]
                
                fig = go.Figure(data=[
                    go.Bar(x=agent_names, y=agent_values, 
                          marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
                ])
                fig.update_layout(
                    title="Agent Predictions",
                    xaxis_title="Agent",
                    yaxis_title="Price ($)",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### 📊 Feature Analysis:")
            feature_df = pd.DataFrame([features])
            st.dataframe(feature_df.style.format("{:.2f}"), use_container_width=True)
    
    # Tab 4: NEW - Market Analysis
    with tab4:
        st.header("📊 Market Analysis Dashboard")
        
        df = st.session_state.df
        
        # Summary statistics
        st.markdown("### 📈 Market Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Price", f"${df['price'].mean():,.0f}")
        with col2:
            st.metric("Median Price", f"${df['price'].median():,.0f}")
        with col3:
            st.metric("Highest Price", f"${df['price'].max():,.0f}")
        with col4:
            st.metric("Total Properties", f"{len(df):,}")
        
        st.markdown("---")
        
        # City analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏙️ Average Prices by City")
            city_prices = df.groupby('city')['price'].mean().sort_values(ascending=False)
            
            fig = px.bar(
                x=city_prices.index,
                y=city_prices.values,
                title="Average Property Prices by City",
                labels={'x': 'City', 'y': 'Average Price ($)'},
                color=city_prices.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📏 Price Distribution by Square Footage")
            fig = px.scatter(
                df.sample(500),
                x='sqft',
                y='price',
                color='city',
                title="Price vs Square Footage",
                labels={'sqft': 'Square Footage', 'price': 'Price ($)'},
                opacity=0.6
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Advanced analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🛏️ Price by Bedrooms")
            bedroom_prices = df.groupby('bedrooms')['price'].mean()
            
            fig = go.Figure(data=[
                go.Bar(x=bedroom_prices.index, y=bedroom_prices.values, 
                      marker_color='lightblue')
            ])
            fig.update_layout(
                xaxis_title="Number of Bedrooms",
                yaxis_title="Average Price ($)",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📍 Location Score vs Price")
            fig = px.scatter(
                df.sample(500),
                x='location_score',
                y='price',
                color='crime_rate',
                title="Impact of Location Quality on Price",
                labels={
                    'location_score': 'Location Score',
                    'price': 'Price ($)',
                    'crime_rate': 'Crime Rate'
                },
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Detailed city comparison
        st.markdown("### 🌍 Detailed City Comparison")
        
        city_stats = df.groupby('city').agg({
            'price': ['mean', 'median', 'min', 'max'],
            'sqft': 'mean',
            'bedrooms': 'mean',
            'age': 'mean',
            'location_score': 'first',
            'crime_rate': 'first'
        }).round(2)
        
        city_stats.columns = ['Avg Price', 'Median Price', 'Min Price', 'Max Price', 
                              'Avg SqFt', 'Avg Bedrooms', 'Avg Age', 'Location Score', 'Crime Rate']
        
        # Format the dataframe
        styled_df = city_stats.style.format({
            'Avg Price': '${:,.0f}',
            'Median Price': '${:,.0f}',
            'Min Price': '${:,.0f}',
            'Max Price': '${:,.0f}',
            'Avg SqFt': '{:.0f}',
            'Avg Bedrooms': '{:.1f}',
            'Avg Age': '{:.1f}',
            'Location Score': '{:.1f}',
            'Crime Rate': '{:.1f}'
        }).background_gradient(subset=['Avg Price'], cmap='YlGn')
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Market insights
        st.markdown("---")
        st.markdown("### 💡 Market Insights")
        
        most_expensive = city_prices.idxmax()
        least_expensive = city_prices.idxmin()
        avg_price_diff = city_prices.max() - city_prices.min()
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            st.info(f"""
            **Most Expensive Market**
            
            {most_expensive}
            
            Avg: ${city_prices.max():,.0f}
            """)
        
        with insight_col2:
            st.info(f"""
            **Most Affordable Market**
            
            {least_expensive}
            
            Avg: ${city_prices.min():,.0f}
            """)
        
        with insight_col3:
            st.info(f"""
            **Price Range**
            
            Difference: ${avg_price_diff:,.0f}
            
            {(avg_price_diff/city_prices.min()*100):.1f}% variation
            """)
        
        # Download data
        st.markdown("---")
        st.markdown("### 📥 Export Data")
        
        csv = city_stats.to_csv()
        st.download_button(
            label="Download City Statistics (CSV)",
            data=csv,
            file_name=f"city_statistics_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()