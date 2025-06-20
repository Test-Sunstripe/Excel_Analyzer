import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from dotenv import load_dotenv
import os
import base64
import io
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import datetime
import re

# Page configuration
st.set_page_config(page_title="AI Analytics Dashboard", layout="wide")

# Load API Key from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'current_view' not in st.session_state:
    st.session_state.current_view = "Upload"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_name' not in st.session_state:
    st.session_state.user_name = None

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .dashboard-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .download-btn {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 4px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .upload-section {
        border: 2px dashed #1E88E5;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .success-message {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #2E7D32;
        margin: 10px 0;
    }
    .info-box {
        background-color: #E3F2FD;
        border-left: 5px solid #1E88E5;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0; 
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 5px solid #1E88E5;
        align-self: flex-end;
    }
    .bot-message {
        background-color: #F5F5F5;
        border-left: 5px solid #9E9E9E;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        height: 60vh;
        overflow-y: auto;
        padding: 1rem;
        background-color: #FAFAFA;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .message-time {
        font-size: 0.8rem;
        color: #757575;
        margin-top: 0.3rem;
        align-self: flex-end;
    }
    .greeting-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)


def get_download_link(df, filename, link_text):
    """Generate a download link for the dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-btn">{link_text}</a>'
    return href


def create_vector_store(df):
    """Create FAISS vector store from dataframe for semantic search"""
    # Convert dataframe to text documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = []

    # Create documents from each row
    for i, row in df.iterrows():
        content = " ".join([f"{col}: {val}" for col, val in row.items()])
        documents.append(Document(page_content=content, metadata={"row_index": i}))

    # Split documents
    docs = text_splitter.split_documents(documents)

    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


def load_data(file):
    """Load data from an uploaded CSV or Excel file."""
    try:
        if file.name.endswith('.csv'):
            # Try to parse dates automatically
            df = pd.read_csv(file, parse_dates=True)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None

        # Detect potential date columns that weren't automatically parsed
        for col in df.select_dtypes(include=['object']).columns:
            # Try to convert to datetime
            try:
                # First check if column name suggests a date
                if any(date_hint in col.lower() for date_hint in ['date', 'time', 'day', 'year', 'month', 'dt_']):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    continue

                # Sample the first few non-null values to check if they look like dates
                sample = df[col].dropna().head(5)
                if len(sample) > 0:
                    # Check if the column might contain dates
                    if all(isinstance(val, str) for val in sample):
                        # Look for date-like patterns in the sample
                        date_patterns = [
                            r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}',  # yyyy-mm-dd or dd/mm/yyyy formats
                            r'\d{1,2}[-/]\w{3}[-/]\d{2,4}',  # dd-mmm-yyyy formats
                            r'\w{3,9} \d{1,2},? \d{2,4}'  # Month dd, yyyy formats
                        ]

                        is_date_like = any(
                            any(re.search(pattern, str(val)) for pattern in date_patterns)
                            for val in sample
                        )

                        if is_date_like:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                # If conversion fails, keep as is
                pass

        # Basic data cleaning
        # Fill missing numeric values with mean
        for col in df.select_dtypes(include=['number']).columns:
            df[col] = df[col].fillna(df[col].mean())

        # Fill missing categorical values with mode
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].notna().any():
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna("Unknown")

        # Fill missing datetime values with the median date
        for col in df.select_dtypes(include=['datetime']).columns:
            if df[col].notna().any():
                median_date = df[col].dropna().median()
                df[col] = df[col].fillna(median_date)

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def generate_visualization_dashboard(df):
    """Generate a comprehensive dashboard with multiple visualizations"""
    st.markdown('<div class="sub-header">Interactive Dashboard</div>', unsafe_allow_html=True)

    # Create tabs for different visualization categories
    tabs = st.tabs(["Overview", "Correlation Analysis", "Distribution Analysis", "Advanced Analysis"])

    with tabs[0]:  # Overview
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Data Summary")
            st.dataframe(df.describe().T)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) >= 1:
                selected_col = st.selectbox("Select column for distribution", numeric_cols)
                fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}",
                                   color_discrete_sequence=['#1E88E5'])
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Missing values visualization
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Missing Values Analysis")
        missing = df.isnull().sum().reset_index()
        missing.columns = ['Column', 'Missing Count']
        missing['Missing Percentage'] = (missing['Missing Count'] / len(df)) * 100
        fig = px.bar(missing[missing['Missing Count'] > 0],
                     x='Column', y='Missing Percentage',
                     title="Missing Values Percentage by Column",
                     color='Missing Percentage',
                     color_continuous_scale='Blues')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:  # Correlation Analysis
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            st.subheader("Correlation Matrix")
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix,
                            text_auto=True,
                            color_continuous_scale='RdBu_r',
                            title="Feature Correlation Heatmap")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Scatter Plot Matrix")
            if len(numeric_cols) > 5:
                selected_cols = st.multiselect("Select columns (max 5 recommended)",
                                               numeric_cols,
                                               default=numeric_cols[:4])
                if selected_cols:
                    fig = px.scatter_matrix(df[selected_cols],
                                            color_discrete_sequence=['#1E88E5'])
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.scatter_matrix(df[numeric_cols],
                                        color_discrete_sequence=['#1E88E5'])
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[2]:  # Distribution Analysis
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categoric_cols = df.select_dtypes(include=['object']).columns.tolist()

        col1, col2 = st.columns(2)

        with col1:
            if numeric_cols:
                st.subheader("Numerical Distributions")
                for col in numeric_cols[:min(4, len(numeric_cols))]:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=df[col], nbinsx=20, name=col,
                                               marker_color='#1E88E5'))
                    fig.update_layout(title=f"Distribution of {col}", height=300)
                    st.plotly_chart(fig, use_container_width=True)

        with col2:
            if categoric_cols:
                st.subheader("Categorical Distributions")
                for col in categoric_cols[:min(4, len(categoric_cols))]:
                    counts = df[col].value_counts().reset_index()
                    counts.columns = [col, 'Count']
                    if len(counts) > 10:
                        # If too many categories, show only top 10
                        counts = counts.head(10)
                        title = f"Top 10 values - {col}"
                    else:
                        title = f"Distribution of {col}"
                    fig = px.pie(counts, values='Count', names=col, title=title)
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[3]:  # Advanced Analysis
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) >= 2:
            st.subheader("PCA Visualization")

            # Only use numeric columns with no missing values
            df_numeric = df[numeric_cols].copy()

            # Standardize the data
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df_numeric)

            # Apply PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df_scaled)

            # Create PCA result dataframe
            pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

            # Try to use a categorical column for coloring if available
            color_options = ['None'] + categoric_cols
            selected_color = st.selectbox("Color points by", color_options)

            if selected_color != 'None' and selected_color in df.columns:
                pca_df[selected_color] = df[selected_color]
                fig = px.scatter(pca_df, x='PC1', y='PC2', color=selected_color,
                                 title="PCA Visualization",
                                 labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                                         'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'})
            else:
                # Apply KMeans clustering
                n_clusters = min(5, len(df))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(df_scaled)
                pca_df['Cluster'] = clusters

                fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                                 title="PCA with KMeans Clustering",
                                 labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                                         'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'})

            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Feature importance
            st.subheader("Principal Component Feature Weights")
            component_weights = pd.DataFrame(
                pca.components_,
                columns=numeric_cols,
                index=['PC1', 'PC2']
            )

            fig = px.imshow(component_weights, text_auto=True, aspect="auto",
                            title="Feature Contribution to Principal Components",
                            color_continuous_scale='RdBu_r')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Need at least 2 numeric columns for advanced analysis.")
        st.markdown('</div>', unsafe_allow_html=True)


def get_greeting():
    """Generate appropriate greeting based on time of day"""
    current_hour = datetime.datetime.now().hour

    if current_hour < 12:
        return "Good morning"
    elif current_hour < 18:
        return "Good afternoon"
    else:
        return "Good evening"


def generate_ai_response(question, df, chat_history):
    """Generate analysis using Gemini AI based on user question with enhanced capabilities for accuracy."""
    try:
        # If API key is not set, provide a message
        if not GEMINI_API_KEY:
            return "To enable AI analysis, please add your Google Gemini API key in the .env file."

        # Enhanced data preparation for better context
        # 1. Standard data info
        data_shape = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"

        # 2. Better column information with sample values
        column_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null_count = df[col].count()
            null_percentage = round((1 - non_null_count / len(df)) * 100, 2)

            # Get sample values based on data type
            if df[col].dtype.kind in 'ifc':  # numeric
                sample = f"min: {df[col].min()}, max: {df[col].max()}, mean: {round(df[col].mean(), 2)}"
            elif df[col].dtype.kind == 'O':  # object/string
                unique_values = df[col].nunique()
                top_values = df[col].value_counts().head(3).to_dict()
                sample = f"{unique_values} unique values. Top values: {top_values}"
            elif df[col].dtype.kind == 'M':  # datetime
                sample = f"range: {df[col].min()} to {df[col].max()}"
            else:
                sample = "sample unavailable"

            column_info.append(f"- {col} ({dtype}): {null_percentage}% missing, {sample}")

        column_desc = "\n".join(column_info)

        # 3. More intelligent data preview that adapts to dataset size
        if len(df) <= 5:
            data_preview = df.to_string(index=False)
        else:
            # Sample data more intelligently - first 2, last 2, and 1 random middle row
            middle_idx = len(df) // 2
            sampled_rows = pd.concat([
                df.head(2),
                df.iloc[[middle_idx]],
                df.tail(2)
            ])
            data_preview = sampled_rows.to_string(index=True)

        # 4. Enhanced statistical summary
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

        # 5. Generate specialized summaries based on column types
        numeric_summary = ""
        if numeric_cols:
            numeric_summary = df[numeric_cols].describe().to_string()

        categorical_summary = ""
        if categorical_cols:
            cat_summary_parts = []
            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                unique_count = df[col].nunique()
                cat_summary_parts.append(f"{col}: {unique_count} unique values")
                if unique_count <= 10:  # Only show distribution for columns with few categories
                    distribution = df[col].value_counts(normalize=True).head(5).to_dict()
                    cat_summary_parts.append(f"  Top values: {distribution}")
            categorical_summary = "\n".join(cat_summary_parts)

        # 6. Calculate correlations if there are numeric columns
        correlation_info = ""
        if len(numeric_cols) >= 2:
            # Find strong correlations
            corr_matrix = df[numeric_cols].corr()
            strong_corrs = []

            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) >= 0.5:  # Only report strong correlations
                        strong_corrs.append(f"{numeric_cols[i]} & {numeric_cols[j]}: {round(corr_value, 2)}")

            if strong_corrs:
                correlation_info = "Strong correlations:\n" + "\n".join(strong_corrs)
            else:
                correlation_info = "No strong correlations found between numeric variables."

        # 7. Enhanced data quality insights
        data_quality_insights = []

        # Check for outliers in numeric columns
        outlier_info = []
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
            if outliers > 0:
                outlier_pct = round((outliers / len(df)) * 100, 2)
                outlier_info.append(f"{col}: {outliers} outliers ({outlier_pct}%)")

        if outlier_info:
            data_quality_insights.append("Outlier detection:\n" + "\n".join(outlier_info))

        # 8. Include recent chat history for context (last 3 exchanges)
        recent_history = chat_history[-6:] if len(chat_history) > 0 else []
        chat_context = "\n".join([f"{'User' if i % 2 == 0 else 'AI'}: {msg}" for i, msg in enumerate(recent_history)])

        # 9. Extracting question type to provide more accurate responses
        question_lower = question.lower()
        question_type = "general"

        # Determine question type based on keywords
        if any(word in question_lower for word in
               ["average", "mean", "median", "sum", "count", "maximum", "minimum", "std", "standard deviation"]):
            question_type = "statistical"
        elif any(word in question_lower for word in ["predict", "forecast", "estimate", "projection", "trend"]):
            question_type = "predictive"
        elif any(word in question_lower for word in
                 ["compare", "correlation", "relationship", "associated", "versus", "vs"]):
            question_type = "comparative"
        elif any(word in question_lower for word in ["group", "segment", "categorize", "classify"]):
            question_type = "grouping"

        # Create a more structured prompt with clearer instructions based on question type
        prompt = f"""You are an advanced data analysis AI assistant specializing in providing accurate, precise answers.

        # Dataset Information
        {data_shape}

        # Columns Detail
        {column_desc}

        # Data Preview
        {data_preview}

        # Statistical Summary for Numeric Columns
        {numeric_summary}

        # Categorical Data Summary
        {categorical_summary}

        # Correlation Information
        {correlation_info}

        # Data Quality Insights
        {'; '.join(data_quality_insights)}

        # Recent Conversation
        {chat_context}

        # User Question
        {question}

        # Question Type
        {question_type}

        Based on the question type "{question_type}", please provide an accurate analysis following these guidelines:

        """

        # Add specific instructions based on question type
        if question_type == "statistical":
            prompt += """
            For this statistical question:
            1. Calculate the exact statistic(s) requested using the actual data values
            2. Show your work by displaying the formula and intermediate values
            3. Present the final result with appropriate precision
            4. Include relevant context such as sample size and any potential issues with the calculation
            5. When appropriate, add visual descriptions of the distribution
            """
        elif question_type == "predictive":
            prompt += """
            For this predictive/estimation question:
            1. Clearly state that you're making a data-based estimate and its limitations
            2. Identify the variables and trends most relevant to the prediction
            3. Describe the method you would use for forecasting based on available data
            4. If possible, provide a range of estimates rather than a single value
            5. Explain what additional data would strengthen the prediction
            """
        elif question_type == "comparative":
            prompt += """
            For this comparative analysis question:
            1. Directly compare the variables requested using appropriate statistical measures
            2. Calculate correlation coefficients or other relationship metrics if relevant
            3. Note any confounding variables that might affect interpretation
            4. Describe the strength and direction of relationships found
            5. Highlight any potential causal relationships but avoid claiming causation without evidence
            """
        elif question_type == "grouping":
            prompt += """
            For this grouping/categorization question:
            1. Create the requested segments based on clear criteria
            2. Calculate summary statistics for each group
            3. Highlight key differences between groups
            4. Note the relative sizes of each group
            5. Suggest potential insights from the grouping pattern
            """
        else:
            prompt += """
            For this general question:
            1. Provide a direct, concise answer to the specific question asked
            2. Support your response with relevant data from the dataset
            3. Include only the analysis that directly answers the question
            4. Be transparent about any assumptions made
            """

        # Add general quality requirements
        prompt += """

        # General Requirements for ALL responses:
        - Begin with a direct answer to the question in 1-2 sentences
        - Provide ONLY factual statements based on the actual data
        - Use precise numbers instead of vague terms
        - Cite specific data points to support your analysis
        - If the data is insufficient to answer fully, clearly state this limitation
        - Format numbers appropriately (percentages, decimal places, etc.)
        - Use markdown formatting for better readability
        - Keep your response focused and concise without unnecessary explanations

        Remember: Accuracy is the HIGHEST priority. If you're unsure about any calculation or conclusion, be transparent about the uncertainty and provide your best estimate with appropriate caveats.
        """

        # Call Gemini API with improved structured instructions
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Set generation parameters for analytically accurate responses
        generation_config = {
            "temperature": 0.1,  # Lower temperature for more precise, analytical responses
            "top_p": 0.95,
            "top_k": 40,
        }

        # Safety settings
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]

        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        # Handle the response
        if hasattr(response, 'text'):
            return response.text
        else:
            # Handle legacy response format
            return response.candidates[0].content.parts[0].text

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()

        # Create a more helpful error message with debugging information
        error_message = f"""I encountered an issue while analyzing your data. Here's what happened:

**Error**: {str(e)}

Let me suggest how we might proceed:

1. If you asked about columns that don't exist in your data, please check the column names.
2. If you asked for a calculation that isn't possible with your data, try rephrasing your question.
3. For complex analyses, try breaking your question into smaller parts.

Could you try rephrasing your question, or ask about something else in your dataset?"""

        return error_message


def display_chat_message(message, is_user=False):
    """Display a chat message with styling"""
    message_class = "user-message" if is_user else "bot-message"
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    st.markdown(f"""
    <div class="chat-message {message_class}">
        <div>{message}</div>
        <div class="message-time">{timestamp}</div>
    </div>
    """, unsafe_allow_html=True)


def ai_chatbot_page():
    """Display the AI chatbot page with conversation history"""
    if st.session_state.data is not None:
        st.markdown('<div class="sub-header">AI Data Analysis Assistant</div>', unsafe_allow_html=True)

        # Initialize chat if first visit to this page
        if len(st.session_state.chat_history) == 0:
            greeting = get_greeting()
            welcome_message = f"{greeting}! I'm your AI data analysis assistant. I can help you explore and understand your dataset. What would you like to know about your data?"
            st.session_state.chat_history.append(welcome_message)

        # Display chat container with history
        # st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for i, message in enumerate(st.session_state.chat_history):
            is_user = i % 2 != 0  # Even indices are bot messages, odd are user
            display_chat_message(message, is_user)
        st.markdown('</div>', unsafe_allow_html=True)

        # Chat input using st.chat_input
        user_question = st.chat_input("Ask about your data...", key="user_question")

        # Process user input
        if user_question:
            # Add user message to chat history
            st.session_state.chat_history.append(user_question)

            # Get AI response
            with st.spinner("Analyzing data..."):
                response = generate_ai_response(user_question, st.session_state.data, st.session_state.chat_history)

            # Add AI response to chat history
            st.session_state.chat_history.append(response)

            # Rerun to display updated chat
            st.rerun()

        # Add option to clear chat history
        if st.button("Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    else:
        st.error("No data available. Please upload data first.")


def display_data_results(df):
    """Display data results after successful upload"""
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)

    # Data preview with expandable view
    with st.expander("Data Preview", expanded=True):
        st.dataframe(df.head(10))

    # Data info in a more compact format
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Rows", df.shape[0])
    cols[1].metric("Columns", df.shape[1])
    cols[2].metric("Numeric Columns", len(df.select_dtypes(include=['number']).columns))
    cols[3].metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))
    st.markdown('</div>', unsafe_allow_html=True)

    # Quick stats
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Missing Data Summary")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing Values': missing.values,
                'Percentage': round(missing.values / len(df) * 100, 2)
            })
            st.dataframe(missing_df)
        else:
            st.markdown('<div class="success-message">No missing values detected in the dataset!</div>',
                        unsafe_allow_html=True)

    with col2:
        st.subheader("Numerical Summary")
        if not df.select_dtypes(include=['number']).empty:
            st.dataframe(df.describe().T)
        else:
            st.info("No numerical columns available in the dataset.")

    # # Download options
    # st.subheader("Download Options")
    # download_cols = st.columns(2)
    # with download_cols[0]:
    #     st.markdown(get_download_link(df, "processed_data.csv", "Download as CSV"),
    #                 unsafe_allow_html=True)
    # with download_cols[1]:
    #     buffer = io.BytesIO()
    #     with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    #         df.to_excel(writer, sheet_name='Data', index=False)
    #         # Add a summary sheet
    #         pd.DataFrame({
    #             'Statistic': ['Rows', 'Columns', 'Missing Values'],
    #             'Value': [df.shape[0], df.shape[1], df.isnull().sum().sum()]
    #         }).to_excel(writer, sheet_name='Summary', index=False)
    #
    #     b64 = base64.b64encode(buffer.getvalue()).decode()
    #     href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="processed_data.xlsx" class="download-btn">Download as Excel</a>'
    #     st.markdown(href, unsafe_allow_html=True)
    #
    # st.markdown('</div>', unsafe_allow_html=True)


# Modify the data_upload_page() function
def data_upload_page():
    """Display the data upload page with improved UX"""
    st.markdown('<div class="sub-header">Data Upload & Analysis</div>', unsafe_allow_html=True)

    # Create an upload section with better visual cues
    # st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📤 Upload Your Dataset")
    st.write("Supported formats: CSV and Excel files")
    file = st.file_uploader("Choose a file", type=["csv", "xlsx"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Check if we have data in session state OR a file is uploaded
    if file:
        # Show a spinner during processing
        with st.spinner("Loading and processing data..."):
            df = load_data(file)
            if df is not None:
                st.session_state.data = df
                # Reset chat history when new data is uploaded
                st.session_state.chat_history = []

                # Create vector store for semantic search
                try:
                    with st.spinner("Creating vector index for semantic search..."):
                        vector_store = create_vector_store(df)
                        st.session_state.vector_store = vector_store
                        st.success("✅ Data loaded and vector index created successfully!")
                except Exception as e:
                    st.warning(f"Data loaded, but couldn't create vector index: {e}")

                # Display data results immediately after successful upload
                display_data_results(df)

                # Add a prompt to try the AI chatbot
                st.markdown("""
                <div class="success-message" style="text-align: center; padding: 20px;">
                    <h3>🎉 Data uploaded successfully!</h3>
                    <p>Now you can chat with our AI assistant about your data. Click the "AI Chatbot" button in the sidebar to get started.</p>
                </div>
                """, unsafe_allow_html=True)

                # REMOVED: Auto-navigate to AI Chatbot
                # The following code is removed to prevent automatic navigation
                # if len(st.session_state.chat_history) == 0:
                #     st.session_state.current_view = "AI"
                #     st.rerun()
            else:
                st.error("Failed to load data. Please check the file format and try again.")

    # Display existing data if available in session state
    elif st.session_state.data is not None:
        st.success("✅ Data already loaded")
        display_data_results(st.session_state.data)

        # Add option to clear the data
        if st.button("Clear Loaded Data"):
            st.session_state.data = None
            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.rerun()


def visualization_page():
    """Display the visualization dashboard page"""
    if st.session_state.data is not None:
        generate_visualization_dashboard(st.session_state.data)
    else:
        st.error("No data available. Please upload data first.")


def main():
    st.markdown('<div class="main-header">Advanced AI-Powered Analytics Dashboard</div>', unsafe_allow_html=True)

    # Sidebar for navigation with improved design
    with st.sidebar:
        st.image("https://img.freepik.com/premium-vector/modern-data-analytic-accounting-logo-design_273648-1180.jpg", use_container_width=True)

        st.title("Navigation")

        # Data info display in sidebar if data is loaded
        if st.session_state.data is not None:
            st.markdown("### Dataset Info")
            st.info(f"📊 {st.session_state.data.shape[0]} rows × {st.session_state.data.shape[1]} columns")

        # Navigation buttons
        st.markdown("### Dashboard Sections")
        if st.button(" 📤 Upload Data ", key="nav_upload", use_container_width=True):
            st.session_state.current_view = "Upload"
            st.rerun()

        viz_button = st.button("📊 Data Visualization", use_container_width=True)
        if viz_button:
            if st.session_state.data is not None:
                st.session_state.current_view = "Visualization"
            else:
                st.warning("Please upload data first!")
                st.session_state.current_view = "Upload"

        ai_button = st.button("🤖 AI Analysis", use_container_width=True)
        if ai_button:
            if st.session_state.data is not None:
                st.session_state.current_view = "AI"
            else:
                st.warning("Please upload data first!")
                st.session_state.current_view = "Upload"

        # Add footer with credits
        st.markdown("---")
        st.markdown("### About")
        with st.expander("About This Dashboard"):
            st.write("""
                      This AI-powered dashboard enables you to:  
                      - Upload and analyze your data effortlessly  
                      - Generate interactive visualizations and insights  
                      - Chat with AI to ask questions based on your uploaded data  
                      - Perform automated data cleaning and preprocessing for better analysis  
                       """)
        st.markdown("AI-Powered Analytics Dashboard")
        st.markdown("Created by Lakshman Kodela")

    # Main content area based on current view
    if st.session_state.current_view == "Upload":
        data_upload_page()
    elif st.session_state.current_view == "Visualization" and st.session_state.data is not None:
        visualization_page()
    elif st.session_state.current_view == "AI" and st.session_state.data is not None:
        ai_chatbot_page()
    elif st.session_state.current_view != "Upload":
        st.error("Please upload data first to access this section.")
        st.warning("Please upload data first!")
        st.chat_message("Please upload data first!")
        st.session_state.current_view = "Upload"
        st.rerun()


# Run the app
if __name__ == "__main__":
    main()

