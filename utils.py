import pandas as pd 

def process_amenities(df, amenities_to_check):
    """
    Process amenities column to create binary features for specific amenities
    """
    amenities_col = 'amenities'
    
    if amenities_col not in df.columns:
        print(f"Column {amenities_col} not found")
        return df
    
    # Create binary columns for each amenity
    for amenity in amenities_to_check:
        # Check if amenity is mentioned in the amenities string
        # Using case-insensitive search
        df[f'has_{amenity.lower()}'] = df[amenities_col].str.contains(
            amenity, case=False, na=False
        )
    
    print(f"Created binary features for: {amenities_to_check}")
    for amenity in amenities_to_check:
        count = df[f'has_{amenity.lower()}'].sum()
        #print(f"  has_{amenity.lower()}: {count} properties have this amenity")
    
    return df

# Rest of your transformation code
def create_buckets(df, column, n_buckets=10):
    """Create buckets for categorical column"""
    value_counts = df[column].value_counts()
    top_categories = value_counts.head(n_buckets-1).index.tolist()
    
    df[f'{column}_bucketed'] = df[column].apply(
        lambda x: x if x in top_categories else 'Other'
    )
    
    #print(f"{column} - Original unique values: {df[column].nunique()}")
    #print(f"{column} - After bucketing: {df[f'{column}_bucketed'].nunique()}")
    
    return f'{column}_bucketed'

def transform_data_complete(df):
    # 1. Convert binary columns
    binary_cols = ['host_is_superhost', 'host_identity_verified', 'is_location_exact', 'instant_bookable']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'t': True, 'f': False})
    print('Binary variables converted')
    
    # 2. Process amenities
    amenities_to_check = ['Internet', 'Wifi', 'Kitchen', 'Heating', 'Washer', 'Cable TV']
    df = process_amenities(df, amenities_to_check)
    
    # 3. Handle other categorical columns
    categorical_cols = ['room_type', 'bed_type', 'cancellation_policy', 
                       'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 
                       'zipcode', 'property_type']
    
    bucketed_columns = []
    for col in categorical_cols:
        if col in df.columns:
            bucketed_col = create_buckets(df, col, n_buckets=10)
            bucketed_columns.append(bucketed_col)
    
    # 4. One-hot encode
    df_encoded = pd.get_dummies(df, columns=bucketed_columns, prefix=bucketed_columns)
    print('Categorical variables encoded')
    
    # 5. Clean up - drop original columns
    columns_to_drop = categorical_cols + ['amenities']  # Drop original amenities too
    df_final = df_encoded.drop(columns_to_drop, axis=1, errors='ignore')

    
    return df_final

import openai
import time

from openai import OpenAI
import pandas as pd
import time
import os
from dotenv import load_dotenv

def create_openai_embeddings(df, text_columns, api_key, dimensions=256):
    """Create embeddings using OpenAI's embedding model (new client)"""
    
    # Initialize the new client
    client = OpenAI(api_key=api_key)
    
    embedding_features = []
    
    for col in text_columns:
        print(f"Processing column: {col}")
        if col in df.columns:
            text_data = df[col].fillna('').astype(str)
            embeddings = []
            
            for idx, text in enumerate(text_data):
                try:
                    if idx % 10 == 0:  # Progress indicator
                        print(f'Processing {idx}/{len(text_data)}')
                    
                    response = client.embeddings.create(
                        input=text[:8000],  # Truncate to avoid token limits
                        model="text-embedding-3-small",
                        dimensions=dimensions
                    )
                    embeddings.append(response.data[0].embedding)
                    
                except Exception as e:
                    print(f"Error at index {idx}: {e}")
                    # Check the actual embedding dimension for this model
                    embeddings.append([0] * dimensions)  # Adjust if needed
            
            embedding_df = pd.DataFrame(
                embeddings,
                columns=[f'{col}_openai_{i}' for i in range(len(embeddings[0]))],
                index=df.index
            )
            embedding_features.append(embedding_df)
    
    return pd.concat(embedding_features, axis=1)

def create_batch_embeddings(df, text_columns, api_key, batch_size=100, dimensions=256):
    """Process embeddings in batches - much faster!"""
    
    client = OpenAI(api_key=api_key)
    embedding_features = []
    
    for col in text_columns:
        print(f"Processing {col} in batches...")
        if col in df.columns:
            text_data = df[col].fillna('').astype(str).tolist()
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(text_data), batch_size):
                batch = text_data[i:i+batch_size]
                print(f"  Batch {i//batch_size + 1}/{len(text_data)//batch_size + 1}")
                
                try:
                    # Truncate texts to avoid token limits
                    batch = [text[:4000] for text in batch]
                    
                    response = client.embeddings.create(
                        input=batch,  # Send entire batch at once!
                        model="text-embedding-3-small",
                        dimensions=dimensions
                    )
                    
                    # Extract embeddings from batch response
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    time.sleep(0.1)  # Brief pause between batches
                    
                except Exception as e:
                    print(f"Batch error: {e}")
                    # Add zero embeddings for failed batch
                    all_embeddings.extend([[0] * dimensions] * len(batch))
            
            embedding_df = pd.DataFrame(
                all_embeddings,
                columns=[f'{col}_openai_{i}' for i in range(dimensions)],
                index=df.index
            )
            embedding_features.append(embedding_df)
    
    return pd.concat(embedding_features, axis=1)

from sklearn.decomposition import PCA

def reduce_embedding_dimensions(embedding_features, n_components=10):
    """Reduce embedding dimensions using PCA"""
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embedding_features)
    
    reduced_df = pd.DataFrame(
        reduced_embeddings,
        columns=[f'text_pca_{i}' for i in range(n_components)],
        index=embedding_features.index
    )
    
    print(f"Reduced from {embedding_features.shape[1]} to {n_components} dimensions")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    return reduced_df