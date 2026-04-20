import pandas as pd
import numpy as np
import os

print("--- STARTING UPDATED ETL PIPELINE ---")

PATH_RAW = 'data/raw/'
PATH_MERGED = 'data/merged/olist_merged_data.csv'
PATH_PROCESSED = 'data/processed/olist_cleaned_data.csv'

# 1. PE-PROCESSING GEOLOCATION
print("Loading and aggregating Geolocation...")
geo = pd.read_csv(os.path.join(PATH_RAW, 'olist_geolocation_dataset.csv'))
geo_agg = geo.groupby('geolocation_zip_code_prefix').agg({
    'geolocation_lat': 'mean',
    'geolocation_lng': 'mean'
}).reset_index()

# 2. LOADING DATASETS
print("Loading core datasets...")
orders = pd.read_csv(os.path.join(PATH_RAW, 'olist_orders_dataset.csv'))
items = pd.read_csv(os.path.join(PATH_RAW, 'olist_order_items_dataset.csv'))
customers = pd.read_csv(os.path.join(PATH_RAW, 'olist_customers_dataset.csv'))
sellers = pd.read_csv(os.path.join(PATH_RAW, 'olist_sellers_dataset.csv'))
products = pd.read_csv(os.path.join(PATH_RAW, 'olist_products_dataset.csv'))
reviews = pd.read_csv(os.path.join(PATH_RAW, 'olist_order_reviews_dataset.csv'))
translation = pd.read_csv(os.path.join(PATH_RAW, 'product_category_name_translation.csv'))

# 3. TRANSLATION & MERGING
print("Merging data...")
products = products.merge(translation, on='product_category_name', how='left')
products['product_category_name'] = products['product_category_name_english'].fillna(products['product_category_name'])
products.drop(columns=['product_category_name_english'], inplace=True)

df = orders.merge(items, on='order_id', how='left')\
           .merge(customers, on='customer_id', how='left')\
           .merge(sellers, on='seller_id', how='left')\
           .merge(products, on='product_id', how='left')\
           .merge(reviews, on='order_id', how='left')

# Adding Geolocation
df = df.merge(geo_agg, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')\
       .rename(columns={'geolocation_lat': 'customer_lat', 'geolocation_lng': 'customer_lng'})\
       .drop(columns=['geolocation_zip_code_prefix'])

df = df.merge(geo_agg, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')\
       .rename(columns={'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'})\
       .drop(columns=['geolocation_zip_code_prefix'])

os.makedirs('data/merged', exist_ok=True)
df.to_csv(PATH_MERGED, index=False)
print(f"Merged data saved to {PATH_MERGED}")

# 4. CLEANING
print("Standardising and Cleaning...")
date_cols = [
    'order_purchase_timestamp', 'order_approved_at', 
    'order_delivered_carrier_date', 'order_delivered_customer_date', 
    'order_estimated_delivery_date', 'shipping_limit_date', 
    'review_creation_date', 'review_answer_timestamp'
]
for col in date_cols:
    df[col] = pd.to_datetime(df[col])

df = df[df['order_status'] == 'delivered']

# Impute physical dimensions
physical_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
for col in physical_cols:
    df[col] = df[col].fillna(df[col].median())

# Handle missing coordinates (impute with state-level median if missing)
df['customer_lat'] = df['customer_lat'].fillna(df.groupby('customer_state')['customer_lat'].transform('median'))
df['customer_lng'] = df['customer_lng'].fillna(df.groupby('customer_state')['customer_lng'].transform('median'))
df['seller_lat'] = df['seller_lat'].fillna(df.groupby('seller_state')['seller_lat'].transform('median'))
df['seller_lng'] = df['seller_lng'].fillna(df.groupby('seller_state')['seller_lng'].transform('median'))

# Feature engineering
df['actual_delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
df['estimated_delivery_days'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.days
df['delivery_delay'] = df['actual_delivery_days'] - df['estimated_delivery_days']
df['is_late'] = df['delivery_delay'] > 0

# Deduplicate
df = df.sort_values('review_creation_date').drop_duplicates(subset=['order_id', 'order_item_id'], keep='last')

# Outliers
q_upper = df['actual_delivery_days'].quantile(0.995)
df = df[df['actual_delivery_days'] <= q_upper]

os.makedirs('data/processed', exist_ok=True)
df.to_csv(PATH_PROCESSED, index=False)

print(f"--- SUCCESS: FINAL CLEANED DATA SAVED TO {PATH_PROCESSED} ---")
