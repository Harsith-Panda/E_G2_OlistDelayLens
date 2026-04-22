import os
import pandas as pd

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_RAW     = os.path.join(ROOT, 'data', 'raw')
PATH_OUT     = os.path.join(ROOT, 'data', 'processed', 'olist_cleaned_data.csv')
os.makedirs(os.path.dirname(PATH_OUT), exist_ok=True)

# Loading raw files
print("Loading raw files...")
orders      = pd.read_csv(f'{PATH_RAW}/olist_orders_dataset.csv')
items       = pd.read_csv(f'{PATH_RAW}/olist_order_items_dataset.csv')
customers   = pd.read_csv(f'{PATH_RAW}/olist_customers_dataset.csv')
sellers     = pd.read_csv(f'{PATH_RAW}/olist_sellers_dataset.csv')
reviews     = pd.read_csv(f'{PATH_RAW}/olist_order_reviews_dataset.csv')
geo         = pd.read_csv(f'{PATH_RAW}/olist_geolocation_dataset.csv')

geo_agg = geo.groupby('geolocation_zip_code_prefix').agg(
    geolocation_lat=('geolocation_lat','mean'),
    geolocation_lng=('geolocation_lng','mean')
).reset_index()

# DS Merge
print("Merging...")
df = (orders
      .merge(items,     on='order_id',    how='left')
      .merge(customers, on='customer_id', how='left')
      .merge(sellers,   on='seller_id',   how='left')
      .merge(reviews,   on='order_id',    how='left')
      .merge(geo_agg,   left_on='customer_zip_code_prefix',
             right_on='geolocation_zip_code_prefix', how='left')
      .rename(columns={'geolocation_lat':'customer_lat','geolocation_lng':'customer_lng'})
      .drop(columns=['geolocation_zip_code_prefix'])
      .merge(geo_agg,   left_on='seller_zip_code_prefix',
             right_on='geolocation_zip_code_prefix', how='left')
      .rename(columns={'geolocation_lat':'seller_lat','geolocation_lng':'seller_lng'})
      .drop(columns=['geolocation_zip_code_prefix'])
     )

print("Cleaning...")
date_cols = [
    'order_purchase_timestamp','order_approved_at',
    'order_delivered_carrier_date','order_delivered_customer_date',
    'order_estimated_delivery_date','shipping_limit_date'
]
for c in date_cols:
    df[c] = pd.to_datetime(df[c], errors='coerce')

df = df[df['order_status'] == 'delivered'].copy()

for c in ['product_weight_g','product_length_cm','product_height_cm','product_width_cm']:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())

for state_col, lat_col, lng_col in [
    ('customer_state','customer_lat','customer_lng'),
    ('seller_state',  'seller_lat',  'seller_lng')
]:
    df[lat_col] = df[lat_col].fillna(df.groupby(state_col)[lat_col].transform('median'))
    df[lng_col] = df[lng_col].fillna(df.groupby(state_col)[lng_col].transform('median'))

# Feature Engineering
df['actual_delivery_days']    = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
df['estimated_delivery_days'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.days
df['delivery_delay']          = df['actual_delivery_days'] - df['estimated_delivery_days']
df['is_late']                 = df['delivery_delay'] > 0
df['is_bad_review']           = df['review_score'] <= 2
df['order_year']              = df['order_purchase_timestamp'].dt.year
df['order_quarter']           = df['order_purchase_timestamp'].dt.to_period('Q').dt.strftime('Q%q-%Y')
df['order_month']             = df['order_purchase_timestamp'].dt.to_period('M').astype(str)
df['shipping_route']          = df['seller_state'] + ' → ' + df['customer_state']

# Deduplicate + outlier cap
if 'review_creation_date' in df.columns:
    df = df.sort_values('review_creation_date')
df = df.drop_duplicates(subset=['order_id','order_item_id'], keep='last')
q_upper = df['actual_delivery_days'].quantile(0.995)
df = df[df['actual_delivery_days'] <= q_upper]

df.to_csv(PATH_OUT, index=False)
print(f"Done → {PATH_OUT} | Shape: {df.shape}")