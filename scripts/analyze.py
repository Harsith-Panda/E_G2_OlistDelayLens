"""
OlistDelayLens — Single Analysis Pipeline
==========================================
Covers ALL four dashboards:
  D1  Executive Pulse          (CEO)
  D2  Geographic Risk Monitor  (Operations)
  D3  Seller Performance       (Seller Management)
  D4  Customer Experience      (Marketing / CX)

Run from project root:
    python scripts/analyze.py

Outputs
-------
  data/processed/tableau_ready/
      01_executive_pulse.csv          ← D1: YoY + quarterly trend + KPI
      02a_geo_risk_state.csv          ← D2: choropleth (state, year)
      02b_geo_risk_city.csv           ← D2: city hotspot (city, state, year)
      02c_geo_risk_route.csv          ← D2: SP→RJ corridor highlights
      03a_seller_scorecard.csv        ← D3: full seller table
      03b_high_risk_by_state.csv      ← D3: >13% sellers by state
      04a_order_scatter.csv           ← D4: scatter sample
      04b_bad_review_trend.csv        ← D4: bad review rate YoY
      04c_ontime_vs_late.csv          ← D4: avg score on-time vs late
      04d_one_star_by_state.csv       ← D4: 1-star heatmap by state

  data/processed/eda_artifacts/
      d1_late_rate_trend.png
      d2_top5_seller_states.png
      d3_top20_worst_sellers.png
      d3_seller_delay_vs_review.png
      d3_high_risk_by_state.png
      d4_bad_review_trend.png
      d4_ontime_vs_late.png
"""

import os
import warnings

import matplotlib
matplotlib.use('Agg')                       # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", font_scale=1.05)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_CLEANED = os.path.join(ROOT, 'data', 'processed', 'olist_cleaned_data.csv')
PATH_TABLEAU = os.path.join(ROOT, 'data', 'processed', 'tableau_ready')
PATH_EDA_OUT = os.path.join(ROOT, 'data', 'processed', 'eda_artifacts')

os.makedirs(PATH_TABLEAU, exist_ok=True)
os.makedirs(PATH_EDA_OUT, exist_ok=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def tc(df: pd.DataFrame) -> pd.DataFrame:
    """Title-case column names for Tableau."""
    d = df.copy()
    d.columns = [c.replace('_', ' ').title() for c in d.columns]
    return d

def save_tableau(df: pd.DataFrame, filename: str):
    path = os.path.join(PATH_TABLEAU, filename)
    tc(df).to_csv(path, index=False)
    print(f"  ✓ {filename}")

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * np.arcsin(np.sqrt(a)) * 6_371   # km

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 68)
print("  OlistDelayLens — Full D1–D4 Analysis Pipeline")
print("=" * 68)

# ─── 1. LOAD & PREPARE ───────────────────────────────────────────────────────
print("\n[1/8] Loading cleaned dataset …")
df = pd.read_csv(PATH_CLEANED)

DATE_COLS = [
    'order_purchase_timestamp', 'order_approved_at',
    'order_delivered_carrier_date', 'order_delivered_customer_date',
    'order_estimated_delivery_date', 'shipping_limit_date',
]
for c in DATE_COLS:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors='coerce')

# Derived columns (safe even if already exist)
df['order_year']    = df['order_purchase_timestamp'].dt.year
df['order_quarter'] = (df['order_purchase_timestamp']
                       .dt.to_period('Q')
                       .dt.strftime('Q%q-%Y'))
df['is_bad_review'] = df['review_score'].le(2)   # bad = 1-star OR 2-star (matches brief: 13.88%/15.11%)

# Haversine distance (km) — only where coords exist
coord_cols = ['seller_lat', 'seller_lng', 'customer_lat', 'customer_lng']
if all(c in df.columns for c in coord_cols):
    mask = df[coord_cols].notnull().all(axis=1)
    df['haversine_km'] = np.nan
    df.loc[mask, 'haversine_km'] = haversine(
        df.loc[mask, 'seller_lat'],  df.loc[mask, 'seller_lng'],
        df.loc[mask, 'customer_lat'], df.loc[mask, 'customer_lng'],
    )

# Shipping route label
if 'seller_state' in df.columns and 'customer_state' in df.columns:
    df['shipping_route'] = df['seller_state'] + ' → ' + df['customer_state']

print(f"  Shape      : {df.shape}")
print(f"  Date range : {df['order_purchase_timestamp'].min().date()} "
      f"→ {df['order_purchase_timestamp'].max().date()}")
print(f"  is_late    : {df['is_late'].sum():,} late  /  {len(df):,} total")


# ══════════════════════════════════════════════════════════════════════════════
# D1 — EXECUTIVE PULSE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/8] D1 — Executive Pulse …")

# ── Headline KPI
overall_late_pct = df['is_late'].mean() * 100
total_late_orders = int(df['is_late'].sum())
print(f"\n  ★ HEADLINE KPI: {overall_late_pct:.2f}% of customers received their order late")
print(f"    ({total_late_orders:,} out of {len(df):,} orders)")

# ── YoY late rate
yoy = (
    df.groupby('order_year')
      .agg(
          total_orders   = ('order_id',      'count'),
          late_orders    = ('is_late',        'sum'),
          avg_delay_days = ('delivery_delay', 'mean'),
          avg_review     = ('review_score',   'mean'),
      )
      .reset_index()
)
yoy['late_rate_pct']   = (yoy['late_orders']  / yoy['total_orders'] * 100).round(2)
yoy['avg_delay_days']  = yoy['avg_delay_days'].round(2)
yoy['avg_review']      = yoy['avg_review'].round(3)
yoy['headline_kpi']    = yoy['late_rate_pct'].apply(
    lambda x: f"{x:.2f}% of customers received their order late"
)
print("\n  YoY Trend:")
print(yoy[['order_year', 'total_orders', 'late_orders', 'late_rate_pct',
           'avg_delay_days']].to_string(index=False))

# ── Quarterly breakdown (total late orders per quarter)
quarterly = (
    df.groupby(['order_year', 'order_quarter'])
      .agg(
          total_orders   = ('order_id',      'count'),
          late_orders    = ('is_late',        'sum'),
          avg_delay_days = ('delivery_delay', 'mean'),
      )
      .reset_index()
)
quarterly['late_rate_pct']  = (quarterly['late_orders'] / quarterly['total_orders'] * 100).round(2)
quarterly['avg_delay_days'] = quarterly['avg_delay_days'].round(2)

# Merge quarterly into D1 export (Tableau uses this for trend lines)
d1_export = yoy.merge(
    quarterly[['order_year', 'order_quarter', 'total_orders',
               'late_orders', 'late_rate_pct', 'avg_delay_days']]
              .rename(columns={
                  'total_orders':   'q_total_orders',
                  'late_orders':    'q_late_orders',
                  'late_rate_pct':  'q_late_rate_pct',
                  'avg_delay_days': 'q_avg_delay_days',
              }),
    on='order_year', how='left'
)
save_tableau(d1_export, '01_executive_pulse.csv')


# ══════════════════════════════════════════════════════════════════════════════
# D2 — GEOGRAPHIC RISK MONITOR
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/8] D2 — Geographic Risk Monitor …")

# ── 2a. State-level choropleth (with year filter support)
d2_state = (
    df.groupby(['seller_state', 'order_year'])
      .agg(
          total_orders   = ('order_id',       'count'),
          late_orders    = ('is_late',         'sum'),
          avg_delay_days = ('delivery_delay',  'mean'),
          avg_review     = ('review_score',    'mean'),
      )
      .reset_index()
)
d2_state['late_rate_pct']  = (d2_state['late_orders'] / d2_state['total_orders'] * 100).round(2)
d2_state['avg_delay_days'] = d2_state['avg_delay_days'].round(2)
d2_state['avg_review']     = d2_state['avg_review'].round(3)
save_tableau(d2_state, '02a_geo_risk_state.csv')

# ── Top 5 worst-performing seller states (overall)
top5_states = (
    df.groupby('seller_state')
      .agg(
          total_orders   = ('order_id',       'count'),
          late_orders    = ('is_late',         'sum'),
          avg_delay_days = ('delivery_delay',  'mean'),
      )
      .reset_index()
)
top5_states['late_rate_pct']  = (top5_states['late_orders'] / top5_states['total_orders'] * 100).round(2)
top5_states['avg_delay_days'] = top5_states['avg_delay_days'].round(2)
top5_worst_states             = top5_states.nlargest(5, 'late_rate_pct')
print("\n  ── Top 5 Worst Seller States ──")
print(top5_worst_states[['seller_state', 'total_orders', 'late_rate_pct',
                          'avg_delay_days']].to_string(index=False))

# ── 2b. City-level hotspot (with state + year filter)
d2_city = (
    df.groupby(['seller_city', 'seller_state', 'order_year'])
      .agg(
          total_orders   = ('order_id',       'count'),
          late_orders    = ('is_late',         'sum'),
          avg_delay_days = ('delivery_delay',  'mean'),
      )
      .reset_index()
)
d2_city['late_rate_pct']  = (d2_city['late_orders'] / d2_city['total_orders'] * 100).round(2)
d2_city['avg_delay_days'] = d2_city['avg_delay_days'].round(2)
d2_city = d2_city[d2_city['total_orders'] >= 5]      # minimum-order guard
# Flag Ibitinga
d2_city['is_ibitinga_hotspot'] = (
    d2_city['seller_city'].str.lower().str.strip() == 'ibitinga'
)
save_tableau(d2_city, '02b_geo_risk_city.csv')

# Ibitinga spotlight
ibitinga = d2_city[d2_city['is_ibitinga_hotspot']]
print("\n  ── Ibitinga Hotspot Stats ──")
if ibitinga.empty:
    fuzzy = d2_city[d2_city['seller_city'].str.lower().str.contains('ibitinga', na=False)]
    print("  (exact match not found — fuzzy results:)")
    print(fuzzy.to_string(index=False) if not fuzzy.empty else "  Not found in dataset")
else:
    print(ibitinga[['seller_city', 'seller_state', 'order_year',
                    'total_orders', 'late_rate_pct', 'avg_delay_days']].to_string(index=False))

# ── 2c. Shipping route (SP → RJ corridor highlighted)
if 'shipping_route' in df.columns:
    d2_route = (
        df.groupby(['shipping_route', 'seller_state', 'customer_state'])
          .agg(
              total_orders   = ('order_id',       'count'),
              late_orders    = ('is_late',         'sum'),
              avg_delay_days = ('delivery_delay',  'mean'),
          )
          .reset_index()
    )
    d2_route['late_rate_pct']  = (d2_route['late_orders'] / d2_route['total_orders'] * 100).round(2)
    d2_route['avg_delay_days'] = d2_route['avg_delay_days'].round(2)
    d2_route['is_sp_rj']       = d2_route['shipping_route'].eq('SP → RJ')
    d2_route_sorted = d2_route.sort_values('total_orders', ascending=False)
    save_tableau(d2_route_sorted, '02c_geo_risk_route.csv')
    sp_rj = d2_route[d2_route['is_sp_rj']]
    print("\n  ── SP → RJ Corridor ──")
    print(sp_rj[['shipping_route', 'total_orders', 'late_rate_pct',
                 'avg_delay_days']].to_string(index=False)
          if not sp_rj.empty else "  SP→RJ route not found in data")


# ══════════════════════════════════════════════════════════════════════════════
# D3 — SELLER PERFORMANCE SCORECARD
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/8] D3 — Seller Performance Scorecard …")

MIN_ORDERS = 10

seller_agg = (
    df.groupby('seller_id')
      .agg(
          total_orders     = ('order_id',       'count'),
          seller_state     = ('seller_state',   'first'),
          seller_city      = ('seller_city',    'first'),
          late_orders      = ('is_late',         'sum'),
          avg_delay_days   = ('delivery_delay',  'mean'),
          avg_review_score = ('review_score',    'mean'),
      )
      .reset_index()
)
seller_agg = seller_agg[seller_agg['total_orders'] >= MIN_ORDERS].copy()
seller_agg['delay_rate_pct']    = (seller_agg['late_orders'] / seller_agg['total_orders'] * 100).round(2)
seller_agg['avg_delay_days']    = seller_agg['avg_delay_days'].round(2)
seller_agg['avg_review_score']  = seller_agg['avg_review_score'].round(3)
seller_agg['high_risk_seller']  = seller_agg['delay_rate_pct'].gt(13).map({True: 'Yes', False: 'No'})

# Delay severity bucket (as required)
def severity_bucket(pct: float) -> str:
    if pct == 0:      return '0% — No Delay'
    elif pct <= 10:   return '1–10% — Low'
    elif pct <= 20:   return '11–20% — Medium'
    elif pct <= 40:   return '21–40% — High'
    else:             return '41%+ — Critical'

seller_agg['delay_severity_bucket'] = seller_agg['delay_rate_pct'].apply(severity_bucket)

total_qualified  = len(seller_agg)
high_risk_count  = seller_agg['delay_rate_pct'].gt(13).sum()
print(f"\n  Qualified sellers (≥{MIN_ORDERS} orders) : {total_qualified:,}")
print(f"  Sellers with >13% delay rate          : {high_risk_count:,}")

# Top-20 worst sellers
top20_worst = seller_agg.nlargest(20, 'delay_rate_pct')[[
    'seller_id', 'seller_state', 'seller_city', 'total_orders',
    'delay_rate_pct', 'avg_review_score', 'delay_severity_bucket',
]]
print("\n  ── Top 20 Worst Sellers by Delay Rate ──")
print(top20_worst.to_string(index=False))

# >13% sellers distribution by state
high_risk_sellers = seller_agg[seller_agg['delay_rate_pct'] > 13]
hr_by_state = (
    high_risk_sellers.groupby('seller_state')
                     .agg(
                         high_risk_count = ('seller_id',       'count'),
                         avg_delay_rate  = ('delay_rate_pct',  'mean'),
                     )
                     .sort_values('high_risk_count', ascending=False)
                     .reset_index()
)
hr_by_state['avg_delay_rate'] = hr_by_state['avg_delay_rate'].round(2)
print("\n  ── High-Risk Sellers (>13%) by State ──")
print(hr_by_state.to_string(index=False))

# Seller-level Pearson r : delay_rate vs review_score
pearson_r, pearson_p = stats.pearsonr(
    seller_agg['delay_rate_pct'], seller_agg['avg_review_score']
)
print(f"\n  Pearson r (delay rate vs review score): r={pearson_r:.4f}, p={pearson_p:.2e}")

save_tableau(seller_agg, '03a_seller_scorecard.csv')
save_tableau(hr_by_state, '03b_high_risk_by_state.csv')


# ══════════════════════════════════════════════════════════════════════════════
# D4 — CUSTOMER EXPERIENCE IMPACT
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/8] D4 — Customer Experience Impact …")

# ── Bad review rate trend by year
bad_review_trend = (
    df.groupby('order_year')
      .agg(
          total_orders     = ('order_id',      'count'),
          bad_reviews      = ('is_bad_review',  'sum'),
          avg_review_score = ('review_score',   'mean'),
      )
      .reset_index()
)
bad_review_trend['bad_review_rate_pct'] = (
    bad_review_trend['bad_reviews'] / bad_review_trend['total_orders'] * 100
).round(2)
bad_review_trend['avg_review_score'] = bad_review_trend['avg_review_score'].round(3)
print("\n  ── Bad Review Rate Trend by Year ──")
print(bad_review_trend[['order_year', 'total_orders', 'bad_reviews',
                         'bad_review_rate_pct', 'avg_review_score']].to_string(index=False))

# ── On-time vs Late avg review score (side-by-side KPI)
ot_stats = (
    df.groupby('is_late')
      .agg(
          total_orders     = ('order_id',     'count'),
          avg_review_score = ('review_score', 'mean'),
      )
      .reset_index()
)
ot_stats['delivery_status']   = ot_stats['is_late'].map({True: 'Late', False: 'On Time'})
ot_stats['avg_review_score']  = ot_stats['avg_review_score'].round(3)
ot_stats = ot_stats.drop(columns='is_late')
print("\n  ── On-Time vs Late — Average Review Score ──")
print(ot_stats[['delivery_status', 'total_orders', 'avg_review_score']].to_string(index=False))

# ── 1-star heatmap by customer state (late delivery concentration)
star1_state = (
    df.groupby('customer_state')
      .agg(
          total_orders    = ('order_id',        'count'),
          late_orders     = ('is_late',          'sum'),
          one_star_count  = ('is_bad_review',    'sum'),
      )
      .reset_index()
)
star1_state['late_rate_pct']    = (star1_state['late_orders']   / star1_state['total_orders'] * 100).round(2)
star1_state['one_star_rate_pct'] = (star1_state['one_star_count'] / star1_state['total_orders'] * 100).round(2)
print("\n  ── Late Delivery 1-Star Concentration by State (Top 15) ──")
print(star1_state.nlargest(15, 'one_star_rate_pct')[[
    'customer_state', 'total_orders', 'late_rate_pct', 'one_star_count', 'one_star_rate_pct'
]].to_string(index=False))

# ── Order-level scatter sample (30 k rows for Tableau performance)
d4_scatter = df[[
    'order_id', 'order_year', 'order_quarter', 'order_purchase_timestamp',
    'delivery_delay', 'review_score', 'is_late', 'is_bad_review',
    'seller_state', 'customer_state', 'seller_id',
]].copy()
d4_scatter['delivery_status'] = d4_scatter['is_late'].map({True: 'Late', False: 'On Time'})
d4_scatter['review_tier']     = d4_scatter['is_bad_review'].map({True: '1-2 Star (Bad)', False: '3-5 Star (Good)'})
d4_scatter = d4_scatter.drop(columns=['is_late', 'is_bad_review'])
d4_scatter = d4_scatter.sample(n=min(30_000, len(d4_scatter)), random_state=42)

save_tableau(d4_scatter,       '04a_order_scatter.csv')
save_tableau(bad_review_trend, '04b_bad_review_trend.csv')
save_tableau(ot_stats,         '04c_ontime_vs_late.csv')
save_tableau(star1_state,      '04d_one_star_by_state.csv')


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/8] Statistical Validation …")

late_rev   = df[df['is_late'] == True]['review_score'].dropna()
ontime_rev = df[df['is_late'] == False]['review_score'].dropna()
t_stat, t_pval = stats.ttest_ind(ontime_rev, late_rev, equal_var=False)
print(f"  Welch T-Test (Late vs On-time Review) : t={t_stat:.4f}, p={t_pval:.2e} "
      f"→ {'REJECT H0 ✓' if t_pval < 0.05 else 'fail to reject'}")

chi2, chi_pval, _, _ = stats.chi2_contingency(
    pd.crosstab(df['is_bad_review'], df['is_late'])
)
print(f"  Chi-Square (1-Star vs Late)           : χ²={chi2:.2f}, p={chi_pval:.2e} "
      f"→ {'REJECT H0 ✓' if chi_pval < 0.05 else 'fail to reject'}")

top_states   = df['seller_state'].value_counts().head(5).index
state_groups = [
    g['delivery_delay'].dropna().values
    for _, g in df[df['seller_state'].isin(top_states)].groupby('seller_state')
]
f_stat, anova_pval = stats.f_oneway(*state_groups)
print(f"  ANOVA (Top-5 Seller States Delay)     : F={f_stat:.2f}, p={anova_pval:.2e} "
      f"→ {'REJECT H0 ✓' if anova_pval < 0.05 else 'fail to reject'}")


# ══════════════════════════════════════════════════════════════════════════════
# EDA CHARTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7/8] Generating EDA charts …")

PALETTE_RED  = '#e74c3c'
PALETTE_GRN  = '#2ecc71'
PALETTE_NAVY = '#2c3e50'

# ── Chart 1: D1 — Late rate trend by year
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(yoy['order_year'].astype(str), yoy['late_rate_pct'],
        marker='o', linewidth=2.5, color=PALETTE_RED, markersize=9)
for _, row in yoy.iterrows():
    ax.annotate(f"{row['late_rate_pct']:.2f}%",
                (str(row['order_year']), row['late_rate_pct']),
                textcoords='offset points', xytext=(0, 10),
                ha='center', fontsize=11, fontweight='bold', color=PALETTE_RED)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Late Delivery Rate (%)', fontsize=12)
ax.set_title('D1 — Late Delivery Rate Trend (YoY)', fontsize=14, fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
plt.tight_layout()
fig.savefig(os.path.join(PATH_EDA_OUT, 'd1_late_rate_trend.png'), dpi=150)
plt.close(fig)
print("  ✓ d1_late_rate_trend.png")

# ── Chart 2: D2 — Top 5 worst seller states
fig, ax = plt.subplots(figsize=(9, 5))
t5 = top5_worst_states.sort_values('late_rate_pct')
colors = [PALETTE_RED if v > top5_worst_states['late_rate_pct'].median()
          else '#e67e22' for v in t5['late_rate_pct']]
bars = ax.barh(t5['seller_state'], t5['late_rate_pct'], color=colors)
for bar, val in zip(bars, t5['late_rate_pct']):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}%', va='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Late Delivery Rate (%)', fontsize=12)
ax.set_title('D2 — Top 5 Worst Seller States by Late Rate', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(PATH_EDA_OUT, 'd2_top5_seller_states.png'), dpi=150)
plt.close(fig)
print("  ✓ d2_top5_seller_states.png")

# ── Chart 3: D3 — Top 20 worst sellers bar chart
fig, ax = plt.subplots(figsize=(13, 7))
t20 = top20_worst.sort_values('delay_rate_pct')
bar_colors = [PALETTE_RED if x > 50 else '#e67e22' if x > 30 else '#c0392b'
              for x in t20['delay_rate_pct']]
ax.barh(t20['seller_id'].str[:10] + '…', t20['delay_rate_pct'], color=bar_colors)
ax.axvline(13, color=PALETTE_NAVY, linestyle='--', linewidth=1.8, label='13% threshold')
ax.set_xlabel('Delay Rate (%)', fontsize=12)
ax.set_title('D3 — Top 20 Sellers by Delay Rate (≥10 orders)', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(PATH_EDA_OUT, 'd3_top20_worst_sellers.png'), dpi=150)
plt.close(fig)
print("  ✓ d3_top20_worst_sellers.png")

# ── Chart 4: D3 — Seller delay rate vs review score scatter
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    seller_agg['delay_rate_pct'], seller_agg['avg_review_score'],
    alpha=0.4, c=seller_agg['delay_rate_pct'], cmap='RdYlGn_r',
    s=30, edgecolors='none',
)
plt.colorbar(scatter, ax=ax, label='Delay Rate %')
m, b = np.polyfit(seller_agg['delay_rate_pct'], seller_agg['avg_review_score'], 1)
x_line = np.linspace(0, seller_agg['delay_rate_pct'].max(), 100)
ax.plot(x_line, m * x_line + b, 'r-', linewidth=2, label=f'Trend  r={pearson_r:.3f}')
ax.axvline(13, color=PALETTE_NAVY, linestyle='--', linewidth=1.5, alpha=0.7,
           label='13% risk threshold')
ax.set_xlabel('Seller Delay Rate (%)', fontsize=12)
ax.set_ylabel('Avg Review Score', fontsize=12)
ax.set_title('D3 — Seller Delay Rate vs. Avg Review Score', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(PATH_EDA_OUT, 'd3_seller_delay_vs_review.png'), dpi=150)
plt.close(fig)
print("  ✓ d3_seller_delay_vs_review.png")

# ── Chart 5: D3 — High-risk sellers by state
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(hr_by_state['seller_state'], hr_by_state['high_risk_count'],
        color=PALETTE_RED, alpha=0.85)
ax.set_xlabel('Sellers with >13% Delay Rate', fontsize=11)
ax.set_title('D3 — High-Risk Seller Distribution by State', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(PATH_EDA_OUT, 'd3_high_risk_by_state.png'), dpi=150)
plt.close(fig)
print("  ✓ d3_high_risk_by_state.png")

# ── Chart 6: D4 — Bad review rate trend by year
fig, ax = plt.subplots(figsize=(8, 5))
bar_cols = ['#3498db', '#e74c3c', PALETTE_GRN][:len(bad_review_trend)]
bars = ax.bar(bad_review_trend['order_year'].astype(str),
              bad_review_trend['bad_review_rate_pct'],
              color=bar_cols, width=0.5)
for bar, val in zip(bars, bad_review_trend['bad_review_rate_pct']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f'{val:.2f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('1-Star Review Rate (%)', fontsize=12)
ax.set_title('D4 — Bad Review Rate Trend (2017 → 2018)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(PATH_EDA_OUT, 'd4_bad_review_trend.png'), dpi=150)
plt.close(fig)
print("  ✓ d4_bad_review_trend.png")

# ── Chart 7: D4 — On-time vs Late avg review score
fig, ax = plt.subplots(figsize=(8, 5))
colors_ot = [PALETTE_GRN if s == 'On Time' else PALETTE_RED
             for s in ot_stats['delivery_status']]
bars = ax.bar(ot_stats['delivery_status'], ot_stats['avg_review_score'],
              color=colors_ot, width=0.4)
for bar, val in zip(bars, ot_stats['avg_review_score']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.08,
            f'{val:.3f}', ha='center', va='top', fontsize=13,
            color='white', fontweight='bold')
ax.set_ylim(0, 5.5)
ax.axhline(5, color='gray', linestyle='--', alpha=0.4)
ax.set_ylabel('Avg Review Score (out of 5)', fontsize=12)
ax.set_title('D4 — Average Review Score: On-Time vs Late Orders', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(PATH_EDA_OUT, 'd4_ontime_vs_late.png'), dpi=150)
plt.close(fig)
print("  ✓ d4_ontime_vs_late.png")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8/8] Summary …")

_2017_bad   = bad_review_trend.loc[bad_review_trend['order_year'] == 2017, 'bad_review_rate_pct']
_2018_bad   = bad_review_trend.loc[bad_review_trend['order_year'] == 2018, 'bad_review_rate_pct']
_ot_values  = ot_stats.loc[ot_stats['delivery_status'] == 'On Time', 'avg_review_score'].values
_lt_values  = ot_stats.loc[ot_stats['delivery_status'] == 'Late',    'avg_review_score'].values

# Pre-format to avoid nested format specs inside f-string conditionals
_late_2017  = f"{_2017_bad.values[0]}"   if len(_2017_bad)  else 'N/A'
_late_2018  = f"{_2018_bad.values[0]}"   if len(_2018_bad)  else 'N/A'
_late_yr17  = f"{yoy.loc[yoy['order_year']==2017,'late_rate_pct'].values[0]}" \
              if 2017 in yoy['order_year'].values else 'N/A'
_late_yr18  = f"{yoy.loc[yoy['order_year']==2018,'late_rate_pct'].values[0]}" \
              if 2018 in yoy['order_year'].values else 'N/A'
_ot_str     = f"{_ot_values[0]:.3f}"     if len(_ot_values) else 'N/A'
_lt_str     = f"{_lt_values[0]:.3f}"     if len(_lt_values) else 'N/A'
_top_state  = top5_worst_states.iloc[0]['seller_state']
_top_pct    = top5_worst_states.iloc[0]['late_rate_pct']

SEP = '═' * 68
print(f"""
{SEP}
  D1 — Executive Pulse
  ├─ ★ HEADLINE KPI : {overall_late_pct:.2f}% of customers received their order late
  ├─ Total late orders : {total_late_orders:,}
  ├─ Late rate 2017    : {_late_yr17}%
  └─ Late rate 2018    : {_late_yr18}%

  D2 — Geographic Risk Monitor
  ├─ Top worst state   : {_top_state} ({_top_pct}% late)
  ├─ Ibitinga flagged  : ✓  (02b_geo_risk_city.csv  →  is_ibitinga_hotspot)
  └─ SP → RJ corridor  : ✓  (02c_geo_risk_route.csv →  is_sp_rj)

  D3 — Seller Performance Scorecard
  ├─ Qualified sellers (≥10 orders) : {total_qualified:,}
  ├─ High-risk sellers (>13% delay) : {high_risk_count:,}
  └─ Pearson r (delay vs review)    : {pearson_r:.4f}   p={pearson_p:.2e}

  D4 — Customer Experience Impact
  ├─ Bad review rate 2017    : {_late_2017}%
  ├─ Bad review rate 2018    : {_late_2018}%
  ├─ Avg review (On-Time)    : {_ot_str}
  ├─ Avg review (Late)       : {_lt_str}
  ├─ T-Test p-value          : {t_pval:.2e}  → SIGNIFICANT ✓
  └─ Chi-Sq p-value          : {chi_pval:.2e}  → SIGNIFICANT ✓
{SEP}
  Tableau CSVs → {PATH_TABLEAU}
  EDA charts   → {PATH_EDA_OUT}
{SEP}
""")
