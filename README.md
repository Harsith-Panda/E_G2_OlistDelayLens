# NST DVA Capstone 2 - OlistDelayLens
**Newton School of Technology | Data Visualization & Analytics**  
A 2-week industry simulation capstone using Python, GitHub, and Tableau to convert raw data into actionable business intelligence.

## Project Overview
| Field | Details |
| :--- | :--- |
| **Project Title** | OlistDelayLens: Analyzing Logistics and Delivery Delays |
| **Sector** | E-Commerce / Logistics |
| **Team ID** | E-G2 |
| **Section** | Group 2 |
| **Faculty Mentor** | Aayushi Vashisht, Satyaki Das |
| **Institute** | Newton School of Technology |
| **Submission Date** | April 29, 2026 |

## Team Members
| Role | Name | GitHub Username |
| :--- | :--- | :--- |
| **Project Lead** | Harsith | @Harsith-Panda |
| **Data Lead** | Harinder | @Harenderchhoker31 |
| **ETL Lead** | Harsith | @Harsith-Panda |
| **Analysis Lead** | Aradhya | @AradhyaTiwari10 |
| **Visualization Lead** | Khushi / Ishan | @Khushi-jain05 / @IshanMaheshwari-777 |
| **Strategy Lead** | Khushi | @Khushi-jain05 |
| **PPT and Quality Lead** | Aradhya / Praanshu |  @AradhyaTiwari10 |

## Business Problem
Olist, a major marketplace integrator in Brazil, operates in a high-growth but logistically complex environment. The core challenge is the "Delivery Gap"—the variance between promised delivery dates and actual arrival times—which is the primary driver of negative customer reviews and brand erosion. This project serves the **Chief Operations Officer (COO)** and **Logistics Managers** by identifying structural bottlenecks in the seller-to-carrier handover and quantifying the "Satisfaction Cost" of these delays to drive operational reform.

### Core Business Question
What are the primary drivers of delivery latency within the Olist ecosystem, and how can Olist optimize its geographic ETA estimations to protect customer satisfaction?

### Decision Supported
This analysis enables the implementation of **Seller Performance SLAs** (specifically dispatch windows), the decentralization of warehousing through **Regional Hubs**, and the deployment of **Dynamic ETA Buffers** for high-risk delivery routes.

## Dataset
| Attribute | Details |
| :--- | :--- |
| **Source Name** | Kaggle (Brazilian E-Commerce Public Dataset by Olist) |
| **Direct Access Link** | [Kaggle Dataset Link](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) |
| **Row Count** | 109,126 (Delivered Orders) |
| **Column Count** | 46 features after processing |
| **Time Period Covered** | Sept 2016 to Sept 2018 |
| **Format** | CSV (Relational) |

### Key Columns Used
| Column Name | Description | Role in Analysis |
| :--- | :--- | :--- |
| `order_delivered_customer_date` | Actual delivery timestamp | Calculating latency/lateness |
| `order_estimated_delivery_date` | Promised delivery timestamp | Benchmarking performance (ETA) |
| `haversine_distance_km` | Calculated distance (Seller to Customer) | Control variable for transit complexity |
| `review_score` | Satisfaction rating (1-5) | Primary outcome variable for impact |

For full column definitions, see `docs/data_dictionary.md`.

## KPI Framework
| KPI | Definition | Formula / Computation |
| :--- | :--- | :--- |
| **Late Delivery Rate (%)** | Overall reliability of the network | `(Count of Late Orders / Total Orders) * 100` |
| **Delivery Delay (Days)** | Magnitude of ETA failure | `Actual Delivery Date - Estimated Delivery Date` |
| **Dispatch Latency** | Seller-side performance | `Carrier Handover Date - Order Approval Date` |
| **Satisfaction Gap** | Impact of delay on NPS | `Avg Review Score (On-Time) - Avg Review Score (Late)` |

## Tableau Dashboard
| Item | Details |
| :--- | :--- |
| **Dashboard URL** | [Tableau Public Link]([https://public.tableau.com/shared/475TX9DDD?:display_count=n&:origin=viz_share_link]) |
| **Executive View** | High-level summary of Late Rate trends (YoY) and Regional Risk Index. |
| **Operational View** | Route-level scatter plots (Distance vs. Delay) and Seller Risk Scorecards. |
| **Main Filters** | Year, Quarter, Seller State, Customer State, Product Category. |

## Key Insights
1.  **Systemic Growth Pain:** The late delivery rate rose from 5.6% in 2017 to 8.1% in 2018, indicating logistics cannot scale with sales volume.
2.  **The Satisfaction Cliff:** On-time orders average a 4.2 score, while late orders drop to 2.5, a massive loss in brand equity.
3.  **The 48-Hour Critical Window:** Sellers who fail to hand over packages to carriers within 48 hours account for 60% of all late deliveries.
4.  **Optimism Bias:** ETAs for North and Northeast routes are underestimated by 3.5 days on average.
5.  **Distance vs. Operations:** Geographic distance explains only 45% of delay; the rest is driven by operational handovers and carrier efficiency.
6.  **Black Friday Spike:** Holiday season volumes trigger a 3x increase in late deliveries due to carrier capacity limits.
7.  **Regional Hotspots:** Maranhão (MA) and Alagoas (AL) have the highest late rates, often exceeding 25% for specific routes.
8.  **Weight Sensitivity:** Heavier furniture and office items are 15% more likely to be delayed than standard electronics.

## Recommendations
| # | Insight | Recommendation | Expected Impact |
| :--- | :--- | :--- | :--- |
| 1 | 48-Hour Handover Rule | Implement a **Seller Dispatch SLA** with penalties for >48h latency. | 15% Reduction in total delays. |
| 2 | Optimism Bias | Deploy **Dynamic ETA Buffers** (+3 days) for North-Northeast routes. | 30% Improvement in ETA Accuracy. |
| 3 | Regional Hotspots | Establish **Regional Consolidation Hubs** in high-risk states. | 2-day reduction in total lead time. |
| 4 | Satisfaction Cliff | Automate **Proactive Delay Alerts** to customers before the ETA expires. | 0.5 point increase in "Late" review scores. |

## Analytical Pipeline
1.  **Define** - Identified the "Logistics-Satisfaction" gap as the core problem.
2.  **Extract** - Sourced 9 relational tables from Olist Kaggle dataset.
3.  **Clean** - Built a Python pipeline in `02_cleaning.ipynb` for timestamp standardization and Haversine distance.
4.  **Analyze** - Conducted T-Tests, ANOVA, and Chi-Square tests in `04_statistical_analysis.ipynb`.
5.  **Visualize** - Developed a 4-dashboard suite in Tableau for Executive and Operational tracking.
6.  **Recommend** - Drafted 4 data-backed strategies for seller and route optimization.
7.  **Report** - Compiled a 15-page LaTeX report in `reports/`.

## Tech Stack
| Tool | Purpose |
| :--- | :--- |
| **Python** | ETL, Haversine Distance, Outlier Detection |
| **SciPy/StatsModels** | T-Tests and ANOVA for Hypothesis Testing |
| **Tableau Public** | Visual Analytics and Geographic Mapping |
| **GitHub** | Version Control and Project Documentation |

## Contribution Matrix
| Team Member | Dataset & Sourcing | ETL & Cleaning | EDA & Analysis | Statistical Analysis | Tableau Dashboard | Report Writing |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Harsith** | Owner | Owner | Support | - | support | Support |
| **Harinder** | Owner | Support | - | - | - | - |
| **Aradhya** | - | support | Owner | Owner | Support | Owner |
| **Ishan** | - | - | Support | - | Owner | - |
| **Khushi** | - | - | Support | - | Owner | Support |
| **Praanshu** | - | - | - | - | support | Owner |

---
**Newton School of Technology - Data Visualization & Analytics | Capstone 2**
