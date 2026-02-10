# Data Dictionary - Inventory Management Dataset

## Overview

This dataset represents a retail inventory management system with sales transactions across multiple stores and product categories. The data follows a star schema with one fact table and four dimension tables.

---

## 1. Fact Table: `fact_sales.csv`

**Description:** Daily sales transactions with inventory levels

| Column | Data Type | Description | Notes |
|--------|-----------|-------------|-------|
| `transaction_id` | String | Unique transaction identifier | Format: T###### |
| `date_id` | String | Transaction date | **MESSY:** Multiple date formats exist |
| `product_id` | String | Product identifier (FK) | **MESSY:** Some have inconsistent casing |
| `store_id` | String | Store identifier (FK) | **MESSY:** Some have inconsistent casing |
| `quantity_sold` | Integer | Number of units sold | **MESSY:** Has missing values and outliers |
| `unit_price` | Float | Price per unit at time of sale | **MESSY:** Some missing values |
| `gross_amount` | Float | Total before discount | quantity_sold * unit_price |
| `discount_percentage` | Integer | Discount applied (0-30%) | 0 = no discount |
| `discount_amount` | Float | Discount value in dollars | |
| `net_amount` | Float | Final transaction amount | gross_amount - discount_amount |
| `is_promotion` | String | Whether item was on promotion | Values: 'Yes', 'No' |
| `current_stock_level` | Integer | Inventory level at transaction time | **MESSY:** Some missing values |
| `reorder_point` | Integer | Minimum stock before reorder | |
| `units_to_stock` | Integer | **TARGET VARIABLE** - Recommended units to stock | |
| `stockout_flag` | Integer | Whether stockout occurred | 1 = Yes, 0 = No |

**Record Count:** ~47,000 transactions
**Known Issues:**
- ~0.5% duplicate transactions
- ~1% outliers in quantity_sold (500-1000 units)
- ~2-5% missing values in various columns
- Inconsistent date formats (YYYY-MM-DD, MM/DD/YYYY, etc.)

---

## 2. Dimension: `dim_products.csv`

**Description:** Product master data

| Column | Data Type | Description | Notes |
|--------|-----------|-------------|-------|
| `product_id` | String | Unique product identifier | Format: P#### |
| `product_name` | String | Product name | |
| `category` | String | Product category | **MESSY:** Typos and case variations |
| `subcategory` | String | Product subcategory | **MESSY:** Some missing values |
| `unit_price` | Float | Standard retail price | **MESSY:** Some missing values |
| `cost_price` | Float | Cost to purchase | |
| `supplier_id` | String | Supplier identifier (FK) | Format: S### |
| `shelf_life_days` | Integer | Days until expiration | Only for perishables |
| `weight_kg` | Float | Product weight | **MESSY:** Some missing values |
| `is_perishable` | String | Whether product expires | 'Yes' or 'No' |

**Categories:** Electronics, Clothing, Groceries, Home & Kitchen, Beauty

**Record Count:** 210 (200 unique + ~10 duplicates)

**Known Issues:**
- Category typos: "Electroncs", "Clothng", "Grocereis", "Beuty"
- Case variations: "ELECTRONICS", "electronics", "Electronics"
- ~10 duplicate product records
- ~5% missing subcategories

---

## 3. Dimension: `dim_stores.csv`

**Description:** Store/location information

| Column | Data Type | Description | Notes |
|--------|-----------|-------------|-------|
| `store_id` | String | Unique store identifier | Format: ST### |
| `store_name` | String | Store name | |
| `store_type` | String | Store format | Supermarket, Express, Hypermarket, Warehouse |
| `city` | String | City location | |
| `region` | String | Geographic region | **MESSY:** Inconsistent naming |
| `store_size_sqft` | Integer | Store area in sq feet | **MESSY:** Some missing values |
| `opening_date` | String | Store opening date | **MESSY:** Some missing values |
| `manager_id` | String | Manager identifier | Format: M#### |
| `num_employees` | Integer | Employee count | **MESSY:** Some missing values |
| `parking_capacity` | Integer | Parking spaces | |

**Regions:** North, South, East, West, Central

**Record Count:** 52 (50 unique + ~3 duplicates)

**Known Issues:**
- Region variations: "NORTH", "north", "Northern", "N"
- ~3 duplicate store records
- ~5% missing values in store_type, store_size_sqft

---

## 4. Dimension: `dim_dates.csv`

**Description:** Calendar dimension with temporal attributes

| Column | Data Type | Description | Notes |
|--------|-----------|-------------|-------|
| `date_id` | String | Date in YYYY-MM-DD format | Primary key |
| `date_display` | String | Display format | **MESSY:** Multiple formats |
| `year` | Integer | Year (2023-2024) | |
| `month` | Integer | Month (1-12) | |
| `month_name` | String | Month name | **MESSY:** Full vs abbreviated |
| `day` | Integer | Day of month (1-31) | |
| `day_of_week` | Integer | Day number (0=Monday, 6=Sunday) | |
| `day_name` | String | Day name | Monday, Tuesday, etc. |
| `week_of_year` | Integer | Week number (1-52) | |
| `quarter` | String/Integer | Quarter | **MESSY:** 'Q1' vs 1 |
| `season` | String | Season | Winter, Spring, Summer, Fall |
| `is_weekend` | Integer | Weekend flag | 1 = Yes, 0 = No |
| `is_holiday` | Integer | US Federal Holiday | 1 = Yes, 0 = No |
| `is_month_end` | Integer | Last day of month | 1 = Yes, 0 = No |

**Date Range:** January 1, 2023 - December 31, 2024 (731 days)

**Known Issues:**
- date_display has mixed formats (YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY)
- quarter column has mixed types ('Q1' vs 1)
- month_name has full names vs abbreviations

---

## 5. Dimension: `dim_suppliers.csv`

**Description:** Supplier/vendor information

| Column | Data Type | Description | Notes |
|--------|-----------|-------------|-------|
| `supplier_id` | String | Unique supplier identifier | Format: S### |
| `supplier_name` | String | Company name | **MESSY:** Some in UPPERCASE |
| `contact_email` | String | Primary contact email | **MESSY:** Some missing |
| `phone` | String | Phone number | **MESSY:** Some show 'N/A' |
| `country` | String | Supplier country | USA, China, India, etc. |
| `lead_time_days` | Integer | Days to receive order | **MESSY:** Some missing |
| `reliability_score` | Float | Performance rating (3.0-5.0) | **MESSY:** Some missing |
| `min_order_quantity` | Integer | Minimum order size | |
| `payment_terms` | String | Payment conditions | Net 30, Net 60, COD, etc. |
| `active_status` | String | Supplier status | **MESSY:** Multiple formats |

**Record Count:** 20 suppliers

**Known Issues:**
- active_status has many variations: 'Active', 'active', 'ACTIVE', 'Yes', '1'
- Phone shows 'N/A' for missing values
- ~10% missing values in contact_email, lead_time_days, reliability_score

---

## Relationships (Star Schema)

```
                    ┌─────────────────┐
                    │  dim_dates      │
                    │  (date_id)      │
                    └────────┬────────┘
                             │
┌─────────────────┐          │          ┌─────────────────┐
│  dim_products   │──────────┼──────────│  dim_stores     │
│  (product_id)   │          │          │  (store_id)     │
└────────┬────────┘          │          └─────────────────┘
         │                   │
         │      ┌────────────┴────────────┐
         │      │                         │
         └──────►      fact_sales         │
                │                         │
         ┌──────►  (transaction_id)       │
         │      │                         │
         │      └─────────────────────────┘
         │
┌────────┴────────┐
│  dim_suppliers  │
│  (supplier_id)  │
└─────────────────┘
```

**Join Keys:**
- fact_sales.product_id → dim_products.product_id
- fact_sales.store_id → dim_stores.store_id
- fact_sales.date_id → dim_dates.date_id
- dim_products.supplier_id → dim_suppliers.supplier_id

---

## Target Variable

**`units_to_stock`** - The recommended number of units to stock for optimal inventory management.

This variable is calculated based on:
- Current stock levels
- Reorder points
- Historical sales patterns
- Buffer for variability

**Goal:** Build a model to predict this value to minimize stockouts and overstock costs.

---

## Recommended Cleaning Steps

1. **Standardize categorical values** (convert to consistent case, fix typos)
2. **Handle missing values** (imputation or removal based on context)
3. **Fix date formats** (convert all to YYYY-MM-DD)
4. **Remove duplicates** (identify and remove based on primary keys)
5. **Handle outliers** (investigate quantity_sold > 100)
6. **Standardize key columns** (ensure FK values match PK values for joins)
7. **Convert data types** (ensure proper numeric/string/date types)
