import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load datasets
order_items = pd.read_csv('olist_order_items_dataset.csv')
orders = pd.read_csv('olist_orders_dataset.csv')
products = pd.read_csv('olist_products_dataset.csv')
categories = pd.read_csv('product_category_name_translation.csv')

# Merge datasets
merged = pd.merge(order_items, orders, on='order_id')
merged = pd.merge(merged, products, on='product_id')
merged = pd.merge(merged, categories, on='product_category_name', how='left')
merged['order_purchase_timestamp'] = pd.to_datetime(merged['order_purchase_timestamp'])
merged['month'] = merged['order_purchase_timestamp'].dt.to_period('M')
monthly_sales = merged.groupby(['month', 'product_category_name_english'])['price'].sum().reset_index()
monthly_sales.rename(columns={'price': 'sales'}, inplace=True)
monthly_sales.to_csv('monthly_sales.csv', index=False)



def forecast_sales_linear_regression(category, data, forecast_steps=6):
    """
    Forecast sales for a specific product category for the next 6 months.
    """
    # Filter data for the specific category
    category_data = data[data['product_category_name_english'] == category]
    category_data = category_data.set_index('month')['sales']

    # Ensure no missing values and convert PeriodIndex to DatetimeIndex
    category_data = category_data.asfreq('M').fillna(0)
    category_data.index = category_data.index.to_timestamp()

    # Prepare data for regression
    X = np.arange(len(category_data)).reshape(-1, 1)  # Months as numeric values
    y = category_data.values

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Forecast future sales
    future_X = np.arange(len(category_data), len(category_data) + forecast_steps).reshape(-1, 1)
    forecast = model.predict(future_X)

    # Create dates for the forecast
    forecast_dates = pd.date_range(
        start=category_data.index[-1] + pd.DateOffset(months=1),
        periods=forecast_steps,
        freq='M'
    )

    # Combine forecast with dates
    forecast_df = pd.DataFrame({
        'month': forecast_dates,
        'predicted_sales': forecast,
        'product_category_name_english': category
    })

    return forecast_df

# Get unique product categories
categories = monthly_sales['product_category_name_english'].unique()

# Create an empty list to store forecasts
all_forecasts = []

# Loop through each category and apply the forecasting function
for category in categories:
    category_forecast = forecast_sales_linear_regression(category, monthly_sales)
    all_forecasts.append(category_forecast)

# Combine all forecasts into a single DataFrame
all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)

print(all_forecasts_df.head())
all_forecasts_df.to_csv('all_categories_forecast_next_6_months.csv', index=False)
