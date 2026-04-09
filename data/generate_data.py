"""
Generate 3 years of monthly sales data with trend, seasonality and noise.
Run once: python3 generate_data.py
"""

import csv
import math
import random
from pathlib import Path

random.seed(42)

months = []
for year in range(2022, 2025):
    for month in range(1, 13):
        months.append(f"{year}-{month:02d}")

rows = []
for i, month in enumerate(months):
    trend = 50000 + i * 800
    # seasonal peak in Nov-Dec, dip in Jan-Feb
    m = int(month.split("-")[1])
    seasonal = 8000 * math.sin((m - 3) * math.pi / 6)
    noise = random.gauss(0, 2000)
    revenue = round(max(trend + seasonal + noise, 10000), 2)
    orders = int(revenue / random.uniform(180, 220))
    rows.append({"month": month, "revenue": revenue, "orders": orders})

out = Path(__file__).parent / "monthly_sales.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["month", "revenue", "orders"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {out}")
