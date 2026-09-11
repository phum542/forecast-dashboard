import pandas as pd
import numpy as np
import json
import os

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
INPUT_DIR = r"C:\Users\Phumrat\Downloads\Interactive Dashboard Testing"
MASTER_FILE = os.path.join(INPUT_DIR, "kpi_enriched_with_simulated_dimensions.csv")
OUTPUT_JSON = os.path.join(INPUT_DIR, "dashboard_data.json")

MONTHS_TO_KEEP = 36
SEG_COLS = {
    "Simulated_Enterprise_MRR": "enterprise",
    "Simulated_MidMarket_MRR": "midmarket",
    "Simulated_SMB_MRR": "smb"
}

# ---------------------------------------------------------
# STEP 1: โหลดไฟล์ Master + Sort ตามวันที่
# ---------------------------------------------------------
df = pd.read_csv(MASTER_FILE, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)
df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)

# ---------------------------------------------------------
# STEP 2: Aggregate เป็นรายเดือน
# ---------------------------------------------------------
monthly = df.groupby("YearMonth").agg(
    Actual_MRR=("Actual_MRR", "mean"),
    Target_MRR=("Target_MRR", "mean"),
    Churn_Rate_Pct=("Churn_Rate_Pct", "mean"),
    CAC=("CAC", "mean"),
    ARPU=("ARPU", "mean")
).tail(MONTHS_TO_KEEP).reset_index()

# ---------------------------------------------------------
# STEP 3: Forecast 3 กรณี (จาก rollback1)
# ---------------------------------------------------------
last13 = monthly["Actual_MRR"].tail(13).values
growth_rates = [(last13[i] - last13[i-1]) / last13[i-1] for i in range(1, len(last13))]

worst_growth, base_growth, best_growth = min(growth_rates), np.mean(growth_rates), max(growth_rates)
latest_actual = monthly["Actual_MRR"].iloc[-1]
latest_target_growth = (monthly["Target_MRR"].iloc[-1] - monthly["Target_MRR"].iloc[-2]) / monthly["Target_MRR"].iloc[-2]

forecast = {
    "worst_case": round(latest_actual * (1 + worst_growth), 2),
    "base_case": round(latest_actual * (1 + base_growth), 2),
    "best_case": round(latest_actual * (1 + best_growth), 2),
    "next_target": round(monthly["Target_MRR"].iloc[-1] * (1 + latest_target_growth), 2),
    "worst_growth_pct": round(worst_growth * 100, 2),
    "base_growth_pct": round(base_growth * 100, 2),
    "best_growth_pct": round(best_growth * 100, 2)
}

# ---------------------------------------------------------
# STEP 4: Segment Growth Rate (จากเวอร์ชันใหม่)
# ---------------------------------------------------------
seg_growth = {}
for col, name in SEG_COLS.items():
    df_seg = df.groupby("YearMonth")[col].mean()
    growth = df_seg.pct_change().tail(12).mean()
    seg_growth[name] = round(growth * 100, 2)

# ---------------------------------------------------------
# STEP 5: Segment Snapshot + Dynamic Stats by Region
# ---------------------------------------------------------
# Snapshot ล่าสุด (จาก rollback1) — ใช้ค่าเฉลี่ยเดือนล่าสุด เพื่อความสม่ำเสมอ
last_month = monthly["YearMonth"].iloc[-1]
df_last_month = df[df["YearMonth"] == last_month]
segment_latest = {
    name: round(df_last_month[col].mean(), 0)
    for col, name in SEG_COLS.items()
}

# What-If Dynamic Stats แยก Region (จากเวอร์ชันใหม่)
seg_stats = df.groupby("Simulated_Region").agg(
    Churn_Rate_Pct=("Churn_Rate_Pct", "mean"),
    CAC=("CAC", "mean")
).round(2).to_dict(orient="index")

# ---------------------------------------------------------
# STEP 6: Region Share — คำนวณจากข้อมูลจริง (แก้ hardcode!)
# ---------------------------------------------------------
region_counts = df["Simulated_Region"].value_counts(normalize=True)
region_share = {
    "all": 1.0,
    "NA": round(region_counts.get("North America", 0), 3),
    "EU": round(region_counts.get("Europe", 0), 3),
    "APAC": round(region_counts.get("APAC", 0), 3),
    "Other": round(region_counts.get("Other", 0), 3)
}

# ---------------------------------------------------------
# STEP 7: Anomaly Detection (จาก rollback1)
# ---------------------------------------------------------
daily_change = df["Actual_MRR"].diff()
z = (daily_change - daily_change.mean()) / daily_change.std()
mask = z.abs() > 2.5
anomalies_df = df[mask][["Date", "Actual_MRR"]].copy()
anomalies_df["Change"] = daily_change[mask]
anomalies = [{"date": r["Date"].strftime("%Y-%m-%d"),
              "mrr": round(r["Actual_MRR"], 0),
              "change": round(r["Change"], 0)}
             for _, r in anomalies_df.sort_values("Date", ascending=False).head(8).iterrows()]

# ---------------------------------------------------------
# STEP 8: Save JSON
# ---------------------------------------------------------
output = {
    "meta": {
        "generated_from_file": "kpi_enriched_with_simulated_dimensions.csv",
        "date_range_end": df["Date"].max().strftime("%Y-%m-%d")
    },
    "months": monthly["YearMonth"].tolist(),
    "actual_mrr": monthly["Actual_MRR"].round(0).tolist(),
    "target_mrr": monthly["Target_MRR"].round(0).tolist(),
    "churn_rate": monthly["Churn_Rate_Pct"].round(2).tolist(),
    "cac": monthly["CAC"].round(1).tolist(),
    "arpu": monthly["ARPU"].round(2).tolist(),
    "forecast": forecast,
    "seg_growth": seg_growth,
    "seg_stats": seg_stats,
    "segment_latest": segment_latest,
    "anomalies": anomalies,
    "region_share": region_share
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Success! Merged pipeline → {OUTPUT_JSON}")
