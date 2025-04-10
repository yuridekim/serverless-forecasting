# FuncProphet: Smarter Prewarming for Serverless Platforms

FuncProphet is an end-to-end serverless infrastructure enhancement that reduces cold-start latency and boosts instance efficiency by combining lightweight time-series forecasting with tunable prewarming policies.

## 🔍 Overview

Modern Function-as-a-Service (FaaS) platforms struggle with the performance–efficiency tradeoff. Cold starts increase tail latency, while idle containers waste resources. FuncProphet tackles both with a novel approach: per-function instance scheduling driven by demand forecasting.

- **Seasonality-aware predictions** via Facebook's Prophet model
- **Granular prewarm policy engine** for dynamic tuning
- **Custom serverless runtime** with metrics collection and trace replay support

## 💡 Motivation

Most serverless platforms apply fixed prewarming heuristics, leading to inefficient resource usage. FuncProphet enables smarter decisions by using actual invocation trends to forecast demand and prewarm only when necessary.

## 📈 Results

- Up to **2× increase in instance duty cycle**
- **Significant reduction in tail latency** for high-traffic functions
- Realistic simulation using Azure serverless traces (2021)

## ⚙️ Architecture

1. **Data Ingestion**: Parse historical traces and monitor live usage
2. **Forecasting Engine**: Time-series modeling using Prophet
3. **Policy Executor**: Applies custom warming/cooling rules to the platform
4. **Metrics Module**: Collects and reports end-to-end latency, utilization, and cold start rates
