from pathlib import Path
from typing import Callable

import base64
from io import BytesIO
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image

from image_validation import (
    ACCEPTANCE_THRESHOLD,
    CLASSIFIER_MODEL_PATH,
    calculate_similarity,
    create_classifier_head,
    create_dataset_fingerprint,
    create_feature_extractor,
    extract_feature_map,
    find_reference_images,
    load_or_create_reference_embeddings,
    normalize_embedding,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / "Turbofan_Engine_Degradation_Dataset.csv"
ANN_MODEL_PATH = BASE_DIR / "models" / "ann_best.keras"
ANN_PREPROCESSING_PATH = BASE_DIR / "models" / "ann_preprocessing.joblib"
IMAGE_MODEL_PATH = CLASSIFIER_MODEL_PATH
BACKGROUND_IMAGE_PATH = BASE_DIR / "Data" / "BG.webp"
CASTING_TRAIN_PATH = BASE_DIR / "Data" / "Image" / "train"
CASTING_VALIDATION_PATH = BASE_DIR / "Data" / "Image" / "validation"

st.set_page_config(
    page_title="Ferro Vision | Casting Defect Detection",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if not BACKGROUND_IMAGE_PATH.is_file():
    raise FileNotFoundError(f"Background image not found: {BACKGROUND_IMAGE_PATH}")

background_image = base64.b64encode(BACKGROUND_IMAGE_PATH.read_bytes()).decode("ascii")
st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"] {{
        display: none;
    }}

    [data-testid="stAppViewContainer"] {{
        background-image:
            linear-gradient(rgba(10, 18, 30, 0.74), rgba(10, 18, 30, 0.80)),
            url("data:image/webp;base64,{background_image}");
        background-position: center;
        background-size: cover;
        background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
        background: rgba(10, 18, 30, 0.25);
    }}

    .stApp {{
        --primary-color: #27c3d5;
        font-family: "Segoe UI", "Inter", sans-serif;
    }}

    [data-testid="stMainBlockContainer"] {{
        color: #ffffff !important;
        background: linear-gradient(145deg, rgba(7, 20, 39, 0.96), rgba(11, 33, 59, 0.92));
        border: 1px solid rgba(106, 188, 216, 0.20);
        border-radius: 1.25rem;
        padding: 1rem clamp(1rem, 3vw, 2.5rem) 2rem;
        margin-top: 0.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.34);
        backdrop-filter: blur(18px);
    }}

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp p,
    .stApp label,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stCaptionContainer"] {{
        color: #ffffff !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.8);
    }}

    .ferro-brand-header {{
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin: 0;
        padding: 0.2rem 0.4rem;
    }}

    .ferro-brand-mark {{
        display: grid;
        width: 2.65rem;
        height: 2.65rem;
        place-items: center;
        border-radius: 0.8rem;
        color: #061521;
        background: linear-gradient(135deg, #5de3ed, #22b7d0);
        font-size: 1.4rem;
        box-shadow: 0 0 22px rgba(39, 195, 213, 0.32);
    }}

    .ferro-brand-name {{
        color: #f4fbff;
        font-size: 1.2rem;
        font-weight: 750;
        letter-spacing: 0.02em;
        line-height: 1.15;
        white-space: nowrap;
    }}

    .ferro-brand-subtitle {{
        margin-top: 0.18rem;
        color: #91b5c9;
        font-size: 0.76rem;
        letter-spacing: 0.01em;
        white-space: nowrap;
    }}

    .st-key-app_navigation {{
        position: sticky;
        top: 0;
        z-index: 1000;
        margin: -0.45rem -0.5rem 0.65rem;
        padding: 0.5rem 0.7rem;
        border: 1px solid rgba(90, 210, 230, 0.22);
        border-radius: 0.95rem;
        background: rgba(5, 18, 36, 0.94);
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.26);
        backdrop-filter: blur(18px);
    }}

    .stApp .st-key-app_navigation [data-testid="stBaseButton-primary"],
    .stApp .st-key-app_navigation [data-testid="stBaseButton-secondary"] {{
        min-height: 2.65rem;
        padding: 0.4rem 0.55rem;
        border: 0 !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0.3rem !important;
        color: #d7e4f0 !important;
        background: transparent !important;
        box-shadow: none !important;
        font-size: 0.85rem;
        font-weight: 600 !important;
        transform: none !important;
    }}

    .stApp .st-key-app_navigation [data-testid="stBaseButton-primary"] {{
        border-bottom-color: #3ca4f2 !important;
        color: #ffffff !important;
    }}

    .stApp .st-key-app_navigation [data-testid="stBaseButton-primary"]:hover,
    .stApp .st-key-app_navigation [data-testid="stBaseButton-secondary"]:hover {{
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.07) !important;
    }}

    .image-page-heading {{
        margin: 0.25rem 0 0;
        color: #f4fbff;
        font-size: clamp(1.7rem, 3vw, 2.45rem);
        font-weight: 750;
        letter-spacing: -0.03em;
    }}

    .image-page-description {{
        margin: 0.45rem 0 1.25rem;
        color: #a9c2d3;
        font-size: 1rem;
        line-height: 1.6;
    }}

    .section-eyebrow {{
        margin: 0 0 0.3rem;
        color: #58d5e5;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        border: 1px solid rgba(115, 190, 219, 0.23) !important;
        border-radius: 1.1rem !important;
        background: linear-gradient(145deg, rgba(19, 43, 68, 0.78), rgba(9, 26, 47, 0.76)) !important;
        box-shadow: 0 16px 38px rgba(0, 0, 0, 0.24);
        backdrop-filter: blur(16px);
    }}

    .stApp [data-testid="stButton"] button[kind="secondary"] {{
        color: #f8fafc;
        background: rgba(15, 23, 42, 0.9);
        border-color: rgba(148, 163, 184, 0.45);
    }}

    .stApp button[kind="primary"],
    .stApp [data-testid="stBaseButton-primary"] {{
        min-height: 3rem;
        border: 1px solid #35cde0 !important;
        border-radius: 0.8rem !important;
        color: #05202b !important;
        background: linear-gradient(135deg, #60e1e9, #25bbd2) !important;
        font-weight: 750 !important;
        box-shadow: 0 8px 22px rgba(39, 195, 213, 0.2);
        transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
    }}

    .stApp button[kind="primary"]:hover,
    .stApp [data-testid="stBaseButton-primary"]:hover {{
        filter: brightness(1.08);
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(39, 195, 213, 0.34);
    }}

    .stApp [data-testid="stButton"] button[kind="primary"]:hover {{
        filter: none;
    }}

    .stApp [data-testid="stFileUploader"] {{
        border: 1px dashed rgba(92, 207, 226, 0.52);
        border-radius: 0.9rem;
        background: rgba(5, 19, 36, 0.45);
        padding: 0;
        overflow: hidden;
    }}

    .stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {{
        border: 0 !important;
        background: linear-gradient(145deg, rgba(17, 43, 68, 0.98), rgba(8, 29, 51, 0.98)) !important;
        color: #e8f7ff !important;
        box-sizing: border-box;
        min-height: 3rem;
        height: 3rem;
        padding: 0.25rem 0.6rem;
        display: flex;
        align-items: center;
    }}

    .stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] > div,
    .stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] section {{
        background-color: transparent !important;
        color: #e8f7ff !important;
    }}

    .stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] * {{
        color: #e8f7ff !important;
    }}

    .stApp [data-testid="stFileUploader"] small,
    .stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
    .stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderFileData"] {{
        color: #d7eaf5 !important;
        opacity: 1 !important;
    }}

    .stApp [data-testid="stFileUploader"] svg {{
        color: #68d8e6 !important;
        fill: currentColor;
    }}

    .stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button {{
        color: #eafcff !important;
        background: linear-gradient(135deg, #164c70, #087b95) !important;
        border: 1px solid rgba(86, 214, 231, 0.58) !important;
        border-radius: 0.55rem !important;
    }}

    .stApp [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button:hover {{
        color: #ffffff !important;
        background: linear-gradient(135deg, #1a577d, #0989a4) !important;
    }}

    .stApp input,
    .stApp textarea,
    .stApp [data-baseweb="select"] > div {{
        color: #f8fafc;
        background-color: rgba(15, 23, 42, 0.92);
    }}

    .stApp input[type="radio"],
    .stApp input[type="checkbox"],
    .stApp input[type="range"] {{
        accent-color: #27c3d5;
    }}

    .stApp [data-baseweb="slider"] [role="slider"] {{
        background-color: #27c3d5 !important;
        border-color: #27c3d5 !important;
    }}

    .stApp [data-testid="stProgressBar"] > div {{
        background-color: #27c3d5 !important;
    }}

    [data-testid="stImage"] img {{
        max-height: 34rem;
        object-fit: contain;
        border-radius: 0.85rem;
    }}

    .inspection-meta {{
        margin-top: 0.9rem;
        padding: 0.75rem 0.9rem;
        border: 1px solid rgba(115, 190, 219, 0.16);
        border-radius: 0.75rem;
        color: #b7cddd;
        background: rgba(4, 17, 33, 0.52);
        font-size: 0.88rem;
        overflow-wrap: anywhere;
    }}

    .inspection-result {{
        margin-top: 1rem;
        padding: 1.25rem;
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 1rem;
        box-shadow: 0 14px 32px rgba(0, 0, 0, 0.24);
        color: #fff;
    }}

    .inspection-result.normal {{
        background: linear-gradient(135deg, rgba(13, 118, 89, 0.94), rgba(17, 75, 74, 0.96));
    }}

    .inspection-result.defective {{
        background: linear-gradient(135deg, rgba(174, 47, 58, 0.96), rgba(91, 32, 54, 0.96));
    }}

    .inspection-result.invalid {{
        background: linear-gradient(135deg, rgba(161, 86, 21, 0.96), rgba(91, 53, 26, 0.96));
    }}

    .st-key-dashboard_page {{
        margin: 0 -1rem;
        padding: clamp(0.75rem, 2vw, 1.5rem);
        border: 1px solid #e3ebf3;
        border-radius: 1.15rem;
        color: #142a47;
        background: linear-gradient(145deg, #f8fafc, #edf3f8);
        box-shadow: 0 16px 42px rgba(0, 0, 0, 0.2);
    }}

    .st-key-dashboard_page h1,
    .st-key-dashboard_page h2,
    .st-key-dashboard_page h3,
    .st-key-dashboard_page p,
    .st-key-dashboard_page [data-testid="stMarkdownContainer"],
    .st-key-dashboard_page [data-testid="stCaptionContainer"] {{
        color: #142a47 !important;
        text-shadow: none !important;
    }}

    .st-key-dashboard_page [data-testid="stVerticalBlockBorderWrapper"] {{
        border: 1px solid #e2eaf2 !important;
        border-radius: 0.9rem !important;
        background: rgba(255, 255, 255, 0.9) !important;
        box-shadow: 0 8px 24px rgba(30, 59, 91, 0.07);
        backdrop-filter: blur(12px);
    }}

    .st-key-rul_chart_panel {{
        box-sizing: border-box;
        min-width: 0;
        padding: 1rem;
        overflow: hidden;
        border: 1px solid rgba(115, 190, 219, 0.18);
        border-radius: 1rem;
        color: #e8f4ff;
        background:
            radial-gradient(ellipse at 50% 0%, rgba(19, 53, 91, 0.36), transparent 68%),
            linear-gradient(145deg, rgba(10, 15, 31, 0.98), rgba(11, 22, 43, 0.98));
        box-shadow: 0 18px 42px rgba(0, 0, 0, 0.24);
        animation: rul-panel-enter 650ms ease-out both;
    }}

    .st-key-rul_chart_panel h2,
    .st-key-rul_chart_panel p,
    .st-key-rul_chart_panel label,
    .st-key-rul_chart_panel [data-testid="stMarkdownContainer"],
    .st-key-rul_chart_panel [data-testid="stCaptionContainer"] {{
        color: #e8f4ff !important;
        text-shadow: none !important;
    }}

    .st-key-rul_chart_panel [data-testid="stPlotlyChart"] {{
        animation: rul-chart-enter 800ms cubic-bezier(0.2, 0.7, 0.25, 1) both;
    }}

    @keyframes rul-panel-enter {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes rul-chart-enter {{
        from {{ opacity: 0; transform: translateY(10px) scale(0.99); }}
        to {{ opacity: 1; transform: translateY(0) scale(1); }}
    }}

    .st-key-dashboard_page [data-baseweb="select"] > div {{
        border-color: #d3deea;
        color: #203b5a !important;
        background-color: #ffffff !important;
    }}

    .st-key-dashboard_page [data-baseweb="select"] input {{
        color: #203b5a !important;
        background-color: transparent !important;
    }}

    .dashboard-hero-copy {{
        padding: 0.55rem 0.25rem 0.25rem;
    }}

    .dashboard-eyebrow {{
        margin: 0 0 0.55rem;
        color: #3279c5 !important;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }}

    .dashboard-hero-copy h1 {{
        margin: 0 0 0.65rem;
        color: #142a47 !important;
        font-size: clamp(1.7rem, 3vw, 2.6rem);
        font-weight: 780;
        letter-spacing: -0.035em;
        line-height: 1.08;
    }}

    .dashboard-hero-copy p {{
        margin: 0;
        color: #334b67 !important;
        font-size: 0.94rem;
        line-height: 1.65;
    }}

    .dashboard-feature-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.55rem;
        margin-top: 1.05rem;
    }}

    .dashboard-feature {{
        display: flex;
        align-items: center;
        gap: 0.55rem;
        min-height: 3.2rem;
        padding: 0.6rem 0.7rem;
        border: 1px solid #e7edf4;
        border-radius: 0.75rem;
        color: #233d5b !important;
        background: rgba(255, 255, 255, 0.83);
        font-size: 0.76rem;
        font-weight: 700;
    }}

    .dashboard-feature-icon {{
        display: grid;
        width: 1.9rem;
        height: 1.9rem;
        flex: 0 0 1.9rem;
        place-items: center;
        border-radius: 0.55rem;
        background: #e5f1ff;
        font-size: 1rem;
    }}

    .dashboard-image-label {{
        margin: 0.5rem auto 0;
        width: fit-content;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        color: #08734f !important;
        background: #d9f5e9;
        font-size: 0.75rem;
        font-weight: 750;
        text-align: center;
    }}

    .dashboard-card-heading {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem;
        margin: 0;
        color: #142a47 !important;
        font-size: 0.95rem;
        font-weight: 760;
    }}

    .dashboard-card-heading span {{
        color: #3b6b9b !important;
    }}

    .dashboard-card-note {{
        color: #5b7088 !important;
        font-size: 0.72rem;
        font-weight: 550;
    }}

    .dashboard-metrics {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.55rem;
        margin-top: 0.75rem;
    }}

    .dashboard-metric {{
        min-height: 4.2rem;
        padding: 0.7rem;
        border: 1px solid transparent;
        border-radius: 0.65rem;
    }}

    .dashboard-metric.total {{
        color: #0756bf !important;
        border-color: #dbeafe;
        background: #eaf3ff;
    }}

    .dashboard-metric.normal {{
        color: #066448 !important;
        border-color: #d2f0e5;
        background: #e8f8f2;
    }}

    .dashboard-metric.defective {{
        color: #a31622 !important;
        border-color: #f9dce0;
        background: #fff0f1;
    }}

    .dashboard-metric strong {{
        display: block;
        font-size: clamp(1.05rem, 1.7vw, 1.4rem);
        line-height: 1.1;
    }}

    .dashboard-metric span {{
        display: block;
        margin-top: 0.3rem;
        color: #4a5f77 !important;
        font-size: 0.7rem;
    }}

    .dashboard-section-title {{
        margin: 0 0 0.7rem;
        color: #142a47 !important;
        font-size: 0.98rem;
        font-weight: 760;
    }}

    .dashboard-chart-title {{
        margin-bottom: 0.45rem;
        color: #142a47 !important;
        font-size: 0.82rem;
        font-weight: 760;
        white-space: nowrap;
    }}

    .dashboard-sample-label {{
        margin: 0.35rem 0 0;
        padding: 0.28rem 0.12rem;
        border-radius: 0.45rem;
        font-size: 0.6rem;
        font-weight: 750;
        text-align: center;
        white-space: nowrap;
    }}

    .dashboard-sample-label.normal {{
        color: #08734f !important;
        background: #def6ed;
    }}

    .dashboard-sample-label.defective {{
        color: #b4232d !important;
        background: #ffeaeb;
    }}

    .dashboard-flow {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.35rem;
        text-align: center;
    }}

    .dashboard-flow-step {{
        padding: 0.55rem 0.15rem;
        color: #304a67 !important;
        font-size: 0.74rem;
        line-height: 1.5;
    }}

    .dashboard-flow-step b {{
        display: block;
        margin: 0.25rem 0;
        color: #142a47 !important;
        font-size: 0.78rem;
    }}

    .dashboard-flow-icon {{
        display: grid;
        width: 2.7rem;
        height: 2.7rem;
        margin: 0 auto;
        place-items: center;
        border-radius: 50%;
        background: #e6f1ff;
        font-size: 1.2rem;
    }}

    .dashboard-inspection-flow {{
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr) auto minmax(0, 1.15fr);
        align-items: center;
        gap: 0.3rem;
        padding: 0.55rem 0;
    }}

    .dashboard-inspection-stage {{
        min-width: 0;
        text-align: center;
    }}

    .dashboard-inspection-icon {{
        display: grid;
        width: 2.5rem;
        height: 2.5rem;
        margin: 0 auto 0.45rem;
        place-items: center;
        border: 1px solid #dbeafe;
        border-radius: 0.75rem;
        color: #1762ad;
        background: #eff6ff;
    }}

    .dashboard-inspection-icon svg {{
        width: 1.45rem;
        height: 1.45rem;
        fill: none;
        stroke: currentColor;
        stroke-linecap: round;
        stroke-linejoin: round;
        stroke-width: 1.7;
    }}

    .dashboard-inspection-stage strong {{
        display: block;
        color: #183652 !important;
        font-size: 0.68rem;
        line-height: 1.25;
    }}

    .dashboard-inspection-stage small {{
        display: block;
        margin-top: 0.2rem;
        color: #60758b !important;
        font-size: 0.58rem;
        line-height: 1.3;
    }}

    .dashboard-inspection-arrow {{
        margin-top: -1.6rem;
        color: #91a6ba !important;
        font-size: 0.9rem;
    }}

    .dashboard-inspection-results {{
        display: grid;
        gap: 0.25rem;
        margin-top: 0.35rem;
    }}

    .dashboard-inspection-results span {{
        display: block;
        padding: 0.18rem 0.2rem;
        border: 1px solid #d2f0e5;
        border-radius: 0.35rem;
        color: #08734f !important;
        background: #e8f8f2;
        font-size: 0.56rem;
        font-weight: 700;
        line-height: 1.2;
    }}

    .dashboard-inspection-results span:last-child {{
        border-color: #f9dce0;
        color: #b4232d !important;
        background: #fff0f1;
    }}

    .dashboard-benefits {{
        margin: 0;
        padding: 0;
        list-style: none;
    }}

    .dashboard-benefits li {{
        margin: 0.62rem 0;
        color: #304a67 !important;
        font-size: 0.78rem;
        line-height: 1.4;
    }}

    .dashboard-benefits li::before {{
        content: "✓";
        display: inline-grid;
        width: 1.1rem;
        height: 1.1rem;
        margin-right: 0.5rem;
        place-items: center;
        border-radius: 50%;
        color: #fff;
        background: #527596;
        font-size: 0.7rem;
        font-weight: 800;
    }}

    .dashboard-mission {{
        height: 100%;
        padding: 0.15rem;
        color: #185b49 !important;
    }}

    .dashboard-mission p {{
        color: #315d53 !important;
        font-size: 0.8rem;
        line-height: 1.6;
    }}

    .dashboard-mission blockquote {{
        margin: 0.55rem 0 0;
        padding: 0.35rem 0 0 0.65rem;
        border-left: 2px solid #67b99f;
        color: #155d48 !important;
        font-size: 0.9rem;
        font-weight: 750;
    }}

    .st-key-dashboard_page [data-testid="stImage"] img {{
        width: 100%;
        max-height: 21rem;
        aspect-ratio: 1 / 1;
        object-fit: cover;
        border: 1px solid #e1e8ef;
        border-radius: 0.65rem;
    }}

    .inspection-result-title {{
        margin: 0;
        color: #fff;
        font-size: clamp(1.05rem, 2vw, 1.35rem);
        font-weight: 800;
        letter-spacing: 0.035em;
    }}

    .inspection-result-detail {{
        margin-top: 0.4rem;
        color: rgba(255, 255, 255, 0.88);
        font-size: 0.95rem;
    }}

    @media (max-width: 700px) {{
        [data-testid="stMainBlockContainer"] {{
            padding: 0.75rem 0.85rem 1.5rem;
        }}

        .ferro-brand-mark {{
            width: 2.35rem;
            height: 2.35rem;
        }}

        .st-key-app_navigation {{
            margin: -0.35rem -0.3rem 0.55rem;
            padding: 0.4rem;
        }}

        .ferro-brand-subtitle {{
            display: none;
        }}

        .stApp .st-key-app_navigation [data-testid="stBaseButton-primary"],
        .stApp .st-key-app_navigation [data-testid="stBaseButton-secondary"] {{
            min-height: 2.35rem;
            padding: 0.35rem 0.2rem;
            font-size: 0.72rem;
        }}

        [data-testid="stImage"] img {{
            max-height: 24rem;
        }}

        .st-key-dashboard_page {{
            margin: 0 -0.45rem;
            padding: 0.65rem;
        }}

        .dashboard-metrics {{
            gap: 0.35rem;
        }}

        .dashboard-metric {{
            min-height: 3.7rem;
            padding: 0.5rem 0.4rem;
        }}

        .dashboard-flow {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading turbofan dataset...")
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_casting_image_paths() -> dict[str, dict[str, list[Path]]]:
    return {
        "train": {
            label: sorted((CASTING_TRAIN_PATH / label).glob("*"))
            for label in ("normal", "defective")
        },
        "validation": {
            label: sorted((CASTING_VALIDATION_PATH / label).glob("*"))
            for label in ("normal", "defective")
        },
    }


@st.cache_resource(show_spinner="Loading the trained ANN...")
def load_ann_artifacts():
    if not ANN_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained ANN model not found: {ANN_MODEL_PATH}")
    if not ANN_PREPROCESSING_PATH.is_file():
        raise FileNotFoundError(
            f"ANN preprocessing data not found: {ANN_PREPROCESSING_PATH}"
        )
    model = tf.keras.models.load_model(ANN_MODEL_PATH)
    preprocessing = joblib.load(ANN_PREPROCESSING_PATH)
    return model, preprocessing["scaler"], preprocessing["feature_columns"]


@st.cache_resource(show_spinner="Loading the trained image classifier...")
def load_image_model():
    if not IMAGE_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained image model not found: {IMAGE_MODEL_PATH}")
    return tf.keras.models.load_model(IMAGE_MODEL_PATH)


@st.cache_resource(show_spinner="Loading image feature extractor...")
def load_image_feature_extractor():
    image_model = load_image_model()
    return (
        image_model,
        create_feature_extractor(image_model),
        create_classifier_head(image_model),
    )


@st.cache_resource(show_spinner=False)
def load_casting_reference_embeddings(
    _feature_extractor: Callable[[tf.Tensor], tf.Tensor],
    model_name: str,
    fingerprint: str,
) -> np.ndarray:
    image_paths = find_reference_images()
    progress_bar = st.progress(0, text="Preparing casting image references...")

    def report_progress(processed: int, total: int) -> None:
        progress_bar.progress(
            processed / total,
            text=f"Extracting casting references: {processed:,}/{total:,}",
        )

    try:
        return load_or_create_reference_embeddings(
            _feature_extractor,
            model_name,
            image_paths,
            fingerprint,
            progress_callback=report_progress,
        )
    finally:
        progress_bar.empty()


def make_rul_prediction(
    model: tf.keras.Model,
    scaler,
    feature_columns: list[str],
    input_row: pd.DataFrame,
) -> float:
    ordered_features = input_row.loc[:, feature_columns]
    numeric_features = ordered_features.apply(pd.to_numeric, errors="coerce")
    values = numeric_features.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("All model feature values must be finite numbers.")
    scaled_features = scaler.transform(numeric_features)
    prediction = float(model.predict(scaled_features, verbose=0).reshape(-1)[0])
    if not np.isfinite(prediction):
        raise ValueError("The model returned an invalid RUL estimate.")
    return prediction


def show_overview(df: pd.DataFrame) -> None:
    image_paths = load_casting_image_paths()
    normal_paths = image_paths["train"]["normal"] + image_paths["validation"]["normal"]
    defective_paths = (
        image_paths["train"]["defective"] + image_paths["validation"]["defective"]
    )
    if not normal_paths or not defective_paths:
        raise FileNotFoundError(
            "The casting dashboard requires labeled images in "
            "Data/Image/train and Data/Image/validation."
        )

    total_images = len(normal_paths) + len(defective_paths)

    def pick_samples(paths: list[Path], count: int) -> list[Path]:
        if count == 1:
            return [paths[len(paths) // 2]]
        return [
            paths[round(index * (len(paths) - 1) / (count - 1))]
            for index in range(count)
        ]

    with st.container(key="dashboard_page"):
        hero_text, hero_image, dashboard_stats = st.columns(
            [1.12, 0.78, 1.42],
            gap="medium",
            vertical_alignment="center",
        )
        with hero_text:
            st.markdown(
                f"""
                <section class="dashboard-hero-copy">
                    <div class="dashboard-eyebrow">AI-powered quality inspection</div>
                    <h1>Casting Defect Detection</h1>
                    <p>Ferro Vision uses computer vision and deep learning to
                    screen cast-metal components as normal or defective.
                    Review the labeled image dataset and inspect example
                    components before running a classification.</p>
                    <div class="dashboard-feature-grid">
                        <div class="dashboard-feature"><span class="dashboard-feature-icon">⚡</span>Automated review</div>
                        <div class="dashboard-feature"><span class="dashboard-feature-icon">◎</span>Image validation</div>
                        <div class="dashboard-feature"><span class="dashboard-feature-icon">▥</span>Defect screening</div>
                        <div class="dashboard-feature"><span class="dashboard-feature-icon">⚙</span>Quality workflow</div>
                    </div>
                </section>
                """,
                unsafe_allow_html=True,
            )
        with hero_image:
            st.image(str(normal_paths[len(normal_paths) // 2]), width="stretch")
            st.markdown(
                '<div class="dashboard-image-label">Normal validation example</div>',
                unsafe_allow_html=True,
            )
        with dashboard_stats:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="dashboard-card-heading">
                        <span>▤ &nbsp; Casting dataset overview</span>
                        <span class="dashboard-card-note">{total_images:,} images</span>
                    </div>
                    <div class="dashboard-metrics">
                        <div class="dashboard-metric total"><strong>{total_images:,}</strong><span>Total images</span></div>
                        <div class="dashboard-metric normal"><strong>{len(normal_paths):,}</strong><span>Normal</span></div>
                        <div class="dashboard-metric defective"><strong>{len(defective_paths):,}</strong><span>Defective</span></div>
                    </div>
                    <div class="dashboard-card-note" style="margin-top:0.55rem">
                        Counts include the training and validation splits.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            distribution_card, count_card = st.columns(2, gap="small")
            with distribution_card:
                with st.container(border=True):
                    st.markdown(
                        '<div class="dashboard-chart-title">Class distribution</div>',
                        unsafe_allow_html=True,
                    )
                    figure, axis = plt.subplots(figsize=(2.75, 2.2))
                    figure.patch.set_facecolor("white")
                    axis.set_facecolor("white")
                    axis.pie(
                        [len(normal_paths), len(defective_paths)],
                        colors=["#39b984", "#ff6868"],
                        startangle=90,
                        counterclock=False,
                        wedgeprops={"width": 0.38, "edgecolor": "white"},
                        autopct="%1.0f%%",
                        pctdistance=0.8,
                        textprops={"color": "#ffffff", "fontsize": 8, "weight": "bold"},
                    )
                    axis.legend(
                        ["Normal", "Defective"],
                        loc="lower center",
                        bbox_to_anchor=(0.5, -0.12),
                        ncol=2,
                        frameon=False,
                        fontsize=7,
                    )
                    axis.set(aspect="equal")
                    figure.tight_layout(pad=0.2)
                    st.pyplot(figure, width="stretch")
                    plt.close(figure)
            with count_card:
                with st.container(border=True):
                    st.markdown(
                        '<div class="dashboard-chart-title">Labeled image counts</div>',
                        unsafe_allow_html=True,
                    )
                    figure, axis = plt.subplots(figsize=(2.75, 2.2))
                    figure.patch.set_facecolor("white")
                    axis.set_facecolor("white")
                    bars = axis.bar(
                        ["Normal", "Defective"],
                        [len(normal_paths), len(defective_paths)],
                        color=["#39b984", "#ff6868"],
                        width=0.58,
                    )
                    axis.tick_params(axis="x", labelsize=7, colors="#435a73")
                    axis.set_yticks([])
                    axis.grid(axis="y", color="#e7edf3", linewidth=0.8)
                    axis.set_axisbelow(True)
                    for bar, count in zip(
                        bars, [len(normal_paths), len(defective_paths)]
                    ):
                        axis.annotate(
                            f"{count:,}",
                            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center",
                            fontsize=8,
                            color="#1c3552",
                            weight="bold",
                        )
                    for spine in axis.spines.values():
                        spine.set_visible(False)
                    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.2, top=0.94)
                    st.pyplot(figure, width="stretch")
                    plt.close(figure)

        samples_column, rul_chart_column = st.columns(
            [1.25, 1.35],
            gap="medium",
        )
        with samples_column:
            with st.container(border=True):
                st.markdown(
                    '<div class="dashboard-section-title">▧ &nbsp; Sample components from validation data</div>',
                    unsafe_allow_html=True,
                )
                sample_paths = pick_samples(normal_paths, 3) + pick_samples(
                    defective_paths, 3
                )
                sample_labels = ["Normal"] * 3 + ["Defective"] * 3
                sample_columns = st.columns(6, gap="small")
                for column, sample_path, label in zip(
                    sample_columns, sample_paths, sample_labels
                ):
                    with column:
                        st.image(str(sample_path), width="stretch")
                        label_class = "normal" if label == "Normal" else "defective"
                        st.markdown(
                            f'<div class="dashboard-sample-label {label_class}">{label}</div>',
                            unsafe_allow_html=True,
                        )
        with rul_chart_column:
            with st.container(key="rul_chart_panel"):
                selected_engine = sorted(df["engine_id"].unique().tolist())[0]
                st.markdown(
                    f'<div class="dashboard-card-note" style="color:#7f9bb9 !important;letter-spacing:.1em;text-transform:uppercase">Engine {selected_engine} &nbsp;·&nbsp; Remaining useful life</div>',
                    unsafe_allow_html=True,
                )
                engine_data = (
                    df.loc[df["engine_id"] == selected_engine]
                    .sort_values("cycle")
                )
                cycles = engine_data["cycle"].to_numpy()
                rul_values = engine_data["RUL"].to_numpy()
                hover_template = (
                    "<b>Engine Cycles</b>: %{x:.0f}<br>"
                    "<b>Remaining Useful Life</b>: %{y:.1f} cycles"
                    "<extra></extra>"
                )
                figure = go.Figure()
                figure.add_trace(
                    go.Scatter(
                        x=cycles,
                        y=rul_values,
                        mode="lines",
                        line={
                            "color": "rgba(0,191,255,0.16)",
                            "width": 13,
                            "shape": "spline",
                            "smoothing": 0.4,
                        },
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                current_cycle = cycles[-1]
                current_rul = rul_values[-1]
                figure.add_trace(
                    go.Scatter(
                        x=cycles,
                        y=rul_values,
                        mode="lines",
                        line={
                            "color": "rgba(0,191,255,0.34)",
                            "width": 7,
                            "shape": "spline",
                            "smoothing": 0.4,
                        },
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                figure.add_trace(
                    go.Scatter(
                        x=cycles,
                        y=rul_values,
                        mode="lines",
                        line={
                            "color": "#00BFFF",
                            "width": 2.8,
                            "shape": "spline",
                            "smoothing": 0.4,
                        },
                        fill="tozeroy",
                        fillgradient={
                            "type": "vertical",
                            "colorscale": [
                                [0, "rgba(0,191,255,0.015)"],
                                [1, "rgba(0,191,255,0.34)"],
                            ],
                        },
                        hovertemplate=hover_template,
                        showlegend=False,
                    )
                )
                figure.add_trace(
                    go.Scatter(
                        x=[current_cycle],
                        y=[current_rul],
                        mode="markers",
                        marker={
                            "size": 22,
                            "color": "rgba(0,191,255,0.18)",
                            "line": {"width": 0},
                        },
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                figure.add_trace(
                    go.Scatter(
                        x=[current_cycle],
                        y=[current_rul],
                        mode="markers",
                        marker={
                            "size": 9,
                            "color": "#DDF8FF",
                            "line": {"color": "#00BFFF", "width": 3},
                        },
                        hovertemplate=hover_template,
                        showlegend=False,
                    )
                )
                figure.update_layout(
                    title={
                        "text": "Remaining Useful Life Trend",
                        "x": 0.02,
                        "xanchor": "left",
                        "font": {
                            "size": 19,
                            "color": "#F1F7FF",
                            "family": "Segoe UI, Inter, sans-serif",
                        },
                    },
                    autosize=True,
                    height=330,
                    margin={"l": 58, "r": 18, "t": 58, "b": 54},
                    paper_bgcolor="#0A0F1F",
                    plot_bgcolor="#0A0F1F",
                    font={"color": "#9BB0C9", "size": 11},
                    hoverlabel={
                        "bgcolor": "#14233A",
                        "bordercolor": "#00BFFF",
                        "font": {"color": "#F1F7FF", "size": 12},
                    },
                    hovermode="x",
                    transition={"duration": 650, "easing": "cubic-in-out"},
                    uirevision=str(selected_engine),
                    xaxis={
                        "title": {
                            "text": "Engine Cycles",
                            "font": {"color": "#AFC3DA", "size": 11},
                        },
                        "showgrid": True,
                        "gridcolor": "rgba(136,164,196,0.12)",
                        "zeroline": False,
                        "showline": False,
                        "tickfont": {"color": "#839AB5", "size": 10},
                        "automargin": True,
                    },
                    yaxis={
                        "title": {
                            "text": "Remaining Useful Life (RUL)",
                            "font": {"color": "#AFC3DA", "size": 11},
                        },
                        "showgrid": True,
                        "gridcolor": "rgba(136,164,196,0.12)",
                        "zeroline": False,
                        "showline": False,
                        "tickfont": {"color": "#839AB5", "size": 10},
                        "rangemode": "tozero",
                        "automargin": True,
                    },
                )
                st.plotly_chart(
                    figure,
                    width="stretch",
                    config={
                        "responsive": True,
                        "displayModeBar": False,
                        "displaylogo": False,
                    },
                    key="overview_rul_chart",
                )

        sensor_columns = sorted(
            (column for column in df.columns if column.startswith("sensor_")),
            key=lambda column: int(column.rsplit("_", 1)[1]),
        )
        engine_ids = sorted(df["engine_id"].unique().tolist())
        with st.container(border=True):
            st.markdown(
                '<div class="dashboard-section-title">▥ &nbsp; Turbofan sensor trends</div>'
                '<div class="dashboard-card-note">Explore how engine telemetry changes across operating cycles.</div>',
                unsafe_allow_html=True,
            )
            engine_control, sensor_control = st.columns(2, gap="medium")
            with engine_control:
                sensor_engine = st.selectbox(
                    "Engine",
                    engine_ids,
                    key="overview_sensor_engine",
                )
            with sensor_control:
                default_sensor_index = (
                    sensor_columns.index("sensor_2")
                    if "sensor_2" in sensor_columns
                    else 0
                )
                selected_sensor = st.selectbox(
                    "Telemetry sensor",
                    sensor_columns,
                    index=default_sensor_index,
                    key="overview_sensor_choice",
                )

            sensor_data = (
                df.loc[df["engine_id"] == sensor_engine]
                .sort_values("cycle")
            )
            sensor_figure = go.Figure(
                go.Scatter(
                    x=sensor_data["cycle"],
                    y=sensor_data[selected_sensor],
                    mode="lines",
                    line={"color": "#00BFFF", "width": 2.5},
                    hovertemplate=(
                        "<b>Cycle</b>: %{x:.0f}<br>"
                        f"<b>{selected_sensor.replace('_', ' ').title()}</b>: "
                        "%{y:.3f}<extra></extra>"
                    ),
                )
            )
            sensor_figure.update_layout(
                title={
                    "text": f"{selected_sensor.replace('_', ' ').title()} · Engine {sensor_engine}",
                    "x": 0.02,
                    "xanchor": "left",
                    "font": {
                        "size": 18,
                        "color": "#F1F7FF",
                        "family": "Segoe UI, Inter, sans-serif",
                    },
                },
                autosize=True,
                height=320,
                margin={"l": 58, "r": 18, "t": 55, "b": 50},
                paper_bgcolor="#0A0F1F",
                plot_bgcolor="#0A0F1F",
                font={"color": "#9BB0C9", "size": 11},
                hoverlabel={
                    "bgcolor": "#14233A",
                    "bordercolor": "#00BFFF",
                    "font": {"color": "#F1F7FF", "size": 12},
                },
                hovermode="x",
                uirevision=f"{sensor_engine}-{selected_sensor}",
                xaxis={
                    "title": "Engine Cycles",
                    "showgrid": True,
                    "gridcolor": "rgba(136,164,196,0.12)",
                    "zeroline": False,
                    "tickfont": {"color": "#839AB5", "size": 10},
                    "automargin": True,
                },
                yaxis={
                    "title": selected_sensor.replace("_", " ").title(),
                    "showgrid": True,
                    "gridcolor": "rgba(136,164,196,0.12)",
                    "zeroline": False,
                    "tickfont": {"color": "#839AB5", "size": 10},
                    "automargin": True,
                },
            )
            st.plotly_chart(
                sensor_figure,
                width="stretch",
                config={
                    "responsive": True,
                    "displayModeBar": False,
                    "displaylogo": False,
                },
                key="overview_sensor_chart",
            )

        workflow_column, inspection_column, benefits_column, mission_column = st.columns(
            [1.4, 1.1, 0.9, 0.95],
            gap="medium",
        )
        with workflow_column:
            with st.container(border=True):
                st.markdown(
                    '<div class="dashboard-section-title">⚙ &nbsp; How it works</div>'
                    '<div class="dashboard-flow">'
                    '<div class="dashboard-flow-step"><span class="dashboard-flow-icon">▧</span><b>1. Input image</b>Select a casting image</div>'
                    '<div class="dashboard-flow-step"><span class="dashboard-flow-icon">▦</span><b>2. AI analysis</b>Extract image features</div>'
                    '<div class="dashboard-flow-step"><span class="dashboard-flow-icon">⌕</span><b>3. Defect screening</b>Check for visual patterns</div>'
                    '<div class="dashboard-flow-step"><span class="dashboard-flow-icon">✓</span><b>4. Quality result</b>Normal or defective</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
        with inspection_column:
            with st.container(border=True):
                st.markdown(
                    '<div class="dashboard-section-title">✦ &nbsp; AI casting inspection</div>'
                    '<div class="dashboard-inspection-flow">'
                    '<div class="dashboard-inspection-stage">'
                    '<div class="dashboard-inspection-icon">'
                    '<svg viewBox="0 0 24 24" aria-hidden="true">'
                    '<path d="M5 4.5h14a1.5 1.5 0 0 1 1.5 1.5v12a1.5 1.5 0 0 1-1.5 1.5H5A1.5 1.5 0 0 1 3.5 18V6A1.5 1.5 0 0 1 5 4.5Z"/>'
                    '<path d="m7 16 3.2-3.5 2.4 2.2 2.1-2.4 2.3 3.7H7Z"/>'
                    '<circle cx="9" cy="8.5" r="1.2"/>'
                    '</svg></div>'
                    '<strong>Input part</strong>'
                    '<small>Casting image</small>'
                    '</div>'
                    '<span class="dashboard-inspection-arrow">→</span>'
                    '<div class="dashboard-inspection-stage">'
                    '<div class="dashboard-inspection-icon">'
                    '<svg viewBox="0 0 24 24" aria-hidden="true">'
                    '<circle cx="10.5" cy="10.5" r="6.5"/>'
                    '<path d="m15.3 15.3 5.2 5.2M10.5 7v7M7 10.5h7"/>'
                    '</svg></div>'
                    '<strong>Feature scan</strong>'
                    '<small>AI checks visual patterns</small>'
                    '</div>'
                    '<span class="dashboard-inspection-arrow">→</span>'
                    '<div class="dashboard-inspection-stage">'
                    '<div class="dashboard-inspection-icon">'
                    '<svg viewBox="0 0 24 24" aria-hidden="true">'
                    '<path d="M12 3.5 20 7v5.2c0 4.3-3.1 7.1-8 8.8-4.9-1.7-8-4.5-8-8.8V7l8-3.5Z"/>'
                    '<path d="m8.5 12.2 2.2 2.2 4.8-5"/>'
                    '</svg></div>'
                    '<strong>Screening</strong>'
                    '<div class="dashboard-inspection-results">'
                    '<span>✓ Normal</span>'
                    '<span>! Defective</span>'
                    '</div>'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
        with benefits_column:
            with st.container(border=True):
                st.markdown(
                    '<div class="dashboard-section-title">▥ &nbsp; Key capabilities</div>'
                    '<ul class="dashboard-benefits">'
                    '<li>Consistent image-based screening</li>'
                    '<li>Validation before classification</li>'
                    '<li>Clear normal/defective output</li>'
                    '<li>Separate RUL prediction workflow</li>'
                    '</ul>',
                    unsafe_allow_html=True,
                )
        with mission_column:
            with st.container(border=True):
                st.markdown(
                    '<div class="dashboard-mission">'
                    '<div class="dashboard-section-title">◎ &nbsp; Our mission</div>'
                    '<p>Support smarter manufacturing through AI-assisted visual inspection and data-informed maintenance.</p>'
                    '<blockquote>Better Vision. Stronger Industry.</blockquote>'
                    '<p style="text-align:right;margin-bottom:0">— Ferro Vision</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )


def show_rul_prediction(df: pd.DataFrame) -> None:
    st.title("Remaining Useful Life Prediction")
    st.write(
        "Predict the remaining operational life of an engine using our AI-powered ANN model. Analyze telemetry sensor data to estimate the number of cycles remaining before failure, enabling predictive maintenance, improved reliability, and reduced downtime."
    )

    model, scaler, feature_columns = load_ann_artifacts()
    mode = st.radio(
        "Input source",
        ["Dataset example", "Upload telemetry CSV", "Manual Entries"],
        horizontal=True,
    )
    input_row = None
    actual_rul = None
    example_context = None

    if mode == "Dataset example":
        engine_ids = sorted(df["engine_id"].unique().tolist())
        engine_id = st.selectbox("Engine", engine_ids)
        engine_data = df.loc[df["engine_id"] == engine_id].sort_values("cycle")
        cycles = engine_data["cycle"].astype(int).tolist()
        default_cycle = cycles[len(cycles) // 2]
        cycle = st.select_slider("Cycle", options=cycles, value=default_cycle)
        selected_row = engine_data.loc[engine_data["cycle"] == cycle].iloc[[0]]
        input_row = selected_row.loc[:, feature_columns]
        actual_rul = float(selected_row["RUL"].iloc[0])
        example_context = f"Engine {engine_id}, cycle {cycle}"
        with st.expander("View selected model inputs"):
            st.dataframe(input_row, width="stretch")
    elif mode == "Upload telemetry CSV":
        template_row = df.loc[:, feature_columns].iloc[[0]]
        st.download_button(
            "Download one-row CSV template",
            template_row.to_csv(index=False).encode("utf-8"),
            file_name="rul_input_template.csv",
            mime="text/csv",
        )
        uploaded_csv = st.file_uploader(
            "Upload a CSV with exactly one telemetry row",
            type=["csv"],
            help=(
                "Include these 25 model features: cycle, op_setting_1–3, "
                "and sensor_1–21. Extra columns such as engine_id or RUL are ignored."
            ),
        )
        if uploaded_csv is not None:
            uploaded_data = pd.read_csv(uploaded_csv)
            missing_columns = [
                column for column in feature_columns if column not in uploaded_data
            ]
            if missing_columns:
                st.error(
                    "CSV is missing required ANN features: "
                    + ", ".join(missing_columns)
                )
            elif len(uploaded_data) != 1:
                st.error(
                    f"Upload exactly one telemetry row; received {len(uploaded_data)}."
                )
            else:
                input_row = uploaded_data.loc[:, feature_columns]
                with st.expander("View uploaded model inputs"):
                    st.dataframe(input_row, width="stretch")
    else:
        st.write(
            "Enter values for all 25 input features. Fields start with "
            "typical dataset medians; edit them to match the engine telemetry."
        )
        median_values = df[feature_columns].median()
        manual_values = {}
        input_columns = st.columns(3)
        for index, feature in enumerate(feature_columns):
            with input_columns[index % len(input_columns)]:
                default_value = float(median_values[feature])
                if feature == "cycle":
                    manual_values[feature] = st.number_input(
                        feature,
                        min_value=1,
                        value=max(1, int(default_value)),
                        step=1,
                        key=f"manual_input_{feature}",
                    )
                else:
                    manual_values[feature] = st.number_input(
                        feature,
                        value=default_value,
                        format="%.5f",
                        key=f"manual_input_{feature}",
                    )
        input_row = pd.DataFrame([manual_values], columns=feature_columns)

    if input_row is not None and st.button("Predict RUL", type="primary"):
        try:
            prediction = make_rul_prediction(
                model, scaler, feature_columns, input_row
            )
        except ValueError as error:
            st.error(str(error))
        else:
            st.metric("Predicted remaining useful life", f"{prediction:.1f} cycles")
            if example_context is not None and actual_rul is not None:
                st.caption(
                    f"Dataset example: {example_context}. The recorded RUL is "
                    f"{actual_rul:.0f} cycles; it is shown as a reference, not used "
                    "as a model input."
                )
            cycles_used = float(input_row["cycle"].iloc[0])
            rul_axis_min = min(0.0, cycles_used, prediction)
            rul_axis_max = max(1.0, cycles_used, prediction) * 1.1
            rul_figure = go.Figure(
                data=[
                    go.Bar(
                        name="Cycles used",
                        x=[cycles_used],
                        y=["Cycles used"],
                        orientation="h",
                        marker={"color": "#00BFFF"},
                        hovertemplate=(
                            "<b>Cycles used</b><br>%{x:.0f} cycles<extra></extra>"
                        ),
                    ),
                    go.Bar(
                        name="Sensor-based estimated RUL",
                        x=[prediction],
                        y=["Estimated remaining life"],
                        orientation="h",
                        marker={"color": "#53C99A"},
                        hovertemplate=(
                            "<b>Estimated remaining life</b><br>"
                            "%{x:.1f} cycles<extra></extra>"
                        ),
                    ),
                ]
            )
            rul_figure.update_layout(
                title={
                    "text": "Cycle Usage and Estimated Remaining Life",
                    "x": 0.02,
                    "xanchor": "left",
                    "font": {
                        "size": 17,
                        "color": "#F1F7FF",
                        "family": "Segoe UI, Inter, sans-serif",
                    },
                },
                autosize=True,
                height=190,
                margin={"l": 120, "r": 30, "t": 62, "b": 48},
                paper_bgcolor="#0A0F1F",
                plot_bgcolor="#0A0F1F",
                font={"color": "#C6D5E6", "size": 11},
                barmode="overlay",
                bargap=0.5,
                legend={
                    "orientation": "h",
                    "x": 0.02,
                    "y": 1.12,
                    "font": {"color": "#AFC3DA", "size": 10},
                },
                xaxis={
                    "title": {
                        "text": "Engine cycles",
                        "font": {"color": "#AFC3DA", "size": 11},
                    },
                    "range": [rul_axis_min, rul_axis_max],
                    "showgrid": True,
                    "gridcolor": "rgba(136,164,196,0.12)",
                    "zeroline": True,
                    "zerolinecolor": "rgba(136,164,196,0.35)",
                    "tickfont": {"color": "#839AB5", "size": 10},
                    "automargin": True,
                },
                yaxis={
                    "showgrid": False,
                    "zeroline": False,
                    "tickfont": {"color": "#C6D5E6", "size": 11},
                    "automargin": True,
                },
                hoverlabel={
                    "bgcolor": "#14233A",
                    "bordercolor": "#00BFFF",
                    "font": {"color": "#F1F7FF", "size": 12},
                },
            )
            st.caption(
                "Each bar starts at zero. The remaining-life estimate is based on "
                "the input cycle and sensor readings."
            )
            st.plotly_chart(
                rul_figure,
                width="stretch",
                config={
                    "responsive": True,
                    "displayModeBar": False,
                    "displaylogo": False,
                },
                key="rul_lifecycle_chart",
            )


def show_image_classification() -> None:
    st.markdown(
        '<h1 class="image-page-heading">Industrial Component Inspection</h1>'
        '<p class="image-page-description">AI-assisted casting quality inspection '
        'with image validation and defect classification.</p>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        selection_column, upload_column = st.columns(
            [3, 1.15], vertical_alignment="bottom"
        )
        with selection_column:
            st.markdown(
                '<p class="section-eyebrow">Image Selection</p>',
                unsafe_allow_html=True,
            )
            uploaded_image = st.file_uploader(
                "Select casting image",
                type=["jpg", "jpeg", "png", "bmp"],
                key="component_image",
                label_visibility="collapsed",
                help="Select a casting component image in JPG, PNG, or BMP format.",
            )
        with upload_column:
            upload_clicked = st.button(
                "⬆️  Upload Image",
                type="primary",
                disabled=uploaded_image is None,
                key="upload_component_image",
                width="stretch",
            )

    selected_bytes = uploaded_image.getvalue() if uploaded_image is not None else None
    if uploaded_image is not None:
        st.caption(f"Selected file: **{uploaded_image.name}**")

    if upload_clicked and uploaded_image is not None and selected_bytes is not None:
        st.session_state["inspection_image_bytes"] = selected_bytes
        st.session_state["inspection_image_name"] = uploaded_image.name
        st.session_state["inspection_result"] = None

    image_bytes = st.session_state.get("inspection_image_bytes")
    image_name = st.session_state.get("inspection_image_name")
    if image_bytes is None or image_name is None:
        st.info("Choose an image, then select **Upload Image** to prepare the inspection.")
        return

    if selected_bytes is not None and selected_bytes != image_bytes:
        st.info("A new file is selected. Select **Upload Image** to inspect it.")
        return

    try:
        with Image.open(BytesIO(image_bytes)) as source_image:
            image_format = source_image.format or "Image"
            image_dimensions = source_image.size
            image = source_image.convert("RGB")
    except (OSError, ValueError) as error:
        st.error(f"Could not read the uploaded image: {error}")
        return

    preview_column, action_column = st.columns([1.3, 1], gap="large")
    with preview_column:
        with st.container(border=True):
            st.markdown(
                '<p class="section-eyebrow">Inspection Sample</p>'
                '<h3>Component Preview</h3>',
                unsafe_allow_html=True,
            )
            st.image(image, width="stretch")
            st.markdown(
                f'<div class="inspection-meta"><strong>{image_name}</strong><br>'
                f'{image_format} &nbsp;·&nbsp; {image_dimensions[0]} × '
                f'{image_dimensions[1]} px &nbsp;·&nbsp; '
                f'{len(image_bytes) / (1024 * 1024):.2f} MB</div>',
                unsafe_allow_html=True,
            )

    with action_column:
        with st.container(border=True):
            st.markdown(
                '<p class="section-eyebrow">AI Quality Check</p>'
                '<h3>Inspection Controls</h3>',
                unsafe_allow_html=True,
            )
            validate_clicked = st.button(
                "🔍  Validate & Classify Image",
                type="primary",
                key="validate_component_image",
                width="stretch",
            )

            if validate_clicked:
                try:
                    with st.spinner("Validating and inspecting component..."):
                        image_model, feature_extractor, classifier_head = (
                            load_image_feature_extractor()
                        )
                        reference_images = find_reference_images()
                        fingerprint = create_dataset_fingerprint(
                            reference_images, image_model.name
                        )
                        reference_embeddings = load_casting_reference_embeddings(
                            feature_extractor, image_model.name, fingerprint
                        )
                        feature_map = extract_feature_map(
                            feature_extractor, image, image_model.name
                        )
                        query_embedding = normalize_embedding(feature_map)
                        maximum_similarity, _ = calculate_similarity(
                            query_embedding, reference_embeddings
                        )

                        if maximum_similarity <= ACCEPTANCE_THRESHOLD:
                            st.session_state["inspection_result"] = {
                                "status": "invalid",
                            }
                        else:
                            pooled_features = np.mean(
                                feature_map, axis=(0, 1)
                            )[None, ...]
                            defective_probability = float(
                                classifier_head(
                                    pooled_features, training=False
                                ).numpy().reshape(-1)[0]
                            )
                            is_defective = defective_probability >= 0.5
                            st.session_state["inspection_result"] = {
                                "status": (
                                    "defective" if is_defective else "normal"
                                ),
                            }
                except (OSError, ValueError, tf.errors.OpError) as error:
                    st.session_state["inspection_result"] = None
                    st.error(f"Inspection failed: {error}")

            result = st.session_state.get("inspection_result")
            if result is not None:
                if result["status"] == "invalid":
                    st.markdown(
                        '<div class="inspection-result invalid">'
                        '<p class="inspection-result-title">⚠️ INVALID IMAGE</p>'
                        '<div class="inspection-result-detail">Please upload a valid '
                        'casting component image.</div></div>',
                        unsafe_allow_html=True,
                    )
                elif result["status"] == "normal":
                    st.markdown(
                        '<div class="inspection-result normal">'
                        '<p class="inspection-result-title">✅ NORMAL COMPONENT</p>'
                        '<div class="inspection-result-detail">Quality Inspection Passed</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="inspection-result defective">'
                        '<p class="inspection-result-title">❌ DEFECTIVE COMPONENT</p>'
                        '<div class="inspection-result-detail">Defect Detected</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )


def main() -> None:
    try:
        df = load_dataset()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    pages = ["Overview", "RUL Prediction", "Image Classification"]
    if "page_navigation" not in st.session_state:
        st.session_state["page_navigation"] = pages[0]

    with st.container(key="app_navigation"):
        brand_column, spacer_column, navigation_column = st.columns(
            [1.5, 1.05, 2.5],
            gap="small",
            vertical_alignment="center",
        )
        with brand_column:
            st.markdown(
                '<header class="ferro-brand-header">'
                '<div class="ferro-brand-mark">⚙</div>'
                '<div><div class="ferro-brand-name">Ferro Vision</div>'
                '<div class="ferro-brand-subtitle">Casting Defect Detection</div></div>'
                '</header>',
                unsafe_allow_html=True,
            )
        spacer_column.empty()
        with navigation_column:
            navigation_columns = st.columns(len(pages), gap="small")
            page_labels = {
                "Overview": "Overview",
                "RUL Prediction": "RUL Predict",
                "Image Classification": "Image Inspect",
            }
            for column, page_name in zip(navigation_columns, pages):
                column.button(
                    page_labels[page_name],
                    key=f"navigate_{page_name}",
                    type=(
                        "primary"
                        if st.session_state["page_navigation"] == page_name
                        else "secondary"
                    ),
                    on_click=lambda selected_page=page_name: st.session_state.update(
                        page_navigation=selected_page
                    ),
                    width="stretch",
                )
    page = st.session_state["page_navigation"]
    if page == "Overview":
        try:
            show_overview(df)
        except FileNotFoundError as error:
            st.error(str(error))
    elif page == "RUL Prediction":
        try:
            show_rul_prediction(df)
        except FileNotFoundError as error:
            st.error(str(error))
    else:
        try:
            show_image_classification()
        except FileNotFoundError as error:
            st.error(str(error))


if __name__ == "__main__":
    main()
