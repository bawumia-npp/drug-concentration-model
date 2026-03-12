"""
Pharmacokinetics Dashboard - Professional Biomedical Simulation Platform
Clinical Pharmacokinetic Analysis System using First-Order Elimination Model
Version: 2.0.0 | Production Ready
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sqlite3
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from io import BytesIO

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Pharmacokinetics Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """Manages SQLite database operations for simulation history."""
    
    def __init__(self, db_path: str = "pharmacokinetics_history.db"):
        """Initialize database connection."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                simulation_name TEXT NOT NULL,
                simulation_mode TEXT NOT NULL,
                c0 REAL NOT NULL,
                k_values TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                time_points INTEGER NOT NULL,
                time_unit TEXT NOT NULL,
                mec REAL,
                mtc REAL,
                cmax REAL,
                tmax REAL,
                half_life REAL,
                auc REAL,
                time_above_mec REAL,
                time_above_mtc REAL,
                interpretation TEXT,
                json_data TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_simulation(self, sim_data: Dict) -> bool:
        """Save simulation to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO simulations 
                (timestamp, simulation_name, simulation_mode, c0, k_values, 
                 start_time, end_time, time_points, time_unit, mec, mtc,
                 cmax, tmax, half_life, auc, time_above_mec, time_above_mtc,
                 interpretation, json_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sim_data['timestamp'],
                sim_data['simulation_name'],
                sim_data['simulation_mode'],
                sim_data['c0'],
                sim_data['k_values'],
                sim_data['start_time'],
                sim_data['end_time'],
                sim_data['time_points'],
                sim_data['time_unit'],
                sim_data['mec'],
                sim_data['mtc'],
                sim_data['cmax'],
                sim_data['tmax'],
                sim_data['half_life'],
                sim_data['auc'],
                sim_data['time_above_mec'],
                sim_data['time_above_mtc'],
                sim_data['interpretation'],
                json.dumps(sim_data['json_data'])
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            return False
    
    def get_all_simulations(self) -> List[Dict]:
        """Retrieve all simulations from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM simulations ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            conn.close()
            
            simulations = []
            for row in rows:
                simulations.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'simulation_name': row[2],
                    'simulation_mode': row[3],
                    'c0': row[4],
                    'k_values': row[5],
                    'start_time': row[6],
                    'end_time': row[7],
                    'time_points': row[8],
                    'time_unit': row[9],
                    'mec': row[10],
                    'mtc': row[11],
                    'cmax': row[12],
                    'tmax': row[13],
                    'half_life': row[14],
                    'auc': row[15],
                    'time_above_mec': row[16],
                    'time_above_mtc': row[17],
                    'interpretation': row[18],
                    'json_data': json.loads(row[19]) if row[19] else {}
                })
            
            return simulations
        except Exception as e:
            st.warning(f"Could not retrieve simulations: {str(e)}")
            return []
    
    def delete_simulation(self, sim_id: int) -> bool:
        """Delete a specific simulation."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM simulations WHERE id = ?", (sim_id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error deleting simulation: {str(e)}")
            return False
    
    def clear_all_simulations(self) -> bool:
        """Clear all simulation history."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM simulations")
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error clearing history: {str(e)}")
            return False


# ============================================================================
# PHARMACOKINETIC MODEL
# ============================================================================

class PharmacokineticModel:
    """Implements first-order elimination pharmacokinetic model."""
    
    @staticmethod
    def first_order_elimination(c0: float, k: float, t: np.ndarray) -> np.ndarray:
        """
        First-order elimination model: C(t) = C0 × exp(-k × t)
        
        Args:
            c0: Initial concentration
            k: Elimination rate constant
            t: Time array
        
        Returns:
            Concentration array
        """
        return c0 * np.exp(-k * t)
    
    @staticmethod
    def generate_time_points(start: float, end: float, num_points: int) -> np.ndarray:
        """Generate linearly spaced time points."""
        return np.linspace(start, end, num_points)
    
    @staticmethod
    def calculate_half_life(k: float) -> float:
        """Calculate half-life from elimination rate constant."""
        return np.log(2) / k
    
    @staticmethod
    def calculate_metrics(t: np.ndarray, c: np.ndarray, mec: Optional[float] = None, 
                         mtc: Optional[float] = None) -> Dict:
        """Calculate pharmacokinetic metrics using numpy.trapezoid for AUC."""
        
        cmax = np.max(c)
        tmax_idx = np.argmax(c)
        tmax = t[tmax_idx]
        
        # FIXED: Use np.trapezoid instead of deprecated np.trapz
        auc = np.trapezoid(c, t)
        
        time_above_mec = None
        time_above_mtc = None
        
        if mec is not None:
            above_mec = c >= mec
            if np.any(above_mec):
                idx_above = np.where(above_mec)[0]
                time_above_mec = t[idx_above[-1]] - t[idx_above[0]]
            else:
                time_above_mec = 0.0
        
        if mtc is not None:
            above_mtc = c > mtc
            if np.any(above_mtc):
                idx_above = np.where(above_mtc)[0]
                time_above_mtc = t[idx_above[-1]] - t[idx_above[0]]
            else:
                time_above_mtc = 0.0
        
        return {
            'cmax': cmax,
            'tmax': tmax,
            'auc': auc,
            'time_above_mec': time_above_mec,
            'time_above_mtc': time_above_mtc
        }


# ============================================================================
# INTERPRETATION ENGINE
# ============================================================================

class InterpretationEngine:
    """Generates clinical interpretations of simulation results."""
    
    @staticmethod
    def generate_interpretation(metrics: Dict, k: float, mec: Optional[float] = None,
                               mtc: Optional[float] = None, time_unit: str = "hours") -> str:
        """Generate clinical interpretation."""
        
        interpretation = []
        half_life = np.log(2) / k
        
        if k < 0.1:
            elimination_speed = "SLOW"
            elimination_desc = "This drug is eliminated slowly from the body. Long-term accumulation may occur with repeated dosing."
        elif k < 0.3:
            elimination_speed = "MODERATE"
            elimination_desc = "This drug is eliminated at a moderate rate. Multiple daily dosing may be appropriate."
        else:
            elimination_speed = "FAST"
            elimination_desc = "This drug is eliminated rapidly from the body. Frequent dosing may be necessary to maintain therapeutic levels."
        
        interpretation.append(f"**Elimination Rate: {elimination_speed}**\n{elimination_desc}")
        interpretation.append(f"\n**Half-Life: {half_life:.2f} {time_unit}**\nTime for concentration to reduce by 50%")
        interpretation.append(f"\n**Peak Concentration (Cmax): {metrics['cmax']:.3f}**\nHighest concentration reached during simulation")
        
        if mec is not None and mtc is not None:
            if metrics['time_above_mtc'] is not None:
                if metrics['time_above_mtc'] == 0:
                    interpretation.append(f"\n**THERAPEUTIC WINDOW: ACCEPTABLE** ✓\nConcentration remains below toxic threshold ({mtc}) throughout the simulation.")
                else:
                    interpretation.append(f"\n⚠️ **WARNING:** Concentration exceeds toxic threshold ({mtc}) for {metrics['time_above_mtc']:.2f} {time_unit}.\nDose adjustment may be required.")
        
        interpretation.append("\n**CLINICAL SUMMARY**\nThis simulation demonstrates first-order drug elimination kinetics. The exponential decay pattern is typical of drugs cleared by hepatic metabolism or renal excretion. Dosing intervals should be based on the half-life and therapeutic window.")
        
        return "\n".join(interpretation)


# ============================================================================
# VISUALIZATION ENGINE
# ============================================================================

class VisualizationEngine:
    """Creates professional clinical visualizations."""
    
    @staticmethod
    def create_concentration_plot(t: np.ndarray, c: np.ndarray, mec: Optional[float] = None,
                                 mtc: Optional[float] = None, label: str = "Drug Concentration",
                                 time_unit: str = "hours") -> go.Figure:
        """Create interactive concentration-time plot."""
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=t, y=c,
            mode='lines',
            name=label,
            line=dict(color='#1f77b4', width=3),
            hovertemplate='<b>Time:</b> %{x:.2f} ' + time_unit + '<br><b>Concentration:</b> %{y:.3f}<extra></extra>'
        ))
        
        if mec is not None:
            fig.add_hline(y=mec, line_dash="dash", line_color="#2ca02c",
                         annotation_text="MEC", annotation_position="right")
        
        if mtc is not None:
            fig.add_hline(y=mtc, line_dash="dash", line_color="#d62728",
                         annotation_text="MTC", annotation_position="right")
        
        fig.update_layout(
            title="<b>Drug Concentration vs Time - First-Order Elimination Model</b>",
            xaxis_title=f"Time ({time_unit})",
            yaxis_title="Concentration (mg/L)",
            hovermode='x unified',
            template='plotly_white',
            height=500,
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='rgba(240, 240, 240, 0.5)'
        )
        
        return fig
    
    @staticmethod
    def create_comparison_plot(time_data: Dict[str, np.ndarray], concentration_data: Dict[str, np.ndarray],
                              mec: Optional[float] = None, mtc: Optional[float] = None,
                              time_unit: str = "hours") -> go.Figure:
        """Create multi-curve comparison plot."""
        
        fig = go.Figure()
        colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for idx, (label, conc) in enumerate(concentration_data.items()):
            t = time_data[label]
            fig.add_trace(go.Scatter(
                x=t, y=conc,
                mode='lines',
                name=label,
                line=dict(color=colors_list[idx % len(colors_list)], width=2.5)
            ))
        
        if mec is not None:
            fig.add_hline(y=mec, line_dash="dash", line_color="#2ca02c",
                         annotation_text="MEC", annotation_position="right")
        
        if mtc is not None:
            fig.add_hline(y=mtc, line_dash="dash", line_color="#d62728",
                         annotation_text="MTC", annotation_position="right")
        
        fig.update_layout(
            title="<b>Comparative Pharmacokinetic Analysis - Multiple Elimination Rates</b>",
            xaxis_title=f"Time ({time_unit})",
            yaxis_title="Concentration (mg/L)",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig


# ============================================================================
# EXPORT ENGINE
# ============================================================================

class ExportEngine:
    """Handles CSV and PDF export functionality."""
    
    @staticmethod
    def export_csv(t: np.ndarray, concentration_data: Dict[str, np.ndarray]) -> bytes:
        """Export simulation data to CSV format."""
        
        df = pd.DataFrame({'Time': t})
        for label, c in concentration_data.items():
            df[label] = c
        
        return df.to_csv(index=False).encode('utf-8')
    
    @staticmethod
    def export_pdf(sim_name: str, params: Dict, metrics: Dict, 
                   interpretation: str) -> BytesIO:
        """Generate professional PDF report."""
        
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter,
                               rightMargin=0.5*inch, leftMargin=0.5*inch,
                               topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#003366'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#003366'),
            spaceAfter=8,
            spaceBefore=8
        )
        
        story.append(Paragraph("PHARMACOKINETICS SIMULATION REPORT", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph(f"<b>Simulation Name:</b> {sim_name}", styles['Normal']))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 0.15*inch))
        
        story.append(Paragraph("SIMULATION PARAMETERS", heading_style))
        
        params_data = [
            ['Parameter', 'Value'],
            ['Initial Concentration (C₀)', f"{params['c0']:.3f} {params.get('unit', 'mg/L')}"],
            ['Elimination Rate (k)', f"{params['k']:.4f}"],
            ['Start Time', f"{params['start_time']}"],
            ['End Time', f"{params['end_time']}"],
            ['Time Points', f"{params['time_points']}"],
        ]
        
        if params.get('mec'):
            params_data.append(['MEC', f"{params['mec']:.3f}"])
        if params.get('mtc'):
            params_data.append(['MTC', f"{params['mtc']:.3f}"])
        
        params_table = Table(params_data, colWidths=[2.5*inch, 3.5*inch])
        params_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(params_table)
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("CALCULATED METRICS", heading_style))
        
        metrics_data = [
            ['Metric', 'Value'],
            ['Maximum Concentration (Cmax)', f"{metrics['cmax']:.3f}"],
            ['Time to Maximum (Tmax)', f"{metrics['tmax']:.2f}"],
            ['Half-Life', f"{metrics['half_life']:.2f}"],
            ['Area Under Curve (AUC)', f"{metrics['auc']:.2f}"],
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2.5*inch, 3.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 0.3*inch))
        story.append(PageBreak())
        
        story.append(Paragraph("CLINICAL INTERPRETATION", heading_style))
        interpretation_text = interpretation.replace("**", "").replace("\n", "<br/>")
        story.append(Paragraph(interpretation_text, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        footer_text = "For research, training, and pharmacokinetic analysis support. Clinical application requires independent professional validation."
        story.append(Paragraph(f"<i>{footer_text}</i>", styles['Normal']))
        
        doc.build(story)
        pdf_buffer.seek(0)
        return pdf_buffer


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'current_interpretation' not in st.session_state:
    st.session_state.current_interpretation = ""

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application flow."""
    
    # Custom CSS
    st.markdown("""
    <style>
    .header-container {
        background: linear-gradient(135deg, #003366 0%, #0066cc 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .header-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #003366;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #003366;
        margin-top: 0.5rem;
    }
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #eee;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize components
    db = DatabaseManager()
    model = PharmacokineticModel()
    interpreter = InterpretationEngine()
    visualizer = VisualizationEngine()
    exporter = ExportEngine()
    
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🔬 Pharmacokinetics Dashboard</h1>
        <p class="header-subtitle">Professional Clinical Drug Concentration Simulation Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["🏠 HOME", "⚙️ SIMULATION", "📊 RESULTS", "💡 INTERPRETATION", "📚 HISTORY", "💾 EXPORT"]
    )
    
    # =====================================================================
    # HOME TAB
    # =====================================================================
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📋 Project Overview")
            st.markdown("""
            The **Pharmacokinetics Simulation Platform** provides clinical-grade simulation of drug concentration 
            dynamics in the bloodstream using the **first-order elimination model**.
            
            This application is designed for:
            - **Pharmacology Students** - Understanding drug kinetics
            - **Clinical Researchers** - Analyzing drug behavior
            - **Pharmacists** - Dosing optimization
            - **Medical Professionals** - Clinical decision support
            """)
        
        with col2:
            st.markdown("### ⚙️ Model Type")
            st.markdown("""
            **First-Order Elimination**
            
            **C(t) = C₀ × e^(-k×t)**
            """)
        
        st.divider()
        
        st.markdown("### 📚 Key Concepts")
        
        tabs = st.tabs(["Model Formula", "Key Parameters", "Clinical Thresholds"])
        
        with tabs[0]:
            st.markdown("""
            #### First-Order Elimination Model
            
            **Mathematical Formula:**
            ```
            C(t) = C₀ × exp(-k × t)
            ```
            
            **Half-Life Relationship:**
            ```
            t₁/₂ = ln(2) / k = 0.693 / k
            ```
            """)
        
        with tabs[1]:
            st.markdown("""
            #### Key Parameters
            
            | Parameter | Description |
            |-----------|-------------|
            | C₀ | Initial concentration |
            | k | Elimination rate constant |
            | t₁/₂ | Time for 50% reduction |
            | Cmax | Maximum concentration |
            | AUC | Total drug exposure |
            """)
        
        with tabs[2]:
            st.markdown("""
            #### Clinical Thresholds
            
            **MEC (Minimum Effective Concentration)** - Below this: no benefit\n
            **MTC (Minimum Toxic Concentration)** - Above this: toxicity risk\n
            **Therapeutic Window** - Between MEC and MTC
            """)
    
    # =====================================================================
    # SIMULATION TAB
    # =====================================================================
    with tab2:
        st.markdown("### 📊 Simulation Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Drug Parameters")
            c0 = st.number_input("Initial Concentration (C₀)", value=100.0, min_value=0.1, step=1.0)
            time_unit = st.selectbox("Time Unit", ["hours", "minutes", "days"])
            concentration_unit = st.selectbox("Concentration Unit", ["mg/L", "μg/mL", "ng/mL", "nmol/L"])
        
        with col2:
            st.subheader("Simulation Range")
            start_time = st.number_input("Start Time", value=0.0, min_value=0.0, step=0.1)
            end_time = st.number_input("End Time", value=24.0, min_value=0.1, step=1.0)
            time_points = st.number_input("Number of Time Points", value=500, min_value=10, step=10)
        
        st.divider()
        
        st.subheader("Simulation Mode")
        sim_mode = st.radio("Choose type:", ["Single Simulation", "Comparison Mode"], horizontal=True)
        
        st.divider()
        
        if sim_mode == "Single Simulation":
            st.subheader("Elimination Rate")
            k = st.number_input("Elimination Rate Constant (k)", value=0.1, min_value=0.001, max_value=1.0, step=0.01)
            
            col1, col2 = st.columns(2)
            with col1:
                half_life = np.log(2) / k
                st.metric("Half-Life", f"{half_life:.2f} {time_unit}")
            with col2:
                st.metric("Mode", "Single Curve")
            
            k_values = [k]
        
        else:
            st.subheader("Multiple Elimination Rates")
            k_input = st.text_input("Elimination Rate Constants (k)", value="0.05, 0.1, 0.2, 0.3")
            
            try:
                k_values = [float(x.strip()) for x in k_input.split(',')]
                k_values = [k for k in k_values if 0.001 <= k <= 1.0]
                
                if not k_values:
                    st.error("No valid k values in range [0.001, 1.0]")
                    k_values = [0.1]
                else:
                    st.success(f"✓ {len(k_values)} curves will be compared")
                
                half_lives = [np.log(2) / k for k in k_values]
                hl_df = pd.DataFrame({
                    'k value': [f"{k:.4f}" for k in k_values],
                    'Half-Life': [f"{hl:.2f} {time_unit}" for hl in half_lives]
                })
                st.dataframe(hl_df, use_container_width=True)
                
            except ValueError:
                st.error("Invalid input. Please enter numbers separated by commas.")
                k_values = [0.1]
        
        st.divider()
        
        st.subheader("Clinical Thresholds (Optional)")
        
        col1, col2 = st.columns(2)
        with col1:
            use_mec = st.checkbox("Use MEC", value=False)
            mec = None
            if use_mec:
                mec = st.number_input("MEC Value", value=10.0, min_value=0.0, step=1.0)
        
        with col2:
            use_mtc = st.checkbox("Use MTC", value=False)
            mtc = None
            if use_mtc:
                mtc = st.number_input("MTC Value", value=150.0, min_value=0.0, step=1.0)
        
        st.divider()
        
        st.subheader("Simulation Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            sim_name = st.text_input("Simulation Name", value=f"Simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        with col2:
            st.write("")
            run_button = st.button("▶️ Run Simulation", use_container_width=True, type="primary")
        
        if run_button:
            try:
                t = model.generate_time_points(start_time, end_time, int(time_points))
                
                results = {
                    'time': t,
                    'concentrations': {},
                    'metrics': {},
                    'parameters': {
                        'c0': c0,
                        'k_values': k_values,
                        'start_time': start_time,
                        'end_time': end_time,
                        'time_points': int(time_points),
                        'time_unit': time_unit,
                        'concentration_unit': concentration_unit,
                        'mec': mec,
                        'mtc': mtc,
                        'sim_mode': sim_mode,
                        'sim_name': sim_name
                    }
                }
                
                for k in k_values:
                    label = f"k = {k:.4f}"
                    c = model.first_order_elimination(c0, k, t)
                    results['concentrations'][label] = c
                    results['metrics'][label] = model.calculate_metrics(t, c, mec, mtc)
                    results['metrics'][label]['k'] = k
                    results['metrics'][label]['half_life'] = np.log(2) / k
                
                st.session_state.simulation_results = results
                st.success("✓ Simulation completed successfully!")
                st.info("Navigate to **RESULTS** tab to view graphs")
                
            except Exception as e:
                st.error(f"Simulation error: {str(e)}")
    
    # =====================================================================
    # RESULTS TAB
    # =====================================================================
    with tab3:
        if st.session_state.simulation_results is None:
            st.warning("⚠️ No simulation results. Run a simulation first.")
        else:
            results = st.session_state.simulation_results
            t = results['time']
            conc_data = results['concentrations']
            metrics = results['metrics']
            params = results['parameters']
            
            st.markdown("### 📈 Simulation Results")
            
            if params['sim_mode'] == "Single Simulation":
                label = list(conc_data.keys())[0]
                fig = visualizer.create_concentration_plot(
                    t, conc_data[label],
                    mec=params['mec'],
                    mtc=params['mtc'],
                    label=label,
                    time_unit=params['time_unit']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = visualizer.create_comparison_plot(
                    {label: t for label in conc_data.keys()},
                    conc_data,
                    mec=params['mec'],
                    mtc=params['mtc'],
                    time_unit=params['time_unit']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            st.markdown("### 📊 Pharmacokinetic Metrics")
            
            metrics_list = []
            for label, metric_dict in metrics.items():
                metrics_list.append({
                    'Parameter': label,
                    'Cmax': f"{metric_dict['cmax']:.3f}",
                    'Tmax': f"{metric_dict['tmax']:.2f}",
                    'Half-Life': f"{metric_dict['half_life']:.2f}",
                    'AUC': f"{metric_dict['auc']:.2f}"
                })
            
            metrics_df = pd.DataFrame(metrics_list)
            st.dataframe(metrics_df, use_container_width=True)
            
            st.divider()
            
            st.markdown("### 🎯 Key Metrics (Primary Curve)")
            
            primary_label = list(conc_data.keys())[0]
            primary_metrics = metrics[primary_label]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Maximum Concentration</div>
                    <div class="metric-value">{primary_metrics['cmax']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Time to Peak</div>
                    <div class="metric-value">{primary_metrics['tmax']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Half-Life</div>
                    <div class="metric-value">{primary_metrics['half_life']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Area Under Curve</div>
                    <div class="metric-value">{primary_metrics['auc']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            st.markdown("### 📋 Detailed Data")
            
            data_dict = {'Time': t.round(3)}
            for label, c in conc_data.items():
                data_dict[label] = c.round(4)
            
            data_df = pd.DataFrame(data_dict)
            st.dataframe(data_df, use_container_width=True)
    
    # =====================================================================
    # INTERPRETATION TAB
    # =====================================================================
    with tab4:
        if st.session_state.simulation_results is None:
            st.warning("⚠️ No simulation results. Run a simulation first.")
        else:
            results = st.session_state.simulation_results
            metrics = results['metrics']
            params = results['parameters']
            
            st.markdown("### 💡 Clinical Interpretation")
            st.markdown("---")
            
            primary_label = list(metrics.keys())[0]
            primary_metrics = metrics[primary_label]
            k_value = primary_metrics['k']
            
            interpretation = interpreter.generate_interpretation(
                primary_metrics,
                k_value,
                mec=params['mec'],
                mtc=params['mtc'],
                time_unit=params['time_unit']
            )
            
            st.session_state.current_interpretation = interpretation
            st.markdown(interpretation)
            
            st.divider()
            
            st.markdown("### 📖 Understanding the Results")
            
            tabs = st.tabs(["Elimination Kinetics", "Clinical Significance", "Therapeutic Window"])
            
            with tabs[0]:
                st.markdown(f"""
                #### Elimination Kinetics Analysis
                
                **Elimination Rate Constant (k): {k_value:.4f}**
                
                **Half-Life: {primary_metrics['half_life']:.2f} {params['time_unit']}**
                
                After each half-life:
                - 1 half-life: 50% remains
                - 2 half-lives: 25% remains
                - 3 half-lives: 12.5% remains
                - 5 half-lives: 3.1% remains (essentially cleared)
                """)
            
            with tabs[1]:
                st.markdown(f"""
                #### Clinical Significance
                
                **Peak Concentration (Cmax): {primary_metrics['cmax']:.3f}**
                
                **Area Under Curve (AUC): {primary_metrics['auc']:.2f}**
                
                AUC represents total drug exposure and correlates with therapeutic efficacy
                and risk of side effects.
                """)
            
            with tabs[2]:
                if params['mec'] and params['mtc']:
                    st.markdown(f"""
                    #### Therapeutic Window Analysis
                    
                    **MEC: {params['mec']}**\n
                    **MTC: {params['mtc']}**\n
                    **Therapeutic Index: {params['mtc'] / params['mec']:.2f}**
                    """)
                else:
                    st.info("Enable MEC and MTC in the Simulation tab for therapeutic window analysis.")
    
    # =====================================================================
    # HISTORY TAB
    # =====================================================================
    with tab5:
        st.markdown("### 📚 Simulation History")
        
        simulations = db.get_all_simulations()
        
        if not simulations:
            st.info("ℹ️ No saved simulations yet. Run a simulation and save it.")
        else:
            col1, col2 = st.columns([0.8, 0.2])
            with col2:
                if st.button("🗑️ Clear All", use_container_width=True):
                    if db.clear_all_simulations():
                        st.success("✓ History cleared")
                        st.rerun()
            
            st.divider()
            
            for sim in simulations:
                with st.expander(f"📊 {sim['simulation_name']} - {sim['timestamp']}", expanded=False):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.markdown("**Parameters:**")
                        st.markdown(f"""
                        - C₀: {sim['c0']}
                        - k: {sim['k_values']}
                        - Mode: {sim['simulation_mode']}
                        - Range: {sim['start_time']}-{sim['end_time']} {sim['time_unit']}
                        """)
                    
                    with col2:
                        st.markdown("**Metrics:**")
                        st.markdown(f"""
                        - Cmax: {sim['cmax']:.3f}
                        - Half-Life: {sim['half_life']:.2f}
                        - AUC: {sim['auc']:.2f}
                        """)
                    
                    with col3:
                        if st.button(f"🗑️", key=f"del_{sim['id']}", use_container_width=True):
                            if db.delete_simulation(sim['id']):
                                st.success("✓ Deleted")
                                st.rerun()
    
    # =====================================================================
    # EXPORT TAB
    # =====================================================================
    with tab6:
        if st.session_state.simulation_results is None:
            st.warning("⚠️ No simulation results to export.")
        else:
            results = st.session_state.simulation_results
            t = results['time']
            conc_data = results['concentrations']
            metrics = results['metrics']
            params = results['parameters']
            
            st.markdown("### 💾 Export Simulation Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📥 CSV Export")
                
                csv_bytes = exporter.export_csv(t, conc_data)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_bytes,
                    file_name=f"{params['sim_name']}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.info("CSV contains time-course data for all curves")
            
            with col2:
                st.subheader("📄 PDF Report")
                
                primary_label = list(conc_data.keys())[0]
                primary_metrics = metrics[primary_label]
                
                pdf_buffer = exporter.export_pdf(
                    params['sim_name'],
                    {
                        'c0': params['c0'],
                        'k': primary_metrics['k'],
                        'start_time': params['start_time'],
                        'end_time': params['end_time'],
                        'time_points': params['time_points'],
                        'time_unit': params['time_unit'],
                        'unit': params['concentration_unit'],
                        'mec': params['mec'],
                        'mtc': params['mtc']
                    },
                    {
                        'cmax': primary_metrics['cmax'],
                        'tmax': primary_metrics['tmax'],
                        'half_life': primary_metrics['half_life'],
                        'auc': primary_metrics['auc']
                    },
                    st.session_state.current_interpretation
                )
                
                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_buffer,
                    file_name=f"{params['sim_name']}_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
                st.info("PDF includes parameters, metrics, and interpretation")
            
            st.divider()
            
            st.subheader("💾 Save to History")
            
            if st.button("Save to History", use_container_width=True):
                primary_label = list(conc_data.keys())[0]
                primary_metrics = metrics[primary_label]
                
                sim_data = {
                    'timestamp': datetime.now().isoformat(),
                    'simulation_name': params['sim_name'],
                    'simulation_mode': params['sim_mode'],
                    'c0': params['c0'],
                    'k_values': str(params['k_values']),
                    'start_time': params['start_time'],
                    'end_time': params['end_time'],
                    'time_points': params['time_points'],
                    'time_unit': params['time_unit'],
                    'mec': params['mec'],
                    'mtc': params['mtc'],
                    'cmax': primary_metrics['cmax'],
                    'tmax': primary_metrics['tmax'],
                    'half_life': primary_metrics['half_life'],
                    'auc': primary_metrics['auc'],
                    'time_above_mec': primary_metrics.get('time_above_mec'),
                    'time_above_mtc': primary_metrics.get('time_above_mtc'),
                    'interpretation': st.session_state.current_interpretation,
                    'json_data': {
                        'concentrations': {k: v.tolist() for k, v in conc_data.items()},
                        'time': t.tolist()
                    }
                }
                
                if db.save_simulation(sim_data):
                    st.success("✓ Simulation saved")
                else:
                    st.error("✗ Save failed")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>
        ⚕️ <b>Professional Clinical Dashboard</b><br>
        For research, training, and pharmacokinetic analysis support.<br>
        Clinical application requires independent professional validation.<br><br>
        <small>© 2026 Pharmacokinetics Simulation Platform | Version 2.0.0</small>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()