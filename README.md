# Acetaminophen PBPK Model under Microgravity Conditions

## 📌 Overview
This repository implements a **Physiologically-Based Pharmacokinetic (PBPK) model** for **acetaminophen (APAP)**, focusing on how **microgravity-induced gene expression changes** influence drug absorption, distribution, metabolism, and excretion (ADME).  
The model integrates:
- **ODE-based compartmental pharmacokinetics**
- **Transporter & enzyme kinetics**
- **Knockout analyses**
- **Microgravity simulation using gene expression log2FC data**

---

## 🧠 Features
- **Compartment Model**: Lumen, Liver, Plasma, Urine, Metabolites (with optional Bile)
- **Parameter Mapping**: Links kinetic rates to key transporters & enzymes (MRP2, MRP3, OATP1B1, OATP1B3, CYP2E1, CYP1A2, CYP3A4)
- **Model Optimization**: Fits model rates to experimental plasma concentration data
- **Knockout Simulations**: Single and combination gene knockouts with AUC & Cmax analysis
- **Microgravity Simulation**: Incorporates gene expression changes (log2FC) into kinetic rates and simulates uncertainty with Monte Carlo sampling
- **Visualization**: Generates publication-ready plots for concentration-time curves, parameter changes, and distribution histograms

---

## 📂 Repository Structure
├── APAPKineticModel.py # Core PBPK model implementation
├── Acetominophen.ipynb # Jupyter notebook for running simulations & analyses
├── Gene_expression_APAP_processed.csv # Processed gene expression log2FC data
└── README.md # Project documentation

pip install -r requirements.txt


from APAPKineticModel import AcetaminophenPBPKModel

model = AcetaminophenPBPKModel()
result = model.optimize(model.rates)
model.plot_model(comp_no=2, title='Plasma Concentration')
means, stds, lower, upper = model.simulate_microgravity(
    gene_csv_path="Gene_expression_APAP_processed.csv",
    n_samples=1000
)
model.knockout(rates=[0])  # Knockout MRP2
model.plot_all_knockouts_analysis()
model.knockout(rates=[0])  # Knockout MRP2
model.plot_all_knockouts_analysis()



