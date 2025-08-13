# AcetaminophenPBPKModel.py

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline
from matplotlib.patches import Patch

class AcetaminophenPBPKModel:
    """
    Attributes:
        comp_names : [Lumen, Liver, Plasma, Urine, Metabolites]
            names of pharmacokinetic compartments

        rates : [MRP2,MRP3,OATP1B1,OATP1B3,CYP2E1,CYP1A2,CYP3A4,renal_Cl,delta,biliary_Cl,passive]
            kinetic rates of acetaminophen movement through model
        
            MRP2 : MRP2 efflux transporter  
                Liver -> Lumen
    
            MRP3 : MRP3 efflux transporter  
                Liver -> Lumen

            OATP1B1 : OATP1B1 uptake transporter  
                Lumen -> Liver

            OATP1B3 : OATP1B3 uptake transporter  
                Lumen -> Liver

            CYP2E1 : Phase I metabolism by CYP2E1 enzyme in liver  
                Liver -> Metabolites

            CYP1A2 : Phase I metabolism by CYP1A2 enzyme in liver  
                Liver -> Metabolites

            CYP3A4 : Phase I metabolism by CYP3A4 enzyme in liver  
                Liver -> Metabolites

            renal_Cl : Renal clearance  
                Plasma -> Urine

            delta : First-pass distribution  
                Liver -> Plasma

            biliary_Cl : Biliary clearance  
                Liver -> Lumen

            passive : Passive absorption  
                Lumen -> Liver
            
        

        exp_plasma_conc : experimental plasma concentrations of acetaminophen (micromolar) from Yong Yue, Agron Collaku, Dongzhou J. Liu 
        https://accp1.onlinelibrary.wiley.com/doi/10.1002/cpdd.367
            
        [0, 2.8, 5.8, 7.5, 9.2, 9.8, 10.0, 10.0, 10.0, 9.8, 9.5, 8.5, 8.0, 7.2, 6.5, 5.5, 4.8, 4.0, 3.2, 2.5, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8]
        
        init_comp_conc : initial amounts in each compartment (moles)
                [0.009925 , 0.0, 0.0, 0.0, 0.0]
                
        times : measurement time points (hours) from Yong Yue, Agron Collaku, Dongzhou J. Liu
                [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

                
    """

    def __init__(self):
        # Initialize attributes following SASP model structure
        self.rate_names = ['MRP2','MRP3','OATP1B1','OATP1B3','CYP2E1','CYP1A2','CYP3A4','renal_Cl','delta','biliary_Cl','passive']
        
        # Fixed: rates array now matches rate_names length (11 parameters)
        self.rates = np.array([0.4, 0.5, 1.0, 0.4, 0.015, 0.1, 1.8, 0.025, 0.1, 0.05, 0.2])
        
        # Experimental data for acetaminophen (placeholder - replace with actual data)
        self.exp_plasma_conc = np.array([0, 2.8, 5.8, 7.5, 9.2, 9.8, 10.0, 10.0, 10.0, 9.8, 9.5, 8.5, 8.0, 7.2, 6.5, 5.5, 4.8, 4.0, 3.2, 2.5, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8])
        
        self.init_comp_conc = np.array([0.00879, 0.0, 0.0, 0.0, 0.0])
        self.times = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        
        # Set plotting parameters
        plt.rcParams.update({'font.size': 12})
        plt.rcParams.update({'font.family': 'Times New Roman'})
        plt.rcParams.update({'axes.linewidth': 1})

    """
    UTILITY METHODS
    """
    
    def reimann_sum(self, x, y):
        """Calculate area under curve using Riemann sum"""
        return np.trapz(y, x)

    """
    PLOTTING METHODS
    """

    def plot_exp_data(self):
        """Plot experimental acetaminophen plasma data"""
        plt.plot(self.times, self.exp_plasma_conc, 'o', c='k', label='Experimental Data')
        plt.ylabel('Acetaminophen Plasma Concentration (μM)')
        plt.xlabel('Time (h)')
        plt.legend(loc='upper right')
        plt.show()

    def plot_rates(self):
        """Plot current rate parameters as bar chart"""
        x = [i for i in range(len(self.rates))]
        plt.bar(x, height=self.rates, color='k')
        plt.xticks(x, self.rate_names, rotation=45)
        for x_pos, y_pos in zip(x, self.rates):
            plt.text(x_pos - 0.3, y_pos + 0.005, str(round(y_pos, 3)))
        plt.ylabel('Rate (h⁻¹)')
        plt.tight_layout()
        plt.show()

    def plot_model(self, comp_no=2, title=''):
        """Plot model predictions vs experimental data"""
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        
        # Create smooth curve
        x, curve = self._get_curve(self.model(self.rates, comp_no=comp_no))
        ax.plot(x, curve, label='optimized\nmodel', ls='dashed', color='k')
        
        if comp_no == 0:
            met_x, met_curve = self._get_curve(self.model(self.rates, comp_no=4))
            ax.plot(met_x, met_curve, ls='dashdot', label='Metabolites', color='k')
            ax.set_ylabel('Acetaminophen Lumen\nConcentration [μM]')
        elif comp_no == 2:
            ax.plot(self.times, self.exp_plasma_conc, 'o', c='k', label='Experimental Data')
            ax.set_ylabel('Acetaminophen Plasma\nConcentration [μM]')
        elif comp_no == 3:
            ax.set_ylabel('micromoles of Acetaminophen')
            perc = curve[-1] / ((10**6) * self.init_comp_conc[0]) * 100
            print(f'Percent of acetaminophen excreted via urine: {round(perc, 3)}%')
        
        ax.set_xlabel('Time [h]')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_title(title)
        fig.tight_layout()
        plt.show()

    def _plot_knockout(self, knockout_rates, name):
        """Plot knockout comparison"""
        fig, ax = plt.subplots(figsize=(1.75, 1.75))
        
        x, curve = self._get_curve(self.model(self.rates, comp_no=2))
        ax.plot(x, curve, label='optimized\nmodel', color='k')
        ax.plot(self.times, self.exp_plasma_conc, 'o', c='k', label='Experimental Data')
        
        x, knockout_curve = self._get_curve(self.model(knockout_rates, comp_no=2))
        ax.plot(x, knockout_curve, label=name, color='k', ls='dashed')
        
        ax.set_xticks([0, 20, 40])
        fig.tight_layout()
        plt.show()

        # Calculate fold changes
        max_wt = curve.max()
        max_ko = knockout_curve.max()
        fold_change = np.abs(max_wt - max_ko) / max_wt
        print(f'PEAK CONC. FOLD CHANGE: {fold_change}')

        # AUC fold change
        auc_wt = self.reimann_sum(x, curve)
        auc_mut = self.reimann_sum(x, knockout_curve)
        fold_change = np.abs(float(auc_wt) - float(auc_mut)) / float(auc_wt)
        print(f'AUC FOLD CHANGE: {fold_change}')

    def _get_curve(self, model_output):
        """Generate smooth spline curve"""
        xy_spline = make_interp_spline(self.times, model_output)
        x = np.linspace(self.times.min(), self.times.max(), 500)
        curve = xy_spline(x)
        return [x, curve]

    """
    COMPARTMENTAL MODEL
    """
    def model(self, p, comp_no=2):
        """
        Acetaminophen PBPK model with differential equations

        p: parameters in shape [MRP2, MRP3, OATP1B1, OATP1B3, CYP2E1, CYP1A2, CYP3A4, renal_Cl, delta, biliary_Cl, passive]
        comp_no: indice of compartment to calculate concentration over time 
            0 : Lumen
            1 : Liver
            2 : Plasma
            3 : Urine 
            4 : Metabolites
            5 : Bile
        """

        def odes(y, t, p):
            """
            System of differential equations for acetaminophen PBPK model
            y: moles of APAP in each compartment [Lumen, Liver, Plasma, Urine, Metabolites, Bile]
            p: [MRP2, MRP3, OATP1B1, OATP1B3, CYP2E1, CYP1A2, CYP3A4, renal_Cl, delta, biliary_Cl, passive]
            """

            # Unpack compartments
            n_lumen, n_liver, n_plasma, n_urine, n_metabolites = y

            # Unpack parameters
            MRP2, MRP3, OATP1B1, OATP1B3, CYP2E1, CYP1A2, CYP3A4, renal_Cl, delta, biliary_Cl, passive = p

            # Total uptake transporters (liver uptake from lumen)
            total_uptake = OATP1B1 + OATP1B3

            # Total efflux transporters (liver to lumen)
            total_efflux = MRP2 + MRP3

            # Total metabolism in liver
            total_metabolism = CYP2E1 + CYP1A2 + CYP3A4

            # Differential equations based on the APAP model structure:

            # Lumen: 
            dldt = (total_efflux + biliary_Cl) * n_liver - (total_uptake + passive) * n_lumen


            # Liver: 
            drdt = (total_uptake + passive) * n_lumen - (total_efflux + total_metabolism + delta + biliary_Cl) * n_liver


            # Plasma:
            dpdt = delta*n_liver - renal_Cl*n_plasma

            # Urine: 
            dudt = renal_Cl * n_plasma

            # Metabolites: receives from liver metabolism
            dmdt = total_metabolism * n_liver

            return [dldt, drdt, dpdt, dudt, dmdt]

        # Return the solutions for one compartment 
        def get_comp_conc(y, comp_no):
            # Physiological Volumes for compartments (L)
            if comp_no == 0 or comp_no == 5:
                volume = 0.105 + .013 # small intestine, large intestine
            elif comp_no == 2:
                volume = 5 * 0.60 # Blood, plasma 
            else:
                volume = 1

            # Get concentration
            comp_conc = np.empty(len(y))
            for i, conc_t in enumerate(y):
                comp_conc[i] = conc_t[comp_no]*((10**6)/volume) # convert to micromolar
            return comp_conc

        # Get numerical solutions to ODEs
        sol = odeint(odes, t=self.times, y0=self.init_comp_conc, args=tuple([p]))

        return get_comp_conc(sol, comp_no)

    """
    OPTIMIZATION
    """

    def _obj(self, p):
        """Objective function for parameter fitting"""
        return np.sum((self.model(p, comp_no=2) - self.exp_plasma_conc)**2)

    def optimize(self, init_rates):
        """Optimize parameters to fit experimental data"""
        res = minimize(self._obj, x0=init_rates, 
                      bounds=[(0, np.inf) for i in range(len(init_rates))])
        self.rates = res.x
        self.plasma_conc = self.model(self.rates, comp_no=2)
        self.lumen_conc = self.model(self.rates, comp_no=0)
        self.plot_rates()
        return res

    """
    VALIDATION
    """

    def knockout(self, rates=[0]):
        """
        rates: array of rate indices to knockout, indices can be found in self.rates_names
        """

        # Make temp rates
        knockout_rates = [r for r in self.rates]
        for r in rates:
            knockout_rates[r] = 0
        
        name = ' '.join(self.rate_names[i] for i in rates) + ' knockout'

        self._plot_knockout(knockout_rates, name)

    def plot_combination_knockout_analysis(self, gene1_idx, gene2_idx, gene1_name, gene2_name):
        """
        Plot comprehensive combination knockout analysis for two genes
        
        Parameters:
        gene1_idx, gene2_idx: indices of genes to knockout
        gene1_name, gene2_name: display names for the genes
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        
        print(f"COMBINATION KNOCKOUT ANALYSIS: {gene1_name} + {gene2_name}")
        print("=" * 70)
        print(f"Analyzing the combined effect of simultaneously knocking out both genes...")

        # Normal model simulation
        normal_plasma = self.model(self.rates, comp_no=2)
        normal_auc = np.trapz(normal_plasma, self.times)
        normal_cmax = np.max(normal_plasma)

        # Individual knockout simulations
        # Gene 1 knockout
        gene1_knockout_rates = self.rates.copy()
        gene1_knockout_rates[gene1_idx] = 0
        gene1_plasma = self.model(gene1_knockout_rates, comp_no=2)
        gene1_auc = np.trapz(gene1_plasma, self.times)
        gene1_cmax = np.max(gene1_plasma)
        gene1_auc_change = (gene1_auc - normal_auc) / normal_auc * 100
        gene1_cmax_change = (gene1_cmax - normal_cmax) / normal_cmax * 100

        # Gene 2 knockout
        gene2_knockout_rates = self.rates.copy()
        gene2_knockout_rates[gene2_idx] = 0
        gene2_plasma = self.model(gene2_knockout_rates, comp_no=2)
        gene2_auc = np.trapz(gene2_plasma, self.times)
        gene2_cmax = np.max(gene2_plasma)
        gene2_auc_change = (gene2_auc - normal_auc) / normal_auc * 100
        gene2_cmax_change = (gene2_cmax - normal_cmax) / normal_cmax * 100

        # Combination knockout
        combo_knockout_rates = self.rates.copy()
        combo_knockout_rates[gene1_idx] = 0
        combo_knockout_rates[gene2_idx] = 0
        combo_knockout_plasma = self.model(combo_knockout_rates, comp_no=2)
        combo_knockout_auc = np.trapz(combo_knockout_plasma, self.times)
        combo_knockout_cmax = np.max(combo_knockout_plasma)
        combo_auc_change = (combo_knockout_auc - normal_auc) / normal_auc * 100
        combo_cmax_change = (combo_knockout_cmax - normal_cmax) / normal_cmax * 100

        # Calculate additive effects
        expected_additive_auc = gene1_auc_change + gene2_auc_change
        expected_additive_cmax = gene1_cmax_change + gene2_cmax_change

        # Calculate synergy/antagonism
        auc_synergy = combo_auc_change - expected_additive_auc
        cmax_synergy = combo_cmax_change - expected_additive_cmax

        # Create comprehensive visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Time course comparison
        ax1.plot(self.times, normal_plasma, 'k-', linewidth=3, label='Normal (Baseline)', alpha=0.8)
        ax1.plot(self.times, gene1_plasma, 'b--', linewidth=2, 
                 label=f'{gene1_name} KO only', alpha=0.8)
        ax1.plot(self.times, gene2_plasma, 'g--', linewidth=2, 
                 label=f'{gene2_name} KO only', alpha=0.8)
        ax1.plot(self.times, combo_knockout_plasma, 'r-', linewidth=3, 
                 label=f'{gene1_name} + {gene2_name} KO', alpha=0.8)
        ax1.scatter(self.times, self.exp_plasma_conc, color='purple', s=60, alpha=0.7, 
                    label='Experimental Data', zorder=5, edgecolors='white', linewidth=1)

        ax1.set_title('Combination Knockout: Time Course Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (hours)', fontsize=12)
        ax1.set_ylabel('Plasma APAP (μM)', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. AUC comparison bar chart
        conditions = ['Normal', f'{gene1_name}\nKO', f'{gene2_name}\nKO', f'{gene1_name}+{gene2_name}\nKO', 'Expected\nAdditive']
        auc_values = [normal_auc, 
                      normal_auc * (1 + gene1_auc_change/100),
                      normal_auc * (1 + gene2_auc_change/100),
                      combo_knockout_auc,
                      normal_auc * (1 + expected_additive_auc/100)]
        colors = ['black', 'blue', 'green', 'red', 'orange']

        bars = ax2.bar(conditions, auc_values, color=colors, alpha=0.7, edgecolor='white', linewidth=1)
        ax2.set_title('AUC Comparison: Individual vs Combination', fontsize=14, fontweight='bold')
        ax2.set_ylabel('AUC (μM⋅h)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, auc_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{value:.1f}', ha='center', va='bottom', fontsize=10)

        # 3. Effect magnitude comparison
        effect_categories = [f'{gene1_name}\nAUC', f'{gene2_name}\nAUC', 'Combination\nAUC', 'Expected\nAdditive']
        effect_values = [gene1_auc_change, gene2_auc_change, combo_auc_change, expected_additive_auc]
        effect_colors = ['blue', 'green', 'red', 'orange']

        bars3 = ax3.bar(effect_categories, effect_values, color=effect_colors, alpha=0.7, edgecolor='white', linewidth=1)
        ax3.set_title('Effect Magnitude Comparison (% Change)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('AUC Change (%)', fontsize=12)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, value in zip(bars3, effect_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (2 if height > 0 else -4),
                     f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

        # 4. Synergy analysis
        synergy_data = [gene1_auc_change, gene2_auc_change, expected_additive_auc, combo_auc_change, auc_synergy]
        synergy_labels = [f'{gene1_name}\nIndividual', f'{gene2_name}\nIndividual', 'Expected\nAdditive', 'Observed\nCombination', 'Synergy\nEffect']
        synergy_colors = ['blue', 'green', 'orange', 'red', 'purple']

        bars4 = ax4.bar(synergy_labels, synergy_data, color=synergy_colors, alpha=0.7, edgecolor='white', linewidth=1)
        ax4.set_title('Synergy Analysis (% Change)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('AUC Change (%)', fontsize=12)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, value in zip(bars4, synergy_data):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + (2 if height > 0 else -4),
                     f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

        plt.tight_layout()
        plt.show()

        # Detailed results summary
        print("\n" + "=" * 70)
        print("COMBINATION KNOCKOUT RESULTS:")
        print("=" * 70)

        print(f"Individual Effects:")
        print(f"  • {gene1_name} knockout alone:     AUC {gene1_auc_change:+.1f}%, Cmax {gene1_cmax_change:+.1f}%")
        print(f"  • {gene2_name} knockout alone:     AUC {gene2_auc_change:+.1f}%, Cmax {gene2_cmax_change:+.1f}%")

        print(f"\nCombination Effect:")
        print(f"  • {gene1_name} + {gene2_name} knockout:  AUC {combo_auc_change:+.1f}%, Cmax {combo_cmax_change:+.1f}%")

        print(f"\nExpected vs Observed:")
        print(f"  • Expected additive (AUC):   {expected_additive_auc:+.1f}%")
        print(f"  • Observed combination (AUC): {combo_auc_change:+.1f}%")
        print(f"  • Synergy/Antagonism (AUC):   {auc_synergy:+.1f}%")

        print(f"\n  • Expected additive (Cmax):   {expected_additive_cmax:+.1f}%")
        print(f"  • Observed combination (Cmax): {combo_cmax_change:+.1f}%")
        print(f"  • Synergy/Antagonism (Cmax):   {cmax_synergy:+.1f}%")

        # Interpretation
        print(f"\n" + "=" * 70)
        print("BIOLOGICAL INTERPRETATION:")
        print("=" * 70)

        if abs(auc_synergy) < 5:
            auc_interaction = "Additive (no significant interaction)"
        elif auc_synergy > 5:
            auc_interaction = "Synergistic (greater than additive effect)"
        else:
            auc_interaction = "Antagonistic (less than additive effect)"

        if abs(cmax_synergy) < 5:
            cmax_interaction = "Additive (no significant interaction)"
        elif cmax_synergy > 5:
            cmax_interaction = "Synergistic (greater than additive effect)"
        else:
            cmax_interaction = "Antagonistic (less than additive effect)"

        print(f"• AUC Interaction Type: {auc_interaction}")
        print(f"• Cmax Interaction Type: {cmax_interaction}")

        print(f"\nMechanistic Insights:")
        print(f"• {gene1_name} and {gene2_name} are both involved in drug metabolism/transport")
        print(f"• Their simultaneous knockout affects acetaminophen clearance")
        print(f"• Combination effect magnitude: {combo_auc_change:.1f}% AUC change")

        if combo_auc_change > 100:
            safety_concern = "HIGH - Major increase in drug exposure"
        elif combo_auc_change > 50:
            safety_concern = "MODERATE - Significant increase in drug exposure"
        elif combo_auc_change > 20:
            safety_concern = "LOW-MODERATE - Noticeable increase in drug exposure"
        else:
            safety_concern = "MINIMAL - Small change in drug exposure"

        print(f"• Clinical Relevance: {safety_concern}")

        # Return results for further analysis
        return {
            'gene1_auc_change': gene1_auc_change,
            'gene2_auc_change': gene2_auc_change,
            'combo_auc_change': combo_auc_change,
            'expected_additive_auc': expected_additive_auc,
            'auc_synergy': auc_synergy,
            'interaction_type': auc_interaction
        }

    """
    MICROGRAVITY INFLUENCE
    """
    def simulate_microgravity(self, gene_csv_path, n_samples=1000, comp_no=2):
        """
        Simulate APAP kinetics under microgravity by propagating gene expression uncertainty.
        Enhanced: Uses optimized rates for 'Normal' and applies gene fold changes for microgravity.
        """
        import seaborn as sns

        # Map model parameters to gene symbols (UPPERCASE for consistency)
        param_gene_map = {
            'MRP2': 'ABCC2',
            'MRP3': 'ABCC3',
            'OATP1B1': 'SLCO1B1',
            'OATP1B3': 'SLCO1B3',
            'CYP2E1': 'CYP2E1',
            'CYP1A2': 'CYP1A2',
            'CYP3A4': 'CYP3A4'
        }
        param_indices = {name: idx for idx, name in enumerate(self.rate_names)}

        # Read gene expression data
        df = pd.read_csv(gene_csv_path)
        df['Gene'] = df['Gene'].str.upper()

        # Generate log2FC distributions for each gene
        log2fc_dists = {}
        for param, gene in param_gene_map.items():
            row = df[df['Gene'] == gene]
            if not row.empty:
                mean = row.iloc[0]['Mean_Log2fc']
                std = 0.1
                log2fc_dists[param] = np.random.normal(mean, std, n_samples)
            else:
                log2fc_dists[param] = np.zeros(n_samples)

        # Simulate model for each sample using optimized rates as baseline
        timeseries = np.empty((n_samples, len(self.times)))
        for i in range(n_samples):
            rates = self.rates.copy()  # self.rates should be optimized before calling this function!
            for param, idx in param_indices.items():
                if param in log2fc_dists:
                    fc = 2 ** log2fc_dists[param][i]
                    rates[idx] *= fc
            timeseries[i, :] = self.model(rates, comp_no=comp_no)

        # Calculate statistics
        means = np.mean(timeseries, axis=0)
        stds = np.std(timeseries, axis=0)
        lower = np.percentile(timeseries, 2.5, axis=0)
        upper = np.percentile(timeseries, 97.5, axis=0)

        # Calculate PK parameters using optimized rates for 'Normal'
        normal = self.model(self.rates, comp_no=comp_no)
        auc_normal = np.trapz(normal, self.times)
        auc_microgravity = np.trapz(means, self.times)
        cmax_normal = np.max(normal)
        cmax_microgravity = np.max(means)
        percent_cmax = 100 * (float(cmax_microgravity) - float(cmax_normal)) / float(cmax_normal)
        percent_auc = 100 * (float(auc_microgravity) - float(auc_normal)) / float(auc_normal)

        # --- Gene fold change comparison: Normal (optimized) vs Microgravity (optimized * 2^log2FC) ---
        gene_comparison = []
        for param, gene in param_gene_map.items():
            idx = param_indices[param]
            row = df[df['Gene'] == gene]
            normal_val = self.rates[idx]  # Use optimized value
            if not row.empty:
                log2fc = row.iloc[0]['Mean_Log2fc']
                fold_change = 2 ** log2fc
                micro_val = normal_val * fold_change
            else:
                log2fc = 0.0
                fold_change = 1.0
                micro_val = normal_val
            gene_comparison.append({
                'Parameter': param,
                'Gene': gene,
                'Normal': normal_val,
                'Microgravity': micro_val,
                'log2FC': log2fc
            })

        gene_comp_df = pd.DataFrame(gene_comparison)

        print('\nGene Parameter Value Comparison (Normal vs Microgravity, using optimized rates):')
        print(gene_comp_df[['Parameter', 'Gene', 'Normal', 'Microgravity', 'log2FC']])

        plt.figure(figsize=(8,5))
        bar_width = 0.35
        indices = np.arange(len(gene_comp_df))
        plt.bar(indices - bar_width/2, gene_comp_df['Normal'], bar_width, label='Normal (Optimized)', color='grey')
        plt.bar(indices + bar_width/2, gene_comp_df['Microgravity'], bar_width, label='Microgravity', color='blue')
        # Fixed: Convert DataFrame column to list for xticks
        plt.xticks(indices, gene_comp_df['Gene'].tolist(), rotation=45)
        plt.ylabel('Optimized Value / Fold Change')
        plt.title('Gene Parameter Value: Normal (Optimized) vs Microgravity')
        plt.legend()
        plt.tight_layout()
        plt.show()

                # --- PK Curve Comparison Plot ---
        plt.figure(figsize=(9,5))

        # Normal gravity (optimized)
        plt.plot(self.times, normal, label='Normal Gravity (Optimized)', color='black', linewidth=2)

        # Microgravity (mean)
        plt.plot(self.times, means, label='Microgravity (Mean)', color='#1565C0', linewidth=2)

        # Microgravity 95% CI
        plt.fill_between(self.times, lower, upper, color='#90caf9', alpha=0.5, label='Microgravity 95% CI')

        # Experimental data (smaller, lighter markers)
        plt.scatter(self.times, self.exp_plasma_conc, color='red', s=35, label='Experimental Data', zorder=5, edgecolor='k', linewidth=0.5)

        # Annotate Cmax and Tmax directly
        tmax_normal = self.times[np.argmax(normal)]
        tmax_micro = self.times[np.argmax(means)]
        plt.scatter([tmax_normal], [cmax_normal], color='black', marker='^', s=90, zorder=6)
        plt.scatter([tmax_micro], [cmax_microgravity], color='#1565C0', marker='^', s=90, zorder=6)
        plt.text(tmax_normal+0.2, cmax_normal+0.2, f'Cmax: {cmax_normal:.2f}\nTmax: {tmax_normal:.1f}h', color='black', fontsize=11)
        plt.text(tmax_micro+0.2, cmax_microgravity+0.2, f'Cmax: {cmax_microgravity:.2f}\nTmax: {tmax_micro:.1f}h', color='#1565C0', fontsize=11)

        # Labels and title
        plt.xlabel('Time (h)')
        plt.ylabel('Plasma APAP (μM)')
        plt.title('APAP Plasma Concentration: Normal vs Microgravity')

        # Clean legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=11, frameon=False)

        plt.tight_layout()
        plt.show()


        plt.figure(figsize=(8,5))

        # Plot mean curve for microgravity with error bars (std dev)
        plt.errorbar(self.times, means, yerr=stds, fmt='o-', color='blue', ecolor='lightblue', 
             elinewidth=2, capsize=5, label='Microgravity (mean ± SD)')

        # Plot mean curve for normal gravity (optimized)
        plt.plot(self.times, normal, 's-', color='black', label='Normal Gravity (Optimized)')

        # Plot experimental data points
        plt.scatter(self.times, self.exp_plasma_conc, color='red', marker='D', label='Experimental Data')

        # Annotate sample size
        plt.text(self.times[-1]*0.7, max(means)*0.9, 'n = {}'.format(n_samples), fontsize=10, color='blue')

        plt.xlabel('Time [h]')
        plt.ylabel('Plasma APAP Concentration [μM]')
        plt.title('APAP Plasma Concentration: Normal vs Microgravity')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        # Calculate Cmax and AUC for each simulation
        cmax_samples = np.max(timeseries, axis=1)
        auc_samples = np.trapz(timeseries, self.times, axis=1)
        # Plot histograms

        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.hist(cmax_samples, bins=30, color='blue', alpha=0.7)
        plt.xlabel('Cmax (μM)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Cmax (Microgravity)')
        plt.subplot(1,2,2)
        plt.hist(auc_samples, bins=30, color='green', alpha=0.7)
        plt.xlabel('AUC')
        plt.ylabel('Frequency')
        plt.title('Distribution of AUC (Microgravity)')
        plt.tight_layout()
        plt.show()

        # --- PRINT SUMMARY ---
        print(f'Cmax Normal: {cmax_normal:.2f} at {tmax_normal:.1f}h, Microgravity (mean): {cmax_microgravity:.2f} at {tmax_micro:.1f}h')
        print(f'AUC Normal: {auc_normal:.2f}, Microgravity (mean): {auc_microgravity:.2f}')
        print(f'Percent change in Cmax: {percent_cmax:.1f}%')
        print(f'Percent change in AUC: {percent_auc:.1f}%')

        # --- Bar graph for percentage changes in gene expression ---
        percent_gene_change = 100 * (gene_comp_df['Microgravity'] - gene_comp_df['Normal']) / gene_comp_df['Normal']
        plt.figure(figsize=(8,5))
        # Fixed: Convert DataFrame column to list for bar plot
        plt.bar(gene_comp_df['Gene'].tolist(), percent_gene_change, color='Green', alpha=0.7)
        plt.ylabel('Percent Change in Expression (%)')
        plt.xlabel('Gene')
        plt.title('Percent Change in Gene Expression (Microgravity vs Normal)')
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.tight_layout()
        plt.show()

        # --- Compartment-wise plots after microgravity influence ---
        comp_names = getattr(self, "comp_names", ["Lumen", "Liver", "Plasma", "Urine", "Metabolites", "Bile"])
        n_compartments = min(len(comp_names), len(self.init_comp_conc))

        for comp_no in range(n_compartments):
            # Simulate microgravity for this compartment
            comp_timeseries = np.empty((n_samples, len(self.times)))
            for i in range(n_samples):
                rates = self.rates.copy()
                for param, idx in param_indices.items():
                    if param in log2fc_dists:
                        fc = 2 ** log2fc_dists[param][i]
                        rates[idx] *= fc
                comp_timeseries[i, :] = self.model(rates, comp_no=comp_no)
            comp_means = np.mean(comp_timeseries, axis=0)
            comp_lower = np.percentile(comp_timeseries, 2.5, axis=0)
            comp_upper = np.percentile(comp_timeseries, 97.5, axis=0)
            comp_normal = self.model(self.rates, comp_no=comp_no)

            plt.figure(figsize=(7,4))
            plt.plot(self.times, comp_normal, label='Normal Gravity', color='k')
            plt.plot(self.times, comp_means, label='Microgravity (mean)', color='b')
            plt.fill_between(self.times, comp_lower, comp_upper, color='b', alpha=0.2, label='Microgravity 95% CI')
            plt.xlabel('Time (h)')
            plt.ylabel(f'{comp_names[comp_no]} Concentration (μM)')
            plt.title(f'{comp_names[comp_no]}: Normal vs Microgravity')
            plt.legend()
            plt.tight_layout()
            plt.show()
        return means, stds, lower, upper

    def plot_all_knockouts_analysis(self, genes_to_analyze=None):
        """
        Plot comprehensive analysis of all individual gene knockouts in a single graph
        
        Parameters:
        genes_to_analyze: dict of gene names to indices, if None uses default key genes
        """
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Default genes to analyze if none provided
        if genes_to_analyze is None:
            genes_to_analyze = {
                'MRP2 (ABCC2)': 0,
                'MRP3 (ABCC3)': 1,
                'OATP1B1 (SLCO1B1)': 2,
                'OATP1B3 (SLCO1B3)': 3,
                'CYP2E1': 4,
                'CYP1A2': 5,
                'CYP3A4': 6
            }
        
        print("COMPREHENSIVE KNOCKOUT ANALYSIS")
        print("=" * 70)
        print("Analyzing individual knockout effects for all key genes...")
        
        # Normal model simulation
        normal_plasma = self.model(self.rates, comp_no=2)
        normal_auc = np.trapz(normal_plasma, self.times)
        normal_cmax = np.max(normal_plasma)
        
        # Set up the comprehensive plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Colors for different knockouts
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
        
        # Store results for analysis
        knockout_results = {}
        auc_changes = []
        cmax_changes = []
        gene_names = []
        
        # Plot normal baseline
        ax1.plot(self.times, normal_plasma, 'k-', linewidth=3, label='Normal (Baseline)', alpha=0.8)
        ax1.scatter(self.times, self.exp_plasma_conc, color='gray', s=40, alpha=0.7, 
                   label='Experimental Data', zorder=5, edgecolors='white', linewidth=1)
        
        # Analyze each knockout
        for i, (gene_name, gene_idx) in enumerate(genes_to_analyze.items()):
            # Create knockout rates
            knockout_rates = self.rates.copy()
            knockout_rates[gene_idx] = 0
            
            # Simulate knockout
            knockout_plasma = self.model(knockout_rates, comp_no=2)
            knockout_auc = np.trapz(knockout_plasma, self.times)
            knockout_cmax = np.max(knockout_plasma)
            
            # Calculate changes
            auc_change = (knockout_auc - normal_auc) / normal_auc * 100
            cmax_change = (knockout_cmax - normal_cmax) / normal_cmax * 100
            
            # Store results
            knockout_results[gene_name] = {
                'auc_change': auc_change,
                'cmax_change': cmax_change,
                'plasma_profile': knockout_plasma
            }
            
            auc_changes.append(auc_change)
            cmax_changes.append(cmax_change)
            gene_names.append(gene_name.replace(' (', '\n(').replace(')', ')'))
            
            # Plot time course
            color = colors[i % len(colors)]
            ax1.plot(self.times, knockout_plasma, '--', linewidth=2, 
                    label=f'{gene_name} KO', alpha=0.8, color=color)
        
        # Configure time course plot
        ax1.set_title('All Gene Knockouts: Time Course Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (hours)', fontsize=12)
        ax1.set_ylabel('Plasma APAP (μM)', fontsize=12)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # AUC changes bar plot
        bars2 = ax2.bar(range(len(gene_names)), auc_changes, 
                       color=colors[:len(gene_names)], alpha=0.7, edgecolor='white', linewidth=1)
        ax2.set_title('AUC Changes by Gene Knockout', fontsize=14, fontweight='bold')
        ax2.set_ylabel('AUC Change (%)', fontsize=12)
        ax2.set_xlabel('Gene', fontsize=12)
        ax2.set_xticks(range(len(gene_names)))
        ax2.set_xticklabels(gene_names, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels on AUC bars
        for bar, value in zip(bars2, auc_changes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (2 if height > 0 else -4),
                    f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        # Cmax changes bar plot
        bars3 = ax3.bar(range(len(gene_names)), cmax_changes, 
                       color=colors[:len(gene_names)], alpha=0.7, edgecolor='white', linewidth=1)
        ax3.set_title('Cmax Changes by Gene Knockout', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Cmax Change (%)', fontsize=12)
        ax3.set_xlabel('Gene', fontsize=12)
        ax3.set_xticks(range(len(gene_names)))
        ax3.set_xticklabels(gene_names, rotation=45, ha='right')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels on Cmax bars
        for bar, value in zip(bars3, cmax_changes):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (2 if height > 0 else -4),
                    f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        # Effect magnitude comparison (bubble plot)
        for i, (auc, cmax) in enumerate(zip(auc_changes, cmax_changes)):
            ax4.scatter(auc, cmax, s=150, alpha=0.7, color=colors[i], 
                       edgecolors='white', linewidth=2)
            ax4.annotate(gene_names[i], (auc, cmax), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8, ha='left')
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_xlabel('AUC Change (%)', fontsize=12)
        ax4.set_ylabel('Cmax Change (%)', fontsize=12)
        ax4.set_title('AUC vs Cmax Changes', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary table
        print("\n" + "=" * 80)
        print("KNOCKOUT ANALYSIS SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Gene Name':<20} {'AUC Change (%)':<15} {'Cmax Change (%)':<16} {'Effect Magnitude'}")
        print("-" * 80)
        
        for gene_name, auc_change, cmax_change in zip(genes_to_analyze.keys(), auc_changes, cmax_changes):
            # Determine effect magnitude
            max_change = max(abs(auc_change), abs(cmax_change))
            if max_change > 50:
                magnitude = "Large"
            elif max_change > 20:
                magnitude = "Moderate"
            elif max_change > 5:
                magnitude = "Small"
            else:
                magnitude = "Minimal"
            
            print(f"{gene_name:<20} {auc_change:>+8.2f}     {cmax_change:>+8.2f}      {magnitude}")
        
        print("=" * 80)
        
        return knockout_results

