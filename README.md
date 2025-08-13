# Acetaminophen Pharmacokinetics in Microgravity

This project validates microgravity-induced changes in acetaminophen pharmacokinetics using a physiologically based pharmacokinetic (PBPK) model and knockout analysis. The analysis is based on gene expression data from NASA's GeneLab studies (GLDS-13 and GLDS-52).

## Features

* **PBPK Modeling**: A multi-compartment model simulating the absorption, distribution, metabolism, and excretion of acetaminophen.
* **Parameter Optimization**: The model is optimized against experimental data to establish baseline pharmacokinetic parameters.
* **Gene Expression Analysis**: Analysis of gene expression changes from microgravity studies to inform the model.
* **Knockout Validation**: In-silico knockout of key transporters and enzymes to validate their individual contributions to pharmacokinetic changes.

## Getting Started

### Prerequisites

* Python 3.x
* Jupyter Notebook
* NumPy
* Matplotlib
* Pandas
* SciPy
* Seaborn

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    ```
2.  Install the required packages:
    ```bash
    pip install numpy matplotlib pandas scipy seaborn
    ```

## Usage

The primary analyses are conducted in the Jupyter notebooks: `Acetominophen.ipynb` and `APAP_model.ipynb`.

1.  **Model Initialization and Optimization**:
    Open and run the `APAP_model.ipynb` notebook to initialize the PBPK model and optimize it against experimental data.

2.  **Gene Expression and Knockout Analysis**:
    The `Acetominophen.ipynb` notebook contains the analysis of gene expression data and the knockout validation experiments.

## File Descriptions

* `Acetominophen.ipynb`: Jupyter notebook for gene expression data analysis and knockout validation.
* `APAP_model.ipynb`: Jupyter notebook for PBPK model initialization and optimization.
* `APAPKineticModel.py`: Python script containing the `AcetaminophenPBPKModel` class.
* `Gene_expression_APAP_processed.csv`: Processed gene expression data from microgravity studies.

## Results

The PBPK model was successfully optimized against experimental data, establishing a baseline for acetaminophen pharmacokinetics. The knockout analysis of key transporters and enzymes provided insights into their individual contributions to the observed changes under microgravity conditions.

## Visualizations

The notebooks generate several plots to visualize the results, including:

* Acetaminophen plasma concentration curves (model vs. experimental data)
* Bar charts of optimized rate parameters
* Visualizations of gene expression changes (Log2 fold change and p-values)
* Knockout analysis comparison plots

## Acknowledgments

* **Data Source**: NASA's GeneLab (studies GLDS-13 and GLDS-52).
* **Experimental Data**: Yong Yue, Agron Collaku, Dongzhou J. Liu, et al. for the experimental plasma concentration data.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss any changes.
