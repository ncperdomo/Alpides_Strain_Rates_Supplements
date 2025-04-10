[![Language](https://img.shields.io/badge/python-3%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/ncperdomo/Alpides_Strain_Rates_Supplements/blob/main/LICENSE)
---
#  <p align=center> **Supplementary material for "Strain Rates along the Alpine-Himalayan Belt from a Comprehensive GNSS Velocity Field"** </p>

###  <p align=center> Journal of Geophysical Research: Solid Earth </p>
###  <p align=center> N. Castro-Perdomo et al., (2025) </p>
###  <p align=center> Corresponding author: N. Castro-Perdomo, jcastrop@iu.edu </p>
---
The results and figures presented in our manuscript are fully reproducible. To replicate our analyses, run the Jupyter notebook ``JGR_Alpides_Supplements.ipynb`` included with this repository.

## **Instructions:**

### **1. Install Git Large File Storage (LFS)**

Before cloning the repository, make sure Git Large File Storage (LFS) is installed. Use the appropriate command based on your operating system:

- **macOS (Homebrew):**
```bash
brew install git-lfs
```

- **macOS (MacPorts):**
```bash
port install git-lfs
```

- **Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install git-lfs
```

### **2. Clone the repository**

Once Git LFS is installed, clone the repository:

```bash
git clone https://github.com/ncperdomo/Alpides_Strain_Rates_Supplements.git
cd Alpides_Strain_Rates_Supplements
```

### **3. Set up the Conda environment (optional but recommended)**

To manage dependencies and ensure reproducibility, we recommend creating a Conda environment using the provided `environment.yml` file. This step is optional if all required packages are already installed on your system.

To create and activate the Conda environment, run:

```bash
conda env create -f environment.yml
conda activate alpides
```

### **4. Launch the Jupyter notebook**

Start the Jupyter notebook to reproduce the manuscript's results:

```bash
jupyter notebook JGR_Alpides_Supplements.ipynb
```

If ``alpides`` does not appear in the Jupyter notebook's kernel list, you can manually register it:

```bash
python -m ipykernel install --user --name alpides --display-name "Python (alpides)"
```
After registering the kernel, **restart Jupyter Notebook** to ensure the new kernel appears in the kernel list.

---

## **Directory structure**

The diagram below outlines the contents and organization of the supplementary material provided with this repository.

```markdown
📦Alpides_Strain_Rates_Supplements
 ┣ 📜JGR_Alpides_Supplements.ipynb
 ┣ 📂combined_velocity_field
 ┃ ┣ 📜combined_vel_aege_clean_scaled.csv
 ┃ ┣ 📜combined_vel_amur_clean_scaled.csv
 ┃ ┣ 📜combined_vel_anat_clean_scaled.csv
 ┃ ┣ 📜combined_vel_arab_clean_scaled.csv
 ┃ ┣ 📜combined_vel_eura_clean_scaled.csv
 ┃ ┣ 📜combined_vel_igb14_clean_scaled.csv
 ┃ ┣ 📜combined_vel_indi_clean_scaled.csv
 ┃ ┣ 📜combined_vel_myan_clean_scaled.csv
 ┃ ┣ 📜combined_vel_nubi_clean_scaled.csv
 ┃ ┣ 📜combined_vel_sina_clean_scaled.csv
 ┃ ┣ 📜combined_vel_tbet_clean_scaled.csv
 ┃ ┣ 📜combined_vel_yang_clean_scaled.csv
 ┃ ┗ 📜combined_vertical_velocity_field.vel
 ┣ 📂strain_rate_data
 ┃ ┣ 📜strain_rates_ALPIDES_JGR.csv
 ┃ ┣ 📜strain_rates_EMED_creep_JGR.csv
 ┃ ┣ 📜strain_rates_EMED_nocreep_JGR.csv
 ┃ ┣ 📜strain_rates_India_Asia_JGR.csv
 ┃ ┗ 📜strain_rates_MED_creep.csv
 ┣ 📂input_data
 ┃ ┣ 📂cpts
 ┃ ┣ 📂datasets
 ┃ ┣ 📂modules
 ┣ 📂output_data
 ┃ ┣ 📂figures
 ┣ 📜LICENSE
 ┣ 📜README.md
 ┣ 📜environment.yml
 ┣ 📜requirements.txt
 ┗ 📜runtime.txt
```

- Combined GNSS velocities rotated into different reference frames are provided in the `combined_velocity_field` folder
- Mean posterior strain rates and uncertainties are provided in the `strain_rate_data` folder
- Ancillary files including fault traces, SKS data, and other supporting files can be found in the `input_data` folder 
- Python modules required for plotting strain rates, harmonizing GNSS velocity uncertainties and performing Metropolis MCMC inversions of GNSS velocities for fault kinematic parameters are included in the `input_data/modules` folder
