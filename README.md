# CrossCoinMapping
CrossCoinMapping is a project designed to map and analyze token similarities based on historical price data using metrics like DTW (Dynamic Time Warping), MAE, RMSE, MAPE, SMAPE, and Cosine Similarity. This project leverages multiprocessing for efficiency and creates visualizations to illustrate token correlations within clusters.

# Project Overview

CrossCoinMapping processes token data and computes similarities between token pairs based on historical prices. It outputs distance metrics such as:

* DTW (Dynamic Time Warping) Distance: Measures similarity between two temporal sequences.
* MAE (Mean Absolute Error): Measures average magnitude of errors.
* RMSE (Root Mean Squared Error): Measures average squared differences.
* MAPE (Mean Absolute Percentage Error): Shows the accuracy of forecast models.
* SMAPE (Symmetric Mean Absolute Percentage Error): A variation of MAPE that accounts for scaling issues.

* Cosine Similarity: Measures the cosine of the angle between two vectors.

This project uses multiprocessing to handle token pairs within clusters, improving performance by parallelizing the similarity computations.

# Project Hierarchy 

CrossCoinMapping
├── data/                        # Data folder (contains token price data)
├── visualizations/               # Output folder for similarity plots
├── src/                         # Source code folder
│   ├── process_similarity.py     # Script to process token similarities
│   ├── metrics.py                # Script containing error and distance metrics
│   └── visualization.py          # Script to plot token similarities
├── .venv/                        # Virtual environment folder
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation

# setup
Ensure that you have the following installed on your system and **python version should 3.10**
## Ubuntu User
1. Clone the repository:
```angular2html
git clone https://github.com/iffishells/CrossCoinMapping.git
cd CrossCoinMapping
```
2. Create and activate a virtual environment(Ubuntu User):
```angular2html
python3.10 -m venv .venv
source .venv/bin/activate
```
3.  Create and activate a virtual environment(window User):
```angular2html
python3.10 -m venv .venv
.venv\Scripts\activate
```
3. Install the required dependencies:
```angular2html
pip install -r requirements.txt
```
4. Running jupyter lab
```angular2html
jupyter lab
```
## Window User
1. Clone the repository:
```angular2html
    git clone https://github.com/iffishells/CrossCoinMapping.git
    cd CrossCoinMapping
```
2. Download python 3.10.11 using following steps:
```angular2html

For installing Python 3.10.11, follow these steps:

1. Download Python 3.10.12 from the following link:
	https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
2. You might have multiple versions of Python installed on your system, so you need to change the default version of Python in the following way:
2.1. Go to System Settings
2.2. Open Environment Variables
2.3. Under System Variables, change the paths for Python and Python Scripts.
It should look something like:
- C:\Users\username\AppData\Local\Programs\Python\Python310
- C:\Users\username\AppData\Local\Programs\Python\Python310\Scripts

```
2. Create and activate a virtual environment(window User):
```angular2html
    python -m venv .venv
    .venv\Scripts\activate
```
3. Install the required dependencies:
```angular2html
    pip install -r requirements.txt
```
4. Running jupyter lab
```angular2html
    jupyter lab
```
## Running the Project
# Step 1: Prepare Your Data
Place your Datasets folder in App/.
## Step 2: Run the Similarity Calculation
To compute the similarity metrics, open the Jupyter notebooks located in the `APP/` folder. You can do this by running:
1.  Cross
## Step 3: View the Results
After the script completes, visualizations for token pairs that meet the similarity threshold will be saved in the visualizations/ folder. The similarity metrics will also be saved in a CSV file .
