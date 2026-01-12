# Robyn MMM Training

This directory contains scripts for training Marketing Mix Models using Meta's Robyn framework.

## Three Approaches

### 1. Python API (Recommended ✅)

**File:** `train_model_python.py`

Pure Python implementation using the official Robyn Python API.

**Benefits:**
- ✅ No R installation required
- ✅ Pure Python code
- ✅ Easier to debug
- ✅ Better IDE support
- ✅ Simpler deployment

**Usage:**
```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train_model_python.py
```

**Documentation:** https://facebookexperimental.github.io/Robyn/docs/robyn-api

### 2. Python + R (via rpy2)

**File:** `train_model.py`

Python wrapper around R Robyn package using rpy2.

**Use when:**
- Need specific R Robyn features not yet in Python API
- Want to leverage existing R Robyn models
- Have R installed and prefer R's robustness

**Requirements:**
- R 4.0+
- Robyn R package
- rpy2 Python package

**Usage:**
```bash
# Uncomment rpy2 in requirements.txt
pip install rpy2

# Install R packages
R -e "install.packages('Robyn')"

# Train model
python train_model.py
```

### 3. Pure R Script

**File:** `train_model.R`

Native R script for Robyn training.

**Use when:**
- Working entirely in R environment
- Need maximum compatibility with R Robyn
- Prefer R for statistical modeling

**Requirements:**
- R 4.0+
- Robyn R package
- bigrquery, yaml, jsonlite packages

**Usage:**
```bash
# Install R packages
R -e "install.packages(c('Robyn', 'bigrquery', 'yaml', 'jsonlite'))"

# Train model
Rscript train_model.R
```

## Configuration

All scripts use the same configuration file:

**File:** `config/robyn_config.yaml`

Key settings:
- `model.channel_var`: Channel column name ("channel")
- `model.spend_var`: Spend column name ("spend")
- `model.dep_var`: Dependent variable ("revenue" or "orders")
- `hyperparameters.iterations`: Training iterations (default: 2000)

## Output

All approaches produce:

1. **Model file:**
   - Python API: `models/robyn_model_{id}.pkl`
   - R/rpy2: `models/robyn_model_{id}.rds`

2. **Results JSON:** `models/robyn_results.json`
   Contains:
   - Channel contributions
   - ROI metrics
   - Model performance metrics
   - Channel list (auto-detected!)

3. **Plots:** `plots/` directory
   - Response curves
   - Budget allocation
   - Model diagnostics

## Budget Allocation

### Python API:
```python
from train_model_python import RobynMMM

trainer = RobynMMM('config/robyn_config.yaml')
allocation = trainer.allocate_budget(
    model_file='models/robyn_model_123.pkl',
    total_budget=100000
)
```

### Pure R:
```bash
Rscript allocate_budget.R models/robyn_model_123.rds 100000
```

## Development vs Production

**Development:**
All scripts include fallback logic for development when Robyn is not installed. They generate placeholder results with error messages.

**Production:**
Install `facebook-robyn` (Python) or Robyn R package for full functionality.

## Scalability

✅ **Channels auto-detected from data!**

No need to update config when adding channels. Just add data in DBT:

```sql
-- In stg_channel_spend.sql
youtube_spend AS (
  SELECT DATE(date) AS date, 'youtube' AS channel, ...
)
```

Run `dbt run`, then `python train_model_python.py` - done!

## Docker Support

See `Dockerfile` for containerized training with R support.

For Python API only (lighter image):
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
# ... rest of dockerfile
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'robyn'"
```bash
pip install facebook-robyn
```

### "R package 'Robyn' not installed"
```R
install.packages('Robyn')
```

### "Error during Robyn training"
Check logs for specific error. Common issues:
- Insufficient data (need 24+ months)
- Missing channels in spend columns
- Data quality (NaN, missing dates)

## Links

- Robyn Python API: https://facebookexperimental.github.io/Robyn/docs/robyn-api
- Robyn R Package: https://facebookexperimental.github.io/Robyn/
- Robyn GitHub: https://github.com/facebookexperimental/Robyn
