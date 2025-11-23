
# 🏏 Cricket Shot Optimization using Machine Learning + Global Optimization

This project predicts the **optimal cricket shot parameters** (bat swing speed, launch angle, timing, spin, etc.) to maximize the **shot distance**, while ensuring the parameters remain physically realistic.  
It uses:

- Machine Learning regression models  
- Multi-algorithm global optimization  
- Cricket physics constraints  
- Automated JSON/CSV logging  
- Visualization of ball trajectory + fielder positions  
- Report generation  

---

## 🚀 Features

✔ ML Model Training (Linear Regression, Random Forest, Gradient Boosting, SVR)  
✔ Automatic model selection based on CV R²  
✔ Physics-based constraints  
✔ Objective function with penalties  
✔ 6 Optimization Algorithms  
✔ Saves results to JSON + CSV  
✔ Generates a cricket-field trajectory PNG  
✔ Fully reproducible  
✔ Ready for Kaggle / Local execution  

---

## 📂 Project Structure

```
project/
│
├── main.py                     # full optimization script
├── output/                     # generated plots + logs (auto created)
│   ├── field_shot_plot_*.png   # visualization of shot + fielders
│   ├── shot_summary_*.json     # prediction + parameters
│   ├── shot_summaries.csv      # aggregated results
│
├── simulated_shots.csv         # dataset (Kaggle path or local)
├── README.md                   # project documentation
```

---

## 📦 Requirements

Install dependencies:

```bash
pip install numpy pandas scikit-learn scipy matplotlib python-docx reportlab
```

---

## 📁 Dataset

The script expects:

```
simulated_shots.csv
```

If running on Kaggle:
- Place the CSV in:  
  `/kaggle/input/simulated-shots/simulated_shots.csv`

If running locally:
- Place it in the same directory as `main.py`, or update the path in the code.

---

## ▶️ How to Run

### **Option 1 — Local Run (Python)**

```bash
python main.py
```

### **Option 2 — Kaggle Notebook**

1. Upload `main.py`
2. Upload dataset to `/kaggle/input/`
3. Run all cells


## 📊 Output Files Generated

After each run, the script automatically generates:

### **1️⃣ JSON Summary**
Contains optimized parameters + predicted distance.

Example:
```
output/shot_summary_2025-01-01T10-32-11.json
```

### **2️⃣ CSV Log**
All runs appended in one place:

```
output/shot_summaries.csv
```

### **3️⃣ Field Plot (PNG)**
Shows:
- Fielder positions  
- Batsman  
- Ball trajectory  
- Landing point  

Example:
```
output/field_shot_plot_2025-01-01T10-32-11.png
```


## 🧠 Optimization Algorithms Used

The script runs 6 different solvers:

- Differential Evolution  
- Basin Hopping  
- Dual Annealing  
- SHGO  
- SLSQP  
- COBYLA  

And automatically selects the **best shot distance** among them.

---

## ⭐ Acknowledgements

- Kaggle environment  
- Scikit-Learn  
- SciPy Optimization Suite  
