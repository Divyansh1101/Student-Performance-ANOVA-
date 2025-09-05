import os
import numpy as np
import pandas as pd

print("Starting Student Performance Analysis (ANOVA)...")

DATA_PATH = os.path.join("data", "student_performance.csv")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------- Load or synthesize dataset -------
if not os.path.exists(DATA_PATH):
    print("Dataset not found, using synthetic data ->", DATA_PATH)
    os.makedirs("data", exist_ok=True)
    rng = np.random.default_rng(42)
    n = 240
    study_hours = rng.choice(["<1 hr","1-2 hrs","2-4 hrs",">4 hrs"], size=n, p=[0.2,0.35,0.3,0.15])
    gender = rng.choice(["Male","Female"], size=n)
    parent_edu = rng.choice(["High School","Bachelor","Master","PhD"], size=n, p=[0.35,0.4,0.2,0.05])
    base = rng.normal(65, 10, size=n)
    # Effects
    study_effect = {"<1 hr":-8,"1-2 hrs":-2,"2-4 hrs":+5,">4 hrs":+8}
    parent_effect = {"High School":-3,"Bachelor":+0,"Master":+3,"PhD":+4}
    score = base + np.vectorize(study_effect.get)(study_hours) + np.vectorize(parent_effect.get)(parent_edu)
    score = np.clip(score, 0, 100)
    df = pd.DataFrame({
        "score": score.round(1),
        "study_hours": study_hours,
        "gender": gender,
        "parental_education": parent_edu
    })
    df.to_csv(DATA_PATH, index=False)
else:
    df = pd.read_csv(DATA_PATH)

# ------- Summary stats -------
summary = df.describe(include="all")
summary_path = os.path.join(RESULTS_DIR, "summary.csv")
summary.to_csv(summary_path)
print(f"Saved summary -> {summary_path}")

# ------- Visualizations -------
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
sns.histplot(df["score"], kde=True)
plt.title("Score Distribution")
plt.xlabel("Score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "histogram_scores.png"), dpi=150)
plt.close()

plt.figure()
sns.boxplot(x="study_hours", y="score", data=df)
plt.title("Scores by Study Hours")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "boxplot_scores_by_study_hours.png"), dpi=150)
plt.close()

plt.figure()
sns.violinplot(x="parental_education", y="score", data=df, inner="quartile")
plt.title("Scores by Parental Education")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "violin_scores_by_parentEdu.png"), dpi=150)
plt.close()

# ------- One-way ANOVA for multiple factors -------
from scipy import stats

def anova_oneway(df, factor, target="score"):
    groups = [g[target].values for _, g in df.groupby(factor)]
    f, p = stats.f_oneway(*groups)
    return f, p, len(groups)

anova_text = []
for factor in ["study_hours", "gender", "parental_education"]:
    f, p, k = anova_oneway(df, factor)
    anova_text.append(f"Factor: {factor} | groups={k} | F={f:.4f} | p-value={p:.6f}")
    # Simple interpretation
    if p < 0.05:
        anova_text.append(f"Interpretation: Significant mean differences across {factor} groups (reject H0 at 5%).")
    else:
        anova_text.append(f"Interpretation: No significant differences across {factor} groups (fail to reject H0).")
    anova_text.append("")

anova_report_path = os.path.join(RESULTS_DIR, "anova_results.txt")
with open(anova_report_path, "w", encoding="utf-8") as f:
    f.write("Student Performance ANOVA Results\n\n")
    f.write("\n".join(anova_text))

print(f"Saved ANOVA results -> {anova_report_path}")
print("Done.")
