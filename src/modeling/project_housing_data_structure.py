import pandas as pd
import matplotlib.pyplot as plt

housing = pd.read_csv("D:\\USER\\git_github\\MLOps_project_1\\data\\raw\\housing.csv")

sample_data = housing.head()
housing_info = housing.info()
ocean_proximity_counts = housing["ocean_proximity"].value_counts()
ocean_proximity_counts
deskripsi_statistik = housing.describe()
deskripsi_statistik

housing.hist(bins=50, figsize=(18, 12))
plt.show()


fig, ax = plt.subplots(figsize=(6, 6))
ax.axis("off")  # Hide axes
table = ax.table(
    cellText=deskripsi_statistik.values,
    colLabels=deskripsi_statistik.columns,
    loc="center",
)

plt.savefig("statistik_housing.png", dpi=300, bbox_inches="tight")
plt.close()
