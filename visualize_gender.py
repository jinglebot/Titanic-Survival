import pandas as pd
import matplotlib.pyplot as plt

female_color = "#FA0000"

# panda dataframe for loading csv files
df = pd.read_csv("data/train.csv")
fig = plt.figure(figsize=(24, 10))

# shows how many survived and how many died
plt.subplot2grid((3,4), (0,0))
df.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Survived")

# shows how many men survived and how many died
plt.subplot2grid((3,4), (0,1))
df.Survived[df.Sex == "male"].value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Men Survived")

# shows how many women survived and how many died
plt.subplot2grid((3,4), (0,2))
df.Survived[df.Sex == "female"].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color=female_color)
plt.title("Women Survived")

# compares how many men and women survived
plt.subplot2grid((3,4), (0,3))
df.Sex[df.Survived == 1].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color=[female_color, 'b'])
plt.title("Sex of Survived")

# shows distribution of age wrt passenger class
plt.subplot2grid((3,4), (1,0), colspan=4)
for x in [1,2,3]:
    df.Survived[df.Pclass == x].plot(kind="kde")
plt.title("Survived wrt Pclass")
plt.legend(("1st", "2nd", "3rd"))

# shows how many rich men survived and how many died
plt.subplot2grid((3,4), (2,0))
df.Survived[(df.Sex == "male") & (df.Pclass == 1)].value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Rich Men Survived")

# shows how many poor men survived and how many died
plt.subplot2grid((3,4), (2,1))
df.Survived[(df.Sex == "male") & (df.Pclass == 3)].value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Poor Men Survived")

# shows how many rich women survived and how many died
plt.subplot2grid((3,4), (2,2))
df.Survived[(df.Sex == "female") & (df.Pclass == 1)].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color=female_color)
plt.title("Rich Women Survived")

# shows how many poor women survived and how many died
plt.subplot2grid((3,4), (2,3))
df.Survived[(df.Sex == "female") & (df.Pclass == 3)].value_counts(normalize=True).plot(kind="bar", alpha=0.5, color=female_color)
plt.title("Poor Women Survived")


plt.show()

