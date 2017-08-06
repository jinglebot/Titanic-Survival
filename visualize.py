import pandas as pd
import matplotlib.pyplot as plt
import utils
# panda dataframe for loading csv files
df = pd.read_csv("data/train.csv")
utils.clean_data(df)
fig = plt.figure(figsize=(24, 10))

# shows how many survived and how many died
plt.subplot2grid((5,3), (0,0))
df.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Survived")

# shows age of those who survived
plt.subplot2grid((5,3), (0,1))
plt.scatter(df.Survived, df.Age, alpha=0.1)
plt.title("Age wrt Survived")

# shows how many passengers in each class
plt.subplot2grid((5,3), (0,2))
df.Pclass.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Pclass")

# shows distribution of age wrt passenger class
plt.subplot2grid((5,3), (1,0), colspan=2)
for x in [1,2,3]:
    df.Age[df.Pclass == x].plot(kind="kde")
plt.title("Pclass wrt Age")
plt.legend(("1st", "2nd", "3rd"))

# shows how many embarked per port
plt.subplot2grid((5,3), (1,2))
df.Embarked.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Embarked")

# shows how many survived that have or doesn't have cabins
plt.subplot2grid((5,3), (2,0), colspan=3)
df.Cabin[df.Survived == 1].value_counts().plot(kind="bar", alpha=0.5)
plt.title("Cabin wrt Survived")

# shows how many survived w cabins and died with cabins
plt.subplot2grid((5,3), (3,0), colspan=3)
for x in [0,1,2,3]:
    df.Survived[df.Cabin == x].plot(kind="kde")
plt.title("Survived wrt Cabin")
plt.legend(("none", "A", "B", "C"))
#, "D", "E", "F", "G"))

print(len(df[(df.Survived == 1) & (df.Cabin != 0)]))  #136  39.77%
print(len(df[(df.Survived == 1) & (df.Cabin == 1)]))  #A = 7
print(len(df[df.Cabin == 1]))                         #15  46.7%
print(len(df[(df.Survived == 1) & (df.Cabin == 2)]))  #B = 35
print(len(df[df.Cabin == 2]))                         #47  74.5%
print(len(df[(df.Survived == 1) & (df.Cabin == 3)]))  #C = 35
print(len(df[df.Cabin == 3]))                         #59  59.3%
print(len(df[(df.Survived == 1) & (df.Cabin == 4)]))  #D = 25 
print(len(df[df.Cabin == 4]))                         #33  75.76%
print(len(df[(df.Survived == 1) & (df.Cabin == 5)]))  #E = 25
print(len(df[df.Cabin == 5]))                         #33  75.76%
print(len(df[(df.Survived == 1) & (df.Cabin == 6)]))  #F = 7
print(len(df[df.Cabin == 6]))                         #12  58.3% 
print(len(df[(df.Survived == 1) & (df.Cabin == 7)]))  #G = 2
print(len(df[df.Cabin == 7]))                         #4   50%
print(len(df[df.Survived == 1]))                      #342
print(len(df[df.Cabin != 0]))                         #204

plt.show()



