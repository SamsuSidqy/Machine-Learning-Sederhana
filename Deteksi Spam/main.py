import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from ascii_magic import AsciiArt
from colorama import Fore


with open("data/spam.json","r") as file:
	data = json.load(file)

with open("data/notspam.json","r") as file:
	dataNot = json.load(file)
	

title = []
for i in range(0,20):
	title.append(1)

for i in range(len(dataNot["notspam"])):
	title.append(2)
	data["spam"].append(dataNot["notspam"][i])


dataSet = {
	"title":title,
	"message":data["spam"],
}

# df = pd.DataFrame(dataSet)
# print(df)


encod = TfidfVectorizer()
X = encod.fit_transform(dataSet["message"]).toarray()
Y = np.array(dataSet["title"])

# df = pd.DataFrame(X.todense().T,
# 	index=encod.get_feature_names_out(),
# 	columns=[f"D{i+1}" for i in range(len(dataSet["message"]))])
# print(df)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)
hasil = model.predict(X_test)
my_art = AsciiArt.from_image('util/py.png')
my_art.to_terminal(columns=45)

inputan = str(input("Masukan Kata / Pesan = "))
sett = encod.transform([inputan])
hasil2 = model.predict(sett)
print(hasil2[0])
if hasil2[0] > 1.2:
	print("Pesan Tersebut Bukan SPAM")
else:
	print("Pesan Tersebut SPAM")

print(f"{Fore.YELLOW} ** 40% Program Ini Kemungkinan Masih Salah, Karena Dikitnya Data **")




