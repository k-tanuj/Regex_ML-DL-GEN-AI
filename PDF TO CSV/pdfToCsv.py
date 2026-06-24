import pdfplumber
import pandas as pd

data = []

with pdfplumber.open("Churn_Modelling.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            data.append([text])

df = pd.DataFrame(data, columns=['Text'])
df.to_csv('output.csv', index=False)

print("CSV created")