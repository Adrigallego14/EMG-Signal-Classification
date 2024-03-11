#Definir la ruta al archivo CSV
Sujeto1 = "/content/drive/MyDrive/TFG/DataEdwin/Dataset/EDW_mod.csv" #12000
Sujeto2 = "/content/drive/MyDrive/TFG/DataEdwin/Dataset/GIANLUCA_mod.csv" #10000
Sujeto3 = "/content/drive/MyDrive/TFG/DataEdwin/Dataset/LEO_mod.csv" #32000


#Leer el archivo CSV con el delimitador adecuado
df1 = pd.read_csv(Sujeto1)
df2 = pd.read_csv(Sujeto2)
df3 = pd.read_csv(Sujeto3)
