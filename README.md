# Compara-o-entre-modelos-de-ML & Income Classification – Logistic Regression & Random Forest

Este projeto implementa um pipeline completo de **classificação de renda** utilizando dois modelos de Machine Learning:

- **Regressão Logística**
- **Random Forest**

O objetivo é prever se um indivíduo possui renda anual **maior que USD 50K**, utilizando dados socioeconômicos públicos (Adult Census Income Dataset).

link do dataset: https://www.kaggle.com/datasets/ayessa/salary-prediction-classification

O script `main.py` executa todo o fluxo automaticamente:

✔ Carrega o dataset  
✔ Realiza pré-processamento completo  
✔ Treina os dois modelos  
✔ Gera métricas em `.csv`  
✔ Exporta gráficos (matriz de confusão etc.)  
✔ Salva resultados na pasta `data/`

---

## 1. Instalação

Requer **Python 3.11+**.

### 1.1. codigo
```bash
git clone https://github.com/Luiz-Fernando-Policarpo-Leandro/Compara-o-entre-modelos-de-ML.git
cd Compara-o-entre-modelos-de-ML
pip install -r requirements.txt
python main.py
```
O programa automaticamente:

* Lê o arquivo salary.csv

* Pré-processa os dados

* Treina os modelos

* Calcula todas as métricas

Gera arquivos de saída na pasta data/

Exporta gráficos e relatórios em .csv
