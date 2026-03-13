from flask import Flask, render_template, request
import joblib  # Utilise joblib car ton script d'entraînement l'utilise
import numpy as np
import os

# 1. CRÉER L'APPLICATION D'ABORD
app = Flask(__name__)

# 2. CHARGER LE MODÈLE ENSUITE
# On utilise un chemin plus robuste pour éviter les erreurs de dossier
model_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'random_forest_model.pkl')
model = joblib.load(model_path)

# 3. LES ROUTES VIENNENT APRÈS
@app.route('/', methods=['GET', 'POST'])

def index():
    prediction = None  # On garde cette ligne pour initialiser la variable

    if request.method == 'POST':
        # 1. On récupère les données
        try:
            data = [
                float(request.form.get('age', 0)),
                float(request.form.get('smokes', 0)),
                float(request.form.get('smokes_years', 0)),
                float(request.form.get('partners', 0)),
                float(request.form.get('first_intercourse', 0)),
                float(request.form.get('pregnancies', 0)),
                float(request.form.get('hormonal_years', 0)),
                float(request.form.get('stds', 0)),
                float(request.form.get('cancer_prev', 0))
            ]

            # 2. Transformation pour l'algorithme (On passe de 9 à 28 colonnes)
            features_full = np.zeros((1, 28)) 
        
            # On place tes 9 données dans les 9 premières colonnes
            features_full[0, 0:9] = data 
        
            # 3. Prédiction avec le tableau complet
            prediction_value = model.predict(features_full)[0]
            
            # 4. On écrase le 'None' par le vrai message
            if prediction_value == 1:
                prediction = "Risque élevé détecté."
            else:
                prediction = "Faible risque détecté."
        
        except Exception as e:
            prediction = f"Erreur lors du calcul : {str(e)}"

    # On renvoie toujours 'prediction' au template
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)