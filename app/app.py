from flask import Flask, render_template, request

app = Flask(__name__)  # <--- C'est cette ligne qui manque !
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Récupération de toutes les données du questionnaire
        data = {
            'age': request.form.get('age'),
            'partners': request.form.get('partners'),
            'pregnancies': request.form.get('pregnancies'),
            'smokes': request.form.get('smokes'),
            'smokes_years': request.form.get('smokes_years'),
            'hormonal_years': request.form.get('hormonal_years'),
            'stds': request.form.get('stds'),
            'cancer_prev': request.form.get('cancer_prev')
        }
        
        # Simulation du résultat (en attendant l'intégration du modèle .pkl)
        prediction = f"Analyse terminée. Risque évalué selon {len(data)} paramètres cliniques."
        
    return render_template('index.html', prediction=prediction)
if __name__ == '__main__':
    app.run(debug=True)
    