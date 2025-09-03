// disease_prediction_project/static/js/script.js

document.querySelectorAll('form').forEach(form => {
    form.addEventListener('submit', function(event) {
        event.preventDefault();
        
        const formId = this.id;
        const diseaseType = formId.split('-')[0];
        const resultElementId = `${diseaseType}-result`;
        
        fetch('/predict/', {
            method: 'POST',
            body: new FormData(this),
        })
        .then(response => response.json())
        .then(data => {
            if (data.prediction !== undefined) {
                const message = data.prediction === 1 ? 'Risk detected! Please consult a doctor.' : 'Low risk.';
                document.getElementById(resultElementId).innerText = message;
            } else {
                document.getElementById(resultElementId).innerText = `Error: ${data.error}`;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById(resultElementId).innerText = 'Prediction failed.';
        });
    });
});