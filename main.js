console.log('Dementia Prediction System Loaded');

document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const submitBtn = document.querySelector('.submit-btn');
    const resultsSection = document.getElementById('results');
    
    submitBtn.disabled = true;
    submitBtn.classList.add('loading');
    submitBtn.textContent = 'Processing...';
    
    resultsSection.style.display = 'none';
    
    const existingError = document.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }
    
    const formData = new FormData(e.target);
    const data = {};
    
    for (let [key, value] of formData.entries()) {
        data[key] = value;
    }
    
    console.log('Form data:', data);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            let errorMessage = `Server error: ${response.status}`;
            try {
                const errorJson = JSON.parse(errorText);
                errorMessage = errorJson.error || errorMessage;
            } catch (e) {
                if (errorText) errorMessage += '. ' + errorText;
            }
            throw new Error(errorMessage);
        }
        
        const responseText = await response.text();
        console.log('Response:', responseText);
        
        if (!responseText || responseText.trim() === '') {
            throw new Error('Empty response from server');
        }
        
        const result = JSON.parse(responseText);
        console.log('Result:', result);
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        displayResults(result);
        
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
        
    } catch (error) {
        console.error('Error:', error);
        let errorMessage = 'An error occurred. ';
        
        if (error.message.includes('Failed to fetch')) {
            errorMessage = 'Cannot connect to server. Make sure server is running.';
        } else if (error.message) {
            errorMessage += error.message;
        }
        
        showError(errorMessage);
    } finally {
        submitBtn.disabled = false;
        submitBtn.classList.remove('loading');
        submitBtn.textContent = '🔍 Predict Risk';
    }
});

function displayResults(result) {
    const resultsSection = document.getElementById('results');
    const resultText = document.getElementById('resultText');
    const resultIcon = document.getElementById('resultIcon');
    const resultDescription = document.getElementById('resultDescription');
    
    resultsSection.style.display = 'block';
    
    if (result.prediction === 1) {
        resultText.textContent = 'YES - Dementia Detected';
        resultIcon.className = 'result-icon negative';
        resultDescription.textContent = 'The model predicts dementia is present. Please consult a healthcare professional for proper evaluation.';
    } else {
        resultText.textContent = 'NO - No Dementia Detected';
        resultIcon.className = 'result-icon positive';
        resultDescription.textContent = 'The model predicts no signs of dementia. Regular health check-ups are always recommended.';
    }
    
    if (result.confidence) {
        resultDescription.textContent += ` (Confidence: ${(result.confidence * 100).toFixed(1)}%)`;
    }
}

function showError(message) {
    const resultsSection = document.getElementById('results');
    
    const existingError = document.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `<strong>Error:</strong> ${message}`;
    
    resultsSection.parentNode.insertBefore(errorDiv, resultsSection);
    
    setTimeout(() => {
        errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 100);
    
    setTimeout(() => {
        errorDiv.remove();
    }, 15000);
}

document.querySelectorAll('input[type="number"]').forEach(input => {
    input.addEventListener('input', function() {
        if (this.value < 0) this.value = 0;
        if (this.id === 'BloodOxygenLevel' && this.value > 100) this.value = 100;
        if (this.id === 'AlcoholLevel' && this.value > 1) this.value = 1;
        if (this.id === 'Cognitive_Test_Scores' && this.value > 10) this.value = 10;
        if (this.id === 'Age' && this.value > 120) this.value = 120;
    });
});

console.log('Ready!');