document.getElementById('analyzeBtn').addEventListener('click', uploadImage);
document.getElementById('imageInput').addEventListener('change', showPreview);

function showPreview() {
  const file = this.files[0];
  const preview = document.getElementById('imagePreview');

  if (file) {
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';
  } else {
    preview.style.display = 'none';
  }
}

function uploadImage() {
  const fileInput = document.getElementById('imageInput');
  const file = fileInput.files[0];
  const preview = document.getElementById('imagePreview');
  const result = document.getElementById('result');
  const error = document.getElementById('error');

  if (!file) {
    error.textContent = '⚠ Please select an image before analyzing.';
    error.style.display = 'block';
    result.style.display = 'none';
    return;
  }

  const formData = new FormData();
  formData.append('image', file);

  fetch('http://localhost:5000/predict', {
  method: 'POST',
  body: formData
})
.then(response => {
  if (!response.ok) {
    throw new Error(`Server error: ${response.status}`);
  }
  return response.json();
})
.then(data => {
  if (data.error) {
    throw new Error(data.error);
  }

  document.getElementById('disease').textContent = data.disease;
  document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(2);
  result.style.display = 'block';
  error.style.display = 'none';
})
.catch(err => {
  error.textContent = `❌ Error: ${err.message}`;
  error.style.display = 'block';
  result.style.display = 'none';
});
}