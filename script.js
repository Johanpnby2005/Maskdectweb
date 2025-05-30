let model;

window.onload = async () => {
  try {
    model = await tf.loadLayersModel('tfjs_model3/model.json');
    console.log("✅ Modelo cargado");

    // Opcional: imprimir pesos para verificar
    model.weights.forEach(w => {
      console.log(w.name, w.shape);
    });

  } catch (err) {
    console.error("❌ Error cargando modelo:", err);
  }
};

const imageInput = document.getElementById('imageInput');
const previewImage = document.getElementById('previewImage');
const resultDiv = document.getElementById('result');

imageInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  const reader = new FileReader();

  reader.onload = function (e) {
    previewImage.src = e.target.result;
  };

  if (file) reader.readAsDataURL(file);
});

async function predict() {
  if (!model) {
    resultDiv.textContent = "Cargando modelo, espera un momento...";
    return;
  }

  const image = tf.browser.fromPixels(previewImage)
    .resizeNearestNeighbor([150, 150]) // <-- 🔧 Cambiado de 224x224 a 150x150
    .toFloat()
    .div(tf.scalar(255.0)) // Normalización
    .expandDims();         // Añade dimensión batch [1, 150, 150, 3]

  const prediction = model.predict(image);
  const predArray = await prediction.array();
  const index = predArray[0].indexOf(Math.max(...predArray[0]));

  const etiquetas = ["Con mascarilla", "Sin mascarilla", "Mascarrila mal puesta"];
  const colores = ["#28a745", "#dc3545", "#ffc107"];


  resultDiv.textContent = `Resultado: ${etiquetas[index]}`;
  resultDiv.style.color = colores[index];
}
