let predChart, liveChart;
let liveData = { labels: [], data: [] };
let chatHistory = [];

document.getElementById('predictBtn').onclick = async () => {
  const symbol = document.getElementById('symbolSelect').value;
  const res = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol }),
  });
  const { predictions, recommendation } = await res.json();

  // Build historical+forecast chart
  const labels = predictions.map((p) => p.date);
  const data = predictions.map((p) => p.predicted_close);
  document.getElementById(
    'recText'
  ).textContent = `Recommendation: ${recommendation}`;

  const ctx = document.getElementById('predChart').getContext('2d');
  if (predChart) predChart.destroy();
  predChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: `${symbol} Forecast`,
          data,
          tension: 0.3,
          borderWidth: 2,
        },
      ],
    },
    options: {
      scales: {
        x: { title: { display: true, text: 'Date' } },
        y: { title: { display: true, text: 'Price' } },
      },
    },
  });

  // Reset live chart data
  liveData = { labels: [], data: [] };
};

async function fetchLive(symbol) {
  const res = await fetch(`/live_price?symbol=${symbol}`);
  return res.json();
}

async function updateLiveChart() {
  const symbol = document.getElementById('symbolSelect').value;
  const { price, timestamp } = await fetchLive(symbol);

  liveData.labels.push(timestamp);
  liveData.data.push(price);
  if (liveData.labels.length > 30) {
    liveData.labels.shift();
    liveData.data.shift();
  }

  document.getElementById('livePriceText').textContent = `${symbol}: $${price}`;

  const ctx = document.getElementById('liveChart').getContext('2d');
  if (liveChart) liveChart.destroy();
  liveChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: liveData.labels,
      datasets: [
        {
          label: `${symbol} Live`,
          data: liveData.data,
          tension: 0.1,
          borderDash: [5, 5],
        },
      ],
    },
    options: {
      scales: {
        x: { display: false },
        y: { title: { display: true, text: 'Price' } },
      },
    },
  });
}

// Update live price every 10 seconds
setInterval(updateLiveChart, 10000);

// Chat
document.getElementById('sendChat').onclick = async () => {
  const symbol = document.getElementById('symbolSelect').value;
  const recommendation =
    document.getElementById('recText').textContent.split(': ')[1] || '';
  const question = document.getElementById('chatInput').value;
  if (!question) return;
  document.getElementById('chatInput').value = '';

  // Append user message
  chatHistory.push({ from: 'you', text: question });
  renderChat();

  const res = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol, recommendation, question }),
  });
  const { reply } = await res.json();
  chatHistory.push({ from: 'stock', text: reply });
  renderChat();
};

function renderChat() {
  const win = document.getElementById('chatWindow');
  win.innerHTML = '';
  chatHistory.forEach((msg) => {
    const div = document.createElement('div');
    div.className =
      msg.from === 'you'
        ? 'text-right text-blue-600'
        : 'text-left text-gray-800';
    div.textContent = (msg.from === 'you' ? 'You: ' : 'Stock: ') + msg.text;
    win.appendChild(div);
  });
  win.scrollTop = win.scrollHeight;
}
