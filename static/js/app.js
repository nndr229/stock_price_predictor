// static/js/app.js

// ---------- Loading Modal Helpers ----------
function showLoading() {
  document.getElementById('loadingModal').classList.remove('hidden');
}
function hideLoading() {
  document.getElementById('loadingModal').classList.add('hidden');
}

// ---------- Rate-Limit Fetch Wrapper ----------
async function postWithRateCheck(path, body) {
  showLoading();
  try {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (res.status === 429) {
      const err = await res.json();
      alert(err.error);
      return null;
    }
    if (!res.ok) {
      const err = await res.json();
      alert(err.error || 'Server error');
      return null;
    }
    return await res.json();
  } catch (e) {
    console.error(e);
    alert('Network error');
    return null;
  } finally {
    hideLoading();
  }
}

// ---------- State ----------
let predChart = null;
let liveChart = null;
let liveData = { labels: [], data: [] };
let chatHistory = [];

// ---------- PREDICT Button Handler ----------
document.getElementById('predictBtn').addEventListener('click', async () => {
  const symbol = document
    .getElementById('symbolSelect')
    .value.trim()
    .toUpperCase();
  const data = await postWithRateCheck('/predict', { symbol });
  if (!data) return;

  const { predictions, recommendation } = data;
  document.getElementById(
    'recText'
  ).textContent = `Recommendation: ${recommendation}`;

  // Prepare and render prediction chart
  const labels = predictions.map((p) => p.date);
  const values = predictions.map((p) => p.predicted_close);
  const ctx = document.getElementById('predChart').getContext('2d');
  if (predChart) predChart.destroy();
  predChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: `${symbol} Forecast`,
          data: values,
          tension: 0.3,
          borderWidth: 2,
        },
      ],
    },
    options: {
      scales: {
        x: { title: { display: true, text: 'Date' } },
        y: { title: { display: true, text: 'Price (USD)' } },
      },
    },
  });

  // Reset live price data
  liveData = { labels: [], data: [] };
});

// ---------- LIVE PRICE Streaming ----------
async function fetchLive(symbol) {
  const res = await fetch(`/live_price?symbol=${symbol}`);
  if (!res.ok) return null;
  return res.json();
}

async function updateLiveChart() {
  const symbol = document
    .getElementById('symbolSelect')
    .value.trim()
    .toUpperCase();
  const live = await fetchLive(symbol);
  if (!live) return;

  const { price, timestamp } = live;
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
        y: { title: { display: true, text: 'Price (USD)' } },
      },
    },
  });
}

// poll every 10 seconds
setInterval(updateLiveChart, 10000);

// ---------- CHAT Interface ----------
document.getElementById('sendChat').addEventListener('click', async () => {
  const symbol = document
    .getElementById('symbolSelect')
    .value.trim()
    .toUpperCase();
  const recommendation =
    document.getElementById('recText').textContent.split(': ')[1] || '';
  const questionInput = document.getElementById('chatInput');
  const question = questionInput.value.trim();
  if (!question) return;

  // append user message
  chatHistory.push({ from: 'you', text: question });
  renderChat();
  questionInput.value = '';

  const data = await postWithRateCheck('/chat', {
    symbol,
    recommendation,
    question,
  });
  if (!data) return;

  chatHistory.push({ from: 'stock', text: data.reply });
  renderChat();
});

// ---------- Chat Rendering ----------
function renderChat() {
  const win = document.getElementById('chatWindow');
  win.innerHTML = '';
  chatHistory.forEach((msg) => {
    const div = document.createElement('div');
    if (msg.from === 'you') {
      div.className = 'text-right text-blue-600';
      div.textContent = `You: ${msg.text}`;
    } else {
      div.className = 'text-left text-gray-800';
      div.textContent = `Stock: ${msg.text}`;
    }
    win.appendChild(div);
  });
  win.scrollTop = win.scrollHeight;
}
