let predChart, liveChart;
let liveData = { labels: [], data: [] };
let chatHistory = [];

// ---------------------
// Helpers / UI feedback
// ---------------------
const $ = (sel) => document.querySelector(sel);

function showLoading(msg = 'Loading…') {
  $('#loadingText').textContent = msg;
  const m = $('#loadingModal');
  m.classList.remove('hidden');
  m.classList.add('flex');
}
function hideLoading() {
  const m = $('#loadingModal');
  m.classList.add('hidden');
  m.classList.remove('flex');
}

function toast(msg) {
  alert(msg); // Simple; replace with a better toast if you want
}

// ---------------------
// Charts initialization
// ---------------------
function initPredChart() {
  const ctx = $('#predChart').getContext('2d');
  predChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{ label: 'Forecast', data: [], tension: 0.3, borderWidth: 2 }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false, // parent has fixed height in HTML
      scales: {
        x: { title: { display: true, text: 'Date' } },
        y: { title: { display: true, text: 'Price' } },
      },
    },
  });
}

function initLiveChart() {
  const ctx = $('#liveChart').getContext('2d');
  liveChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{ label: 'Live', data: [], tension: 0.1, borderDash: [5, 5] }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false, // parent has fixed height in HTML
      scales: {
        x: { display: false },
        y: { title: { display: true, text: 'Price' } },
      },
    },
  });
}

function resetCharts() {
  $('#recText').textContent = '';
  $('#livePriceText').textContent = '';
  $('#feedBadge').textContent = 'feed: —';
  liveData = { labels: [], data: [] };

  // Reset chart data without recreating charts
  predChart.data.labels = [];
  predChart.data.datasets[0].label = 'Forecast';
  predChart.data.datasets[0].data = [];
  predChart.update();

  liveChart.data.labels = [];
  liveChart.data.datasets[0].label = 'Live';
  liveChart.data.datasets[0].data = [];
  liveChart.update();
}

// ---------------------
// API helpers
// ---------------------
async function apiPredict(symbol) {
  const res = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Predict failed: ${res.status} ${text}`);
  }
  return res.json();
}

async function apiChat(symbol, recommendation, question) {
  const res = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol, recommendation, question }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Chat failed: ${res.status} ${text}`);
  }
  return res.json();
}

async function apiLive(symbol) {
  const res = await fetch(`/live_price?symbol=${encodeURIComponent(symbol)}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Live failed: ${res.status} ${text}`);
  }
  return res.json();
}

// ---------------------
// Predict flow
// ---------------------
$('#predictBtn').onclick = async () => {
  const symbol = $('#symbolSelect').value;
  showLoading(`Predicting ${symbol}…`);
  try {
    const {
      predictions = [],
      recommendation = 'Unknown',
      feed,
    } = await apiPredict(symbol);

    // Update forecast chart
    const labels = predictions.map((p) => p.date);
    const data = predictions.map((p) => p.predicted_close);
    predChart.data.labels = labels;
    predChart.data.datasets[0].label = `${symbol} Forecast`;
    predChart.data.datasets[0].data = data;
    predChart.update();

    $('#recText').textContent = `Recommendation: ${recommendation}`;
    if (feed) $('#feedBadge').textContent = `feed: ${feed}`;

    // Reset live series to reflect a new symbol session
    liveData = { labels: [], data: [] };
    liveChart.data.labels = [];
    liveChart.data.datasets[0].label = `${symbol} Live`;
    liveChart.data.datasets[0].data = [];
    liveChart.update();
  } catch (err) {
    console.error(err);
    toast(err.message || 'Prediction error');
  } finally {
    hideLoading();
  }
};

// ---------------------
// Live price updater
// ---------------------
async function updateLiveChart() {
  const symbol = $('#symbolSelect').value;
  try {
    const { price, timestamp, feed, source } = await apiLive(symbol);
    if (feed)
      $('#feedBadge').textContent = `feed: ${feed}${
        source ? ' • ' + source : ''
      }`;

    if (typeof price === 'number' && timestamp) {
      liveData.labels.push(timestamp);
      liveData.data.push(price);
      if (liveData.labels.length > 60) {
        liveData.labels.shift();
        liveData.data.shift();
      }

      $('#livePriceText').textContent = `${symbol}: $${price}`;

      // Update chart data, don’t recreate
      liveChart.data.labels = liveData.labels;
      liveChart.data.datasets[0].label = `${symbol} Live`;
      liveChart.data.datasets[0].data = liveData.data;
      liveChart.update();
    }
  } catch (err) {
    console.warn('Live update error:', err.message);
    // no modal for periodic errors
  }
}
setInterval(updateLiveChart, 10000); // every 10s
updateLiveChart(); // run once on load

// ---------------------
// Chat flow
// ---------------------
$('#sendChat').onclick = async () => {
  const symbol = $('#symbolSelect').value;
  const recommendation = (
    $('#recText').textContent.split(': ')[1] || ''
  ).trim();
  const question = $('#chatInput').value.trim();
  if (!question) return;
  $('#chatInput').value = '';

  chatHistory.push({ from: 'you', text: question });
  renderChat();

  showLoading(`Asking about ${symbol}…`);
  try {
    const { reply } = await apiChat(symbol, recommendation, question);
    chatHistory.push({ from: 'stock', text: reply });
    renderChat();
  } catch (err) {
    console.error(err);
    chatHistory.push({ from: 'stock', text: `Error: ${err.message}` });
    renderChat();
  } finally {
    hideLoading();
  }
};

function renderChat() {
  const win = $('#chatWindow');
  win.innerHTML = '';
  chatHistory.forEach((msg) => {
    const row = document.createElement('div');
    const bubble = document.createElement('div');
    if (msg.from === 'you') {
      row.className = 'flex justify-end mb-2';
      bubble.className =
        'bg-blue-600 text-white px-3 py-2 rounded-lg max-w-[80%]';
    } else {
      row.className = 'flex justify-start mb-2';
      bubble.className =
        'bg-gray-200 text-gray-900 px-3 py-2 rounded-lg max-w-[80%]';
    }
    bubble.textContent = msg.text;
    row.appendChild(bubble);
    win.appendChild(row);
  });
  win.scrollTop = win.scrollHeight;
}

// ---------------------
// Reset
// ---------------------
$('#resetBtn').onclick = () => {
  chatHistory = [];
  renderChat();
  resetCharts();
};

// Initialize baseline charts on first load
initPredChart();
initLiveChart();
resetCharts();
