import os
import json
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# LangChain + Gemini (via Google GenAI SDK)
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")
# SECRET_KEY for session cookies
app.secret_key = os.getenv("FLASK_SECRET_KEY", "replace-with-your-own-secret")

# — Alpaca client setup —
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
APCA_API_BASE = "https://paper-api.alpaca.markets"
alpaca = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET,
                       APCA_API_BASE, api_version='v2')

# — Gemini LLM setup —
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.2)


def rate_limit(endpoint: str, max_calls: int = 2, period: int = 60) -> bool:
    """
    Returns False if the user has already made `max_calls` in the last `period` seconds.
    Otherwise records the call and returns True.
    """
    now = datetime.utcnow().timestamp()
    key = f"rl_{endpoint}"
    timestamps = session.get(key, [])
    # Keep only timestamps within the window
    window_start = now - period
    timestamps = [ts for ts in timestamps if ts > window_start]
    if len(timestamps) >= max_calls:
        session[key] = timestamps  # prune old
        return False
    # record new call
    timestamps.append(now)
    session[key] = timestamps
    return True


def fetch_stock_data(*args, **kwargs) -> str:
    # ... (same robust implementation as before) ...
    # (omitted here for brevity; keep your final version)
    pass


tools = [
    Tool(
        name="fetch_stock_data",
        func=fetch_stock_data,
        description="Fetches daily closes JSON. Supports many arg formats."
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=False
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not rate_limit("predict"):
        return jsonify(error="Rate limit exceeded: max 2 predictions per minute"), 429

    symbol = request.json.get("symbol", "AAPL").upper()
    prompt = (
        f"You are a senior financial  market forecaster who works on a dummy dataset. "
        f"Use fetch_stock_data to get the last 100 days of '{symbol}' closing prices. "
        f"Based on cutting-edge statistical and ML methods, predict the next 5 trading days' closing prices. "
        f"Respond in JSON with:\n"
        f"  predictions: [{{date: 'YYYY-MM-DD', predicted_close: float}}, ...],\n"
        f"  recommendation: one of 'Buy', 'Hold', or 'Sell'\n"
        f"Do not use the word json in the output.'\n"
    )
    raw_output = agent.run(input=prompt)
    print(raw_output)
    try:
        result = json.loads(raw_output)
    except:
        result = {"predictions": [], "recommendation": "Unknown",
                  "raw_output": raw_output.strip()}
    return jsonify(symbol=symbol, **result)


@app.route("/chat", methods=["POST"])
def chat():
    if not rate_limit("chat"):
        return jsonify(error="Rate limit exceeded: max 2 chats per minute"), 429

    data = request.json or {}
    symbol = data.get("symbol", "AAPL")
    recommendation = data.get("recommendation", "")
    question = data.get("question", "")
    messages = [
        ("system",
         f"You are {symbol} stock itself. You just recommended '{recommendation}'."),
        ("human", question)
    ]
    resp = llm.invoke(messages)
    return jsonify(reply=resp.content)

# ... live_price route stays the same ...


@app.route("/live_price")
def live_price():
    symbol = request.args.get("symbol", "AAPL").upper()
    trade = alpaca.get_latest_trade(symbol)
    price = trade.price
    # ts = datetime.fromisoformat(trade.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    return jsonify(symbol=symbol, price=price, timestamp=trade.timestamp)


if __name__ == "__main__":
    app.run(debug=True)
