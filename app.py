import os
import json
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# LangChain + Gemini (via Google GenAI SDK)
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")

# — Alpaca client setup —
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
APCA_API_BASE = "https://paper-api.alpaca.markets"
alpaca = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET,
                       APCA_API_BASE, api_version='v2')

# — Gemini LLM setup —
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)


def fetch_stock_data(*args, **kwargs) -> str:
    """
    Returns JSON of the last `limit` daily closing prices for `symbol`.
    Handles calls as:
      fetch_stock_data("AAPL")
      fetch_stock_data({"SYMBOL": "AAPL", "LIMIT": 50})
      fetch_stock_data(SYMBOL="AAPL", limit=50)
      fetch_stock_data("SYMBOL='AAPL', LIMIT=100")
    """
    # Defaults
    symbol = None
    limit = 100

    # 1) If first arg is a dict
    if args and isinstance(args[0], dict):
        d = args[0]
        symbol = d.get("symbol") or d.get("SYMBOL")
        if d.get("limit") or d.get("LIMIT"):
            try:
                limit = int(d.get("limit") or d.get("LIMIT"))
            except:
                pass

    # 2) Keyword args (case-insensitive)
    for k, v in kwargs.items():
        kl = k.lower()
        if kl == "symbol":
            symbol = str(v)
        elif kl == "limit":
            try:
                limit = int(v)
            except:
                pass

    # 3) Single string positional arg
    if args and isinstance(args[0], str):
        s = args[0].strip()
        # a) Plain ticker?
        if re.fullmatch(r"[A-Za-z]{1,5}", s):
            symbol = s
        else:
            # b) Parse key=value segments
            for part in re.split(r",\s*", s):
                if "=" in part:
                    k, v = part.split("=", 1)
                    key = k.strip().strip("'\"").lower()
                    val = v.strip().strip("'\"")
                    if key == "symbol":
                        symbol = val
                    elif key == "limit":
                        try:
                            limit = int(val)
                        except:
                            pass
            # c) Fallback: pick first ticker-like token that's not 'symbol' or 'limit'
            if not symbol:
                for token in re.findall(r"[A-Za-z]{1,5}", s):
                    if token.lower() not in ("symbol", "limit"):
                        symbol = token
                        break

    if not symbol:
        raise ValueError("fetch_stock_data: no symbol could be parsed")

    symbol = symbol.upper()

    # 4) Fetch bars from Alpaca
    try:
        bars = alpaca.get_bars(symbol, tradeapi.TimeFrame.Day, limit=limit).df
    except Exception as e:
        # Return a JSON error for debugging rather than raising
        return json.dumps({"error": str(e), "data": []})

    records = bars.reset_index()[["timestamp", "close"]]
    records["timestamp"] = records["timestamp"].dt.strftime("%Y-%m-%d")
    return json.dumps(records.to_dict(orient="records"))


# Wrap it as a LangChain Tool
tools = [
    Tool(
        name="fetch_stock_data",
        func=fetch_stock_data,
        description=(
            "Fetches the last daily closing prices for a ticker as JSON. "
            "Call with fetch_stock_data('AAPL'), fetch_stock_data('SYMBOL=\"AAPL\", LIMIT=50'), "
            "or fetch_stock_data(SYMBOL='AAPL', limit=50), etc."
        )
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
    symbol = request.json.get("symbol", "AAPL").upper()
    prompt = (
        f"You are a senior quantitative market forecaster working on dummy data. "
        f"Use fetch_stock_data to get the last 100 days of '{symbol}' closing prices. "
        f"Based on cutting-edge statistical and ML methods, predict the next 5 trading days' closing prices. "
        f"Respond in JSON with:\n"
        f"  predictions: [{{date: 'YYYY-MM-DD', predicted_close: float}}, ...],\n"
        f"  recommendation: one of 'Buy', 'Hold', or 'Sell'\n"
        f"Do not use the keyword json in the json.'\n"
    )

    # Pass as a named input key
    raw_output = agent.run(input=prompt)
    print(raw_output)
    # Safe JSON parse
    try:
        result = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        result = {
            "predictions": [],
            "recommendation": "Unknown",
            "raw_output": raw_output.strip()
        }

    return jsonify(symbol=symbol, **result)


@app.route("/live_price")
def live_price():
    symbol = request.args.get("symbol", "AAPL").upper()
    trade = alpaca.get_latest_trade(symbol)
    price = trade.price
    # ts = datetime.fromisoformat(trade.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    return jsonify(symbol=symbol, price=price, timestamp=trade.timestamp)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    symbol = data.get("symbol", "AAPL")
    recommendation = data.get("recommendation", "")
    question = data.get("question", "")
    messages = [
        ("system", f"You are {symbol} stock itself. You just recommended '{recommendation}'. "
         "Answer the user’s questions in character, referencing your own advice."),
        ("human", question)
    ]
    resp = llm.invoke(messages)
    return jsonify(reply=resp.content)


if __name__ == "__main__":
    app.run(debug=True)
