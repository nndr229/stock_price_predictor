import os
import json
import re
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame, APIError

# LangChain + Gemini
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI

# -----------------------
# Environment & App Setup
# -----------------------
load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "replace-with-your-own-secret")

# Alpaca
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
APCA_API_BASE = os.getenv(
    "APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
# "iex" for free, "sip" for paid
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").lower()

alpaca = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET,
                       APCA_API_BASE, api_version='v2')

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)

# -----------
# Utilities
# -----------


def rate_limit(endpoint: str, max_calls: int = 2, period: int = 60) -> bool:
    """
    Returns False if the user has already made `max_calls` in the last `period` seconds.
    Otherwise records the call and returns True.
    """
    now = datetime.utcnow().timestamp()
    key = f"rl_{endpoint}"
    timestamps = session.get(key, [])
    window_start = now - period
    timestamps = [ts for ts in timestamps if ts > window_start]
    if len(timestamps) >= max_calls:
        session[key] = timestamps
        return False
    timestamps.append(now)
    session[key] = timestamps
    return True


def _session_key(symbol: str) -> str:
    return f"ctx_{symbol.upper()}"


def _ts_to_datestr(ts) -> str:
    # ts could be pandas.Timestamp or datetime; normalize
    try:
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return str(ts)[:10]


# -------------------------
# Robust symbol arg parsing
# -------------------------
EXCLUDE_TOKENS = {"SYMBOL", "DAYS"}


def _clean_ticker(s: str) -> str:
    s = (s or "").strip().upper()
    # keep only letters/digits/.- and trim length
    s = re.sub(r"[^A-Z0-9.\-]", "", s)
    return s[:10]


def _parse_symbol_days(*args, **kwargs):
    # days
    days = kwargs.get("days") or kwargs.get("DAYS") or 100
    try:
        days = int(days)
    except Exception:
        days = 100

    # prefer kwargs for symbol
    raw = kwargs.get("symbol") or kwargs.get("SYMBOL")
    if isinstance(raw, str) and raw.strip():
        return _clean_ticker(raw), days

    # otherwise, inspect first positional arg (LLM may shove everything there)
    raw = str(args[0]) if args else ""
    u = raw.upper()

    # key=value pattern
    m = re.search(r"(?:SYMBOL|TICKER)\s*[:=]\s*['\"]?([A-Z0-9.\-]{1,10})", u)
    if m:
        return _clean_ticker(m.group(1)), days

    # quoted token
    m = re.search(r"['\"]([A-Z][A-Z0-9.\-]{0,9})['\"]", u)
    if m and m.group(1) not in EXCLUDE_TOKENS:
        return _clean_ticker(m.group(1)), days

    # first bare token that isn't a label
    for tok in re.findall(r"\b[A-Z][A-Z0-9.\-]{0,9}\b", u):
        if tok not in EXCLUDE_TOKENS:
            return _clean_ticker(tok), days

    # fallback: use raw
    return _clean_ticker(raw), days

# -----------------------
# Market data construction
# -----------------------


def _safe_latest_trade(symbol: str):
    try:
        lt = alpaca.get_latest_trade(symbol, feed=ALPACA_DATA_FEED)
        return {
            "price": float(lt.price),
            "timestamp": getattr(lt, "timestamp", None),
            "source": "latest_trade"
        }
    except APIError as e:
        if "no trade found" in str(e).lower():
            return None
        raise


def _ts_to_datestr(ts) -> str:
    # ts could be pandas.Timestamp or datetime
    try:
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return str(ts)[:10]


def _ts_to_isostr(ts) -> str:
    try:
        return ts.isoformat()
    except Exception:
        return str(ts)


def _safe_latest_trade(symbol: str):
    try:
        lt = alpaca.get_latest_trade(symbol, feed=ALPACA_DATA_FEED)
        return {
            "price": float(lt.price),
            "timestamp": _ts_to_isostr(getattr(lt, "timestamp", None)),
            "source": "latest_trade"
        }
    except APIError as e:
        if "no trade found" in str(e).lower():
            return None
        raise


def build_market_context(symbol: str, days: int = 100) -> dict:
    symbol = symbol.upper()
    end = datetime.utcnow()
    start = end - timedelta(days=days * 2)

    bars = alpaca.get_bars(
        symbol,
        TimeFrame.Day,
        start.isoformat() + "Z",
        end.isoformat() + "Z",
        limit=days,
        adjustment="raw",
        feed=ALPACA_DATA_FEED,
    )

    closes = []
    for b in bars:
        dt = getattr(b, "t", None) or getattr(b, "timestamp", None)
        if dt is None:
            continue
        closes.append({"date": _ts_to_datestr(dt), "close": float(b.c)})

    closes = sorted(closes, key=lambda x: x["date"])[:days]

    latest = _safe_latest_trade(symbol)
    if latest is None:
        if closes:
            latest = {
                "price": float(closes[-1]["close"]),
                "timestamp": closes[-1]["date"],  # already string
                "source": "last_close_fallback"
            }
        else:
            latest = {"price": None, "timestamp": None, "source": "none"}

    ctx = {
        "symbol": symbol,
        "closes": closes,
        "latest_trade": latest,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "feed": ALPACA_DATA_FEED,
    }
    session[_session_key(symbol)] = ctx
    return ctx


def fetch_stock_data(*args, **kwargs) -> str:
    """
    LLM-facing tool. Accepts messy args, returns JSON string context:
    {
      "symbol": "AAPL",
      "closes": [{"date":"YYYY-MM-DD","close":123.45}, ...],
      "latest_trade": {"price": 123.45, "timestamp": "...", "source": "..."},
      "fetched_at": "...",
      "feed": "iex"
    }
    """
    symbol, days = _parse_symbol_days(*args, **kwargs)
    ctx = build_market_context(symbol, days)
    return json.dumps(ctx)


tools = [
    Tool(
        name="fetch_stock_data",
        func=fetch_stock_data,
        description="Fetch last N daily closes and latest trade. Args accepted: symbol (str), days (int)."
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=False
)

# -------
# Routes
# -------


@app.route("/")
def index():
    # Provide your own templates/index.html as needed
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not rate_limit("predict"):
        return jsonify(error="Rate limit exceeded: max 2 predictions per minute"), 429

    symbol = request.json.get("symbol", "AAPL").upper()

    prompt = (
        f"You are a senior financial market forecaster working strictly from live data. "
        f"First, call fetch_stock_data(symbol='{symbol}', days=100). "
        f"Use only the returned JSON values to compute the next 5 trading days' closing prices. "
        f"Return pure JSON (no markdown) with exactly: "
        f'{{"predictions":[{{"date":"YYYY-MM-DD","predicted_close":float}},{{...}} x5],'
        f'"recommendation":"Buy|Hold|Sell"}}'
    )

    raw_output = agent.run(input=prompt)
    print(raw_output)
    # Extract final JSON object if model added text
    match = re.search(r"\{.*\}\s*$", raw_output, flags=re.S)
    payload = match.group(0) if match else raw_output.strip()

    try:
        result = json.loads(payload)
    except Exception:
        result = {"predictions": [], "recommendation": "Unknown",
                  "raw_output": raw_output.strip()}

    # Save shared context + predictions for /chat
    session["last_prediction"] = {
        "symbol": symbol,
        "context": session.get(_session_key(symbol)),
        "result": result
    }
    return jsonify(symbol=symbol, **result)


@app.route("/chat", methods=["POST"])
def chat():
    if not rate_limit("chat"):
        return jsonify(error="Rate limit exceeded: max 2 chats per minute"), 429

    data = request.json or {}
    symbol = data.get("symbol", "AAPL").upper()
    question = data.get("question", "")

    # Ensure we're using the same snapshot context as /predict (or build if missing)
    lp = session.get("last_prediction")
    if not lp or lp.get("symbol") != symbol:
        ctx = build_market_context(symbol, days=100)
        session["last_prediction"] = {
            "symbol": symbol, "context": ctx, "result": {}}
        lp = session["last_prediction"]

    ctx = lp.get("context", {}) or {}
    prior_result = lp.get("result", {})
    prior_reco = prior_result.get("recommendation", "Unknown")

    messages = [
        ("system",
         "You are a financial analyst. Use ONLY the provided context. "
         f"symbol={symbol}; feed={ctx.get('feed')}; "
         f"latest_trade_price={ctx.get('latest_trade', {}).get('price')}; "
         f"last_close={ctx.get('closes', [{}])[-1].get('close') if ctx.get('closes') else None}; "
         f"prior_recommendation='{prior_reco}'. "
         "If something is unknown, say you don't know."),
        ("human", question)
    ]
    resp = llm.invoke(messages)
    return jsonify(reply=resp.content)


@app.route("/live_price")
def live_price():
    symbol = request.args.get("symbol", "AAPL").upper()
    try:
        lt = alpaca.get_latest_trade(symbol, feed=ALPACA_DATA_FEED)
        price = float(lt.price)
        ts = getattr(lt, "timestamp", None)
        return jsonify(symbol=symbol, price=price, timestamp=ts, feed=ALPACA_DATA_FEED)
    except APIError as e:
        if "no trade found" in str(e).lower():
            # Fallback to last close if available in session
            ctx = session.get(_session_key(symbol)) or {}
            closes = ctx.get("closes") or []
            if closes:
                return jsonify(
                    symbol=symbol,
                    price=float(closes[-1]["close"]),
                    timestamp=closes[-1]["date"],
                    feed=ALPACA_DATA_FEED,
                    source="last_close_fallback"
                ), 206
        # Unknown error
        return jsonify(error=str(e)), 502


# -----------
# Entrypoint
# -----------
if __name__ == "__main__":
    # For production, use a proper WSGI server (gunicorn/uvicorn) and set DEBUG via env.
    app.run(debug=True)
