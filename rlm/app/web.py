"""
RLM Strategy Web Dashboard
Flask application with API routes for live portfolio and backtest results.
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, render_template, jsonify, request

from app.config import INITIAL_CAPITAL, DB_PATH
from app.collector import fetch_all_live_prices

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_PATH = BASE_DIR / "data" / "backtest_result.json"


def _get_db():
    """Get a database connection with Row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _get_live_portfolio():
    """Get current portfolio state from DB."""
    conn = _get_db()
    try:
        # Open positions
        positions = [dict(r) for r in conn.execute(
            "SELECT * FROM positions WHERE status='OPEN'"
        ).fetchall()]

        # Calculate cash from trade history
        row = conn.execute("""
            SELECT
                COALESCE(SUM(CASE WHEN side='SELL' THEN amount - fees ELSE 0 END), 0)
              - COALESCE(SUM(CASE WHEN side='BUY'  THEN amount + fees ELSE 0 END), 0)
            AS net FROM trades
        """).fetchone()
        net = row['net'] if row else 0
        cash = INITIAL_CAPITAL + net

        # Fetch live prices for held stocks
        held_codes = [p['code'] for p in positions]
        live_prices = {}
        if held_codes:
            try:
                live_prices = fetch_all_live_prices(held_codes)
            except Exception as e:
                logger.warning("Failed to fetch live prices: %s", e)

        # Enrich positions with current price & unrealized P&L
        for p in positions:
            code = p['code']
            if code in live_prices and live_prices[code]['price'] > 0:
                cur_price = live_prices[code]['price']
            else:
                cur_price = p['entry_price']  # fallback
            p['current_price'] = cur_price
            p['current_value'] = cur_price * p['shares']
            p['pnl_pct'] = (cur_price / p['entry_price'] - 1) * 100 if p['entry_price'] else 0

        stock_value = sum(p['current_value'] for p in positions)
        total_value = cash + stock_value
        return_pct = (total_value / INITIAL_CAPITAL - 1) * 100

        return {
            "cash": round(cash),
            "stock_value": round(stock_value),
            "total_value": round(total_value),
            "return_pct": round(return_pct, 2),
            "positions": positions,
            "num_positions": len(positions),
        }
    finally:
        conn.close()


def _load_backtest_result():
    """Load backtest_result.json if it exists."""
    if RESULT_PATH.exists():
        try:
            with open(RESULT_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error("Failed to load backtest result: %s", e)
    return {}


def create_app():
    app = Flask(__name__,
                template_folder=str(BASE_DIR / "templates"))

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------
    @app.route('/')
    def dashboard():
        """Main dashboard - live portfolio only."""
        try:
            portfolio = _get_live_portfolio()
        except Exception as e:
            logger.error("Failed to load live portfolio: %s", e)
            portfolio = {
                "cash": 0, "stock_value": 0, "total_value": 0,
                "return_pct": 0.0, "positions": [], "num_positions": 0,
            }

        try:
            conn = _get_db()
            trades = [dict(r) for r in conn.execute(
                "SELECT * FROM trades ORDER BY created_at DESC LIMIT 20"
            ).fetchall()]
            market_row = conn.execute(
                "SELECT * FROM market_state ORDER BY date DESC LIMIT 1"
            ).fetchone()
            market = dict(market_row) if market_row else {}
            conn.close()
        except Exception as e:
            logger.error("Failed to load dashboard data: %s", e)
            trades, market = [], {}

        return render_template(
            'dashboard.html',
            portfolio=portfolio,
            trades=trades,
            market=market,
        )

    # ------------------------------------------------------------------
    # API: Portfolio
    # ------------------------------------------------------------------
    @app.route('/api/portfolio')
    def api_portfolio():
        """Return current live portfolio state."""
        if not DB_PATH.exists():
            return jsonify({
                "cash": 0, "stock_value": 0, "total_value": 0,
                "return_pct": 0.0, "positions": [], "num_positions": 0,
            })
        try:
            return jsonify(_get_live_portfolio())
        except Exception as e:
            logger.error("Failed to read portfolio: %s", e)
            return jsonify({"error": str(e)}), 500

    # ------------------------------------------------------------------
    # API: Trades
    # ------------------------------------------------------------------
    @app.route('/api/trades')
    def api_trades():
        """Return recent trades (default limit 100)."""
        limit = request.args.get('limit', 100, type=int)
        if not DB_PATH.exists():
            return jsonify([])
        try:
            conn = _get_db()
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            conn.close()
            return jsonify([dict(r) for r in rows])
        except Exception as e:
            logger.error("Failed to read trades: %s", e)
            return jsonify([])

    # ------------------------------------------------------------------
    # API: Snapshots
    # ------------------------------------------------------------------
    @app.route('/api/snapshots')
    def api_snapshots():
        """Return daily snapshots (last 90 days)."""
        if not DB_PATH.exists():
            return jsonify([])
        try:
            conn = _get_db()
            cutoff = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            rows = conn.execute(
                "SELECT * FROM daily_snapshot WHERE date >= ? ORDER BY date ASC",
                (cutoff,),
            ).fetchall()
            conn.close()
            return jsonify([dict(r) for r in rows])
        except Exception as e:
            logger.error("Failed to read snapshots: %s", e)
            return jsonify([])

    # ------------------------------------------------------------------
    # API: Market State
    # ------------------------------------------------------------------
    @app.route('/api/market')
    def api_market():
        """Return all market_state entries, newest first."""
        if not DB_PATH.exists():
            return jsonify([])
        try:
            conn = _get_db()
            rows = conn.execute(
                "SELECT * FROM market_state ORDER BY date DESC"
            ).fetchall()
            conn.close()
            return jsonify([dict(r) for r in rows])
        except Exception as e:
            logger.error("Failed to read market state: %s", e)
            return jsonify([])

    # ------------------------------------------------------------------
    # API: Backtest
    # ------------------------------------------------------------------
    @app.route('/api/backtest')
    def api_backtest():
        """Return backtest_result.json contents."""
        result = _load_backtest_result()
        if not result:
            return jsonify({})
        return jsonify(result)

    return app


if __name__ == '__main__':
    from app.config import PORT
    app = create_app()
    app.run(host='0.0.0.0', port=PORT, debug=True)
