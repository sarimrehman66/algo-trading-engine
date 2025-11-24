import streamlit as st
import pandas as pd
import time
import random
import collections
from collections import deque
from datetime import datetime
import requests

# ==========================================
# PART 1: DATA STRUCTURES (Backend Logic)
# ==========================================
# (This part is IDENTICAL to your previous code, just moved to web)

class CircularBuffer:
    """DSA: Circular Queue for O(1) streaming history"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = deque(maxlen=capacity)

    def add(self, value):
        self.queue.append(value)

    def get_all(self):
        return list(self.queue)

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    """DSA: Trie for O(L) Ticker Search"""
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        self.node = node.is_end_of_word = True

    def search_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        return self._find_words_from_node(node, prefix)

    def _find_words_from_node(self, node, prefix):
        words = []
        if node.is_end_of_word:
            words.append(prefix)
        for char, child_node in node.children.items():
            words.extend(self._find_words_from_node(child_node, prefix + char))
        return words

class StrategyParser:
    """DSA: Stack (Shunting Yard) for parsing strategies"""
    def __init__(self):
        self.precedence = {'AND': 1, 'OR': 0, '>': 2, '<': 2, '==': 2}

    def evaluate(self, strategy_str, indicators):
        try:
            tokens = strategy_str.replace('(', ' ( ').replace(')', ' ) ').split()
            postfix = self._shunting_yard(tokens)
            return self._evaluate_postfix(postfix, indicators)
        except Exception:
            return False

    def _shunting_yard(self, tokens):
        output_queue = []
        operator_stack = []
        for token in tokens:
            if token.replace('.','',1).isdigit():
                output_queue.append(float(token))
            elif token in ['RSI', 'PRICE', 'MA']:
                output_queue.append(token)
            elif token in self.precedence:
                while (operator_stack and operator_stack[-1] in self.precedence and
                       self.precedence[operator_stack[-1]] >= self.precedence[token]):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output_queue.append(operator_stack.pop())
                operator_stack.pop()
        while operator_stack:
            output_queue.append(operator_stack.pop())
        return output_queue

    def _evaluate_postfix(self, postfix, indicators):
        stack = []
        for token in postfix:
            if isinstance(token, float):
                stack.append(token)
            elif token in indicators:
                stack.append(indicators[token])
            elif token in self.precedence:
                val2 = stack.pop()
                val1 = stack.pop()
                if token == '>': stack.append(val1 > val2)
                elif token == '<': stack.append(val1 < val2)
                elif token == '==': stack.append(val1 == val2)
                elif token == 'AND': stack.append(val1 and val2)
                elif token == 'OR': stack.append(val1 or val2)
        return stack[0] if stack else False

class MarketGraph:
    """DSA: Graph + BFS for Risk Clustering"""
    def __init__(self):
        self.adj_list = collections.defaultdict(list)
        self.edges_info = [] 

    def build_graph(self, coin_bots):
        self.adj_list.clear()
        self.edges_info.clear()
        tickers = list(coin_bots.keys())
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                t1, t2 = tickers[i], tickers[j]
                corr = self._calculate_correlation(coin_bots[t1].history.get_all(), 
                                                 coin_bots[t2].history.get_all())
                
                if corr > 0.80:
                    self.adj_list[t1].append(t2)
                    self.adj_list[t2].append(t1)
                    self.edges_info.append(f"{t1} <--> {t2} (Corr: {corr:.2f})")

    def find_risk_clusters(self):
        visited = set()
        clusters = []
        for node in self.adj_list:
            if node not in visited:
                component = []
                queue = deque([node])
                visited.add(node)
                while queue:
                    curr = queue.popleft()
                    component.append(curr)
                    for neighbor in self.adj_list[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                clusters.append(component)
        return clusters

    def _calculate_correlation(self, data_x, data_y):
        if len(data_x) != len(data_y) or len(data_x) < 2: return 0
        n = len(data_x)
        sum_x = sum(data_x)
        sum_y = sum(data_y)
        sum_xy = sum(x*y for x,y in zip(data_x, data_y))
        sum_x2 = sum(x**2 for x in data_x)
        sum_y2 = sum(y**2 for y in data_y)
        try:
            numerator = n * sum_xy - sum_x * sum_y
            denominator = math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
            return numerator / denominator if denominator != 0 else 0
        except ValueError:
            return 0

class CoinBot:
    def __init__(self, ticker, strategy):
        self.ticker = ticker
        self.history = CircularBuffer(capacity=20)
        self.strategy = strategy
        self.current_price = 0
        self.indicators = {'RSI': 50, 'MA': 0}
        self.last_action = "WAIT"
        self.holdings = 0.0
        self.last_update_time = "N/A"

    def update(self, price):
        self.current_price = price
        self.history.add(price)
        self.last_update_time = datetime.now().strftime("%H:%M:%S")
        self._calculate_indicators()

    def _calculate_indicators(self):
        data = self.history.get_all()
        if not data: return
        self.indicators['PRICE'] = self.current_price
        self.indicators['MA'] = sum(data) / len(data)
        if len(data) > 1:
            change = data[-1] - data[-2]
            if change > 0: self.indicators['RSI'] = min(100, self.indicators['RSI'] + 5)
            else: self.indicators['RSI'] = max(0, self.indicators['RSI'] - 5)

# ==========================================
# PART 2: WEB APPLICATION (Streamlit)
# ==========================================

# --- 1. Setup Session State (Memory for Web App) ---
if 'registry' not in st.session_state:
    st.session_state.registry = {} # Hash Map for Coins
if 'cash' not in st.session_state:
    st.session_state.cash = 10000.00
if 'trie' not in st.session_state:
    t = Trie()
    coins = ["BTC", "ETH", "SOL", "ADA", "DOGE", "SHIB", "PEPE", "XRP", "LTC", "LINK", "UNI", "DOT", "MATIC"]
    for c in coins: t.insert(c)
    st.session_state.trie = t
if 'parser' not in st.session_state:
    st.session_state.parser = StrategyParser()
if 'graph' not in st.session_state:
    st.session_state.graph = MarketGraph()
if 'logs' not in st.session_state:
    st.session_state.logs = []

def add_log(msg):
    st.session_state.logs.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def fetch_live_prices():
    """Fetch prices from Binance API"""
    try:
        url = "https://api.binance.com/api/v3/ticker/price"
        response = requests.get(url, timeout=5)
        data = response.json()
        return {item['symbol']: float(item['price']) for item in data}
    except:
        return {}

def execute_logic():
    """Core Engine Loop: Updates all bots once"""
    market_prices = fetch_live_prices()
    
    # List conversion to avoid runtime dictionary size change errors
    active_bots = list(st.session_state.registry.items())
    
    for ticker, bot in active_bots:
        symbol = f"{ticker}USDT"
        if symbol in market_prices:
            new_price = market_prices[symbol]
            bot.update(new_price)
            
            # Evaluate Strategy (Stack)
            should_buy = st.session_state.parser.evaluate(bot.strategy, bot.indicators)
            
            if should_buy and bot.last_action != "BOUGHT":
                # Buy Logic
                cost = 1000.0
                if st.session_state.cash >= cost:
                    amount = cost / new_price
                    st.session_state.cash -= cost
                    bot.holdings += amount
                    bot.last_action = "BOUGHT"
                    add_log(f"✅ BUY: {amount:.4f} {ticker} @ ${new_price:.2f}")
            elif not should_buy:
                bot.last_action = "WAIT"

# --- 2. UI Layout ---

st.set_page_config(page_title="Algo Trading Engine", layout="wide")

# Sidebar: Controls
st.sidebar.title("⚙️ Engine Controls")
st.sidebar.metric("Paper Wallet", f"${st.session_state.cash:,.2f}")

# Add Coin Section
st.sidebar.subheader("Add Asset")
search_term = st.sidebar.text_input("Search Ticker (e.g., B, E, SHIB)", "").upper()

# Trie Search Logic
found_coins = []
if search_term:
    found_coins = st.session_state.trie.search_prefix(search_term)
    st.sidebar.caption(f"Trie found: {', '.join(found_coins[:5])}")

selected_coin = st.sidebar.selectbox("Select Coin", found_coins if found_coins else ["Type to search..."])
strategy_input = st.sidebar.text_input("Strategy (Stack Logic)", "PRICE > MA AND RSI > 60")

if st.sidebar.button("Add Coin to Engine"):
    if selected_coin and selected_coin != "Type to search...":
        if selected_coin not in st.session_state.registry:
            new_bot = CoinBot(selected_coin, strategy_input)
            # Init price
            prices = fetch_live_prices()
            start_price = prices.get(f"{selected_coin}USDT", 0.0)
            new_bot.update(start_price)
            st.session_state.registry[selected_coin] = new_bot
            add_log(f"Registered {selected_coin}")
        else:
            st.sidebar.error("Coin already active")

# Remove Coin
if st.session_state.registry:
    coin_to_remove = st.sidebar.selectbox("Remove Coin", list(st.session_state.registry.keys()))
    if st.sidebar.button("Remove Selected"):
        del st.session_state.registry[coin_to_remove]
        add_log(f"Removed {coin_to_remove}")
        st.rerun()

# --- 3. Main Dashboard ---

st.title("📊 Real-Time Algorithmic Trading Engine")
st.markdown("This dashboard uses **Tries** for search, **Stacks** for strategy parsing, and **Graphs** for risk analysis.")

# Auto-Refresh Loop
if st.checkbox("🔴 Start Live Data Feed (Binance API)", value=False):
    time.sleep(2) # 2 Second Interval
    execute_logic()
    st.rerun()

# Data Table
if st.session_state.registry:
    data = []
    for t, bot in st.session_state.registry.items():
        val = bot.holdings * bot.current_price
        data.append({
            "Ticker": t,
            "Price ($)": f"{bot.current_price:,.4f}",
            "Holdings": f"{bot.holdings:.4f}",
            "Value ($)": f"{val:.2f}",
            "RSI": bot.indicators['RSI'],
            "MA": f"{bot.indicators['MA']:.2f}",
            "Signal": bot.last_action,
            "Last Update": bot.last_update_time
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No coins active. Use the sidebar to add coins (e.g., BTC, ETH).")

# --- 4. Risk Graph Section ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🕸️ Market Risk Graph (BFS)")
    if st.button("Scan for Risk Clusters"):
        st.session_state.graph.build_graph(st.session_state.registry)
        clusters = st.session_state.graph.find_risk_clusters()
        
        if not clusters:
            st.write("No active data to analyze.")
        else:
            for i, cluster in enumerate(clusters):
                if len(cluster) > 1:
                    st.error(f"⚠️ **CLUSTER {i+1} (High Risk):** {', '.join(cluster)}")
                    st.caption("These assets are highly correlated (>80%). Diversify!")
                else:
                    st.success(f"✅ **GROUP {i+1} (Independent):** {cluster[0]}")

with col2:
    st.subheader("📜 System Logs")
    log_text = "\n".join(st.session_state.logs[:10])
    st.text_area("Engine Output", log_text, height=200, disabled=True)