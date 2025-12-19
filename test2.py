import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import datetime
import warnings
import os
import glob

warnings.filterwarnings('ignore')

# ==========================================
# 1. Конфигурация стратегии
# ==========================================
CONFIG = {
    'data_folder': '.', 
    'test_limit': None,
    
    # Спецификация данных
    'target_instrument_id': 0, # Биткоин
    'TYPE_SWAP': 0,            # Спот
    'TYPE_CALL': 2,            # Колл-опционы
    
    # Параметры модели 
    'smoothing_factor': 6.5, 
    'min_time_to_expiry': 0.006, 
    'min_liquidity_depth': 1.0,
    'max_spread_pct': 0.10,
    
    # --- Торговые правила ---
    # Порог входа: торгуем, если расхождение > 2% волатильности
    'trade_threshold_iv': 0.02, 
    
    # Размер ставки ($ на одну сделку)
    'bet_size_usd': 1000.0
}

# ==========================================
# 2. Математическое ядро
# ==========================================
class BlackScholes:
    @staticmethod
    def d1(S, K, T, r, sigma):
        with np.errstate(divide='ignore', invalid='ignore'):
            return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def delta(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return 0.0
        return norm.cdf(BlackScholes.d1(S, K, T, r, sigma))

    @staticmethod
    def price(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def implied_volatility(price, S, K, T, r):
        intrinsic = max(0, S - K)
        if price <= intrinsic + 1e-6: return np.nan
        try:
            return brentq(lambda sigma: BlackScholes.price(S, K, T, r, sigma) - price, 0.01, 5.0)
        except:
            return np.nan

# ==========================================
# 3. Модель (Сплайн)
# ==========================================
class SplineModel:
    def __init__(self):
        self.spline = None
        self.F = None; self.T = None
    
    def fit(self, df, spot):
        self.F = spot; self.T = df['T'].iloc[0]
        df = df.sort_values('strike')
        
        # Координаты: лог-денежность и общая дисперсия
        log_mon = np.log(df['strike'] / spot)
        total_var = (df['raw_iv'] ** 2) * self.T
        
        try:
            self.spline = UnivariateSpline(
                x=log_mon, 
                y=total_var, 
                w=df['weight'],
                s=CONFIG['smoothing_factor'], 
                k=3 # Кубический сплайн подходит для прайсинга
            )
            return True
        except:
            return False

    def get_iv(self, strike):
        if not self.spline: return 0.0
        d = np.log(strike / self.F)
        # Защита от отрицательной дисперсии
        var = np.maximum(1e-6, self.spline(d))
        return np.sqrt(var / self.T)

# ==========================================
# 4. Торговый симулятор
# ==========================================
class VolatilityTrader:
    def __init__(self, folder):
        self.files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
        if CONFIG['test_limit']:
            self.files = self.files[:CONFIG['test_limit']]
        print(f"Файлов для анализа: {len(self.files)}")

    def load_day(self, filepath):
        try:
            df_raw = pd.read_parquet(filepath)
            df = df_raw[df_raw['instrument_id'] == CONFIG['target_instrument_id']].copy()
            if df.empty: return None, None
            
            timestamp = df['exchange_ts'].max()
            df = df[df['exchange_ts'] == timestamp].copy()
            
            swaps = df[df['instrument_type'] == CONFIG['TYPE_SWAP']]
            if swaps.empty: return None, None
            spot = (swaps['best_bid_price'].iloc[0] + swaps['best_ask_price'].iloc[0]) / 2
            
            # Фильтр: только колл-опционы
            df = df[df['instrument_type'] == CONFIG['TYPE_CALL']].copy()
            
            ns_in_year = 1e9 * 365 * 24 * 3600
            df['T'] = (df['maturity'] - timestamp) / ns_in_year
            df = df[df['T'] > CONFIG['min_time_to_expiry']]
            if df.empty: return None, None
            
            # Выбираем одну дату экспирации
            target_mat = df['maturity'].mode()[0]
            df = df[df['maturity'] == target_mat].copy()
            mid = (df['best_bid_price'] + df['best_ask_price']) / 2
            df['price_usd'] = mid * spot
            
            # Фильтры качества данных
            df['depth'] = df['bid_amount_total'] + df['ask_amount_total']
            df = df[df['depth'] > CONFIG['min_liquidity_depth']]
            spread = df['best_ask_price'] - df['best_bid_price']
            df['spread_pct'] = spread / mid
            df = df[df['spread_pct'] < CONFIG['max_spread_pct']]
            df['weight'] = 1.0 / (spread + 1e-9)
            
            df['raw_iv'] = df.apply(lambda r: BlackScholes.implied_volatility(
                r['price_usd'], spot, r['strike'], r['T'], 0), axis=1)
            df = df.dropna(subset=['raw_iv'])
            
            # Фильтр экстремальных значений
            df = df[df['raw_iv'] < 3.0]
            
            return df, spot
        except:
            return None, None

    def run(self):
        cumulative_pnl = 0.0
        pnl_history = []
        dates = []
        trade_count = 0
        
        print(f"\nЗапуск стратегии арбитража волатильности...")
        print(f"{'Date':<12} | {'Trades':<6} | {'Daily P&L':<10} | {'Total P&L':<10}")
        print("-" * 50)

        for i in range(len(self.files) - 1):
            df_t0, S0 = self.load_day(self.files[i])
            df_t1, S1 = self.load_day(self.files[i+1])
            
            if df_t0 is None or df_t1 is None or len(df_t0) < 5: continue
            if df_t0['maturity'].iloc[0] != df_t1['maturity'].iloc[0]: continue
            
            # 1. Строим модель справедливой цены
            model = SplineModel()
            if not model.fit(df_t0, S0): continue
            
            common_strikes = np.intersect1d(df_t0['strike'], df_t1['strike'])
            
            daily_pnl = 0.0
            daily_trades = 0
            
            for k in common_strikes:
                row = df_t0[df_t0['strike'] == k].iloc[0]
                
                market_iv = row['raw_iv']
                market_price = row['price_usd']
                
                # Справедливая IV по модели
                fair_iv = model.get_iv(k)
                
                # --- Торговый сигнал ---
                diff = market_iv - fair_iv
                
                # Если рынок > модель (+ порог) -> продаем (дорого)
                if diff > CONFIG['trade_threshold_iv']:
                    position = -1 # Short Call
                # Если рынок < модель (- порог) -> покупаем (дешево)
                elif diff < -CONFIG['trade_threshold_iv']:
                    position = 1  # Long Call
                else:
                    continue # Нет сигнала
                
                # --- Исполнение сделки ---
                # Рассчитываем кол-во контрактов на $1000
                contracts = CONFIG['bet_size_usd'] / market_price
                
                # Хеджируем дельту (используем рыночную IV для надежности)
                delta = BlackScholes.delta(S0, k, row['T'], 0, market_iv)
                hedge_ratio = -position * contracts * delta # Купили опцион -> Продали фьючерс
                
                # --- Расчет P&L на следующий день ---
                row_next = df_t1[df_t1['strike'] == k].iloc[0]
                price_next = row_next['price_usd']
                
                # PnL от опциона
                pnl_opt = position * contracts * (price_next - market_price)
                # PnL от хеджа (фьючерса)
                pnl_hedge = hedge_ratio * (S1 - S0)
                
                trade_pnl = pnl_opt + pnl_hedge
                daily_pnl += trade_pnl
                daily_trades += 1
            
            # Фильтр аномалий (для чистоты статистики)
            if abs(daily_pnl) > 5000: continue

            cumulative_pnl += daily_pnl
            trade_count += daily_trades
            
            ts_val = df_t0['exchange_ts'].max()
            dt_str = str(datetime.datetime.fromtimestamp(ts_val/1e9).date())
            dates.append(dt_str)
            pnl_history.append(cumulative_pnl)
            
            print(f"{dt_str:<12} | {daily_trades:<6} | ${daily_pnl:<9.2f} | ${cumulative_pnl:<9.2f}")

        # --- Итоговый отчет ---
        print("\n" + "="*50)
        print("="*50)
        print(f"Всего сделок: {trade_count}")
        print(f"Итоговый P&L: ${cumulative_pnl:,.2f}")
        
        color = 'green' if cumulative_pnl > 0 else 'red'
        plt.figure(figsize=(12, 6))
        plt.plot(dates, pnl_history, color=color, linewidth=2)
        plt.fill_between(dates, pnl_history, color=color, alpha=0.1)
        plt.axhline(0, color='black', linestyle='--')
        plt.title(f'Cumulative P&L (Threshold={CONFIG["trade_threshold_iv"]:.1%})')
        plt.ylabel('Profit ($)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    trader = VolatilityTrader(CONFIG['data_folder'])
    trader.run()