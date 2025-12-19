import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import datetime
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================
CONFIG = {
    'file_name': '2024-11-10_00-00-00--2024-11-11_00-00-00.parquet',
    
    # ID актива
    'target_instrument_id': 0, # BTC
    
    # Коды типов 
    'TYPE_SWAP': 0,   # Spot
    'TYPE_CALL': 2,   # Call Only
    'TYPE_PUT': 3,    # Ignored
    
    # --- ПАРАМЕТРЫ МОДЕЛИ ---
    'smoothing_factor': 25.0,    
    'risk_free_rate': 0.0,
    
    # --- НОВЫЕ ФИЛЬТРЫ ---
    # 1. Время: отсекаем < 2.2 дней
    'min_time_to_expiry': 0.006, 
    
    # 2. Ликвидность: Минимальная сумма заявок (Bid Qty + Ask Qty) в стакане
    'min_liquidity_depth': 1.0,  
    
    # 3. Спред: Если спред > 10%, считаем котировку нерыночной
    'max_spread_pct': 0.10
}

# ==========================================
# 2. МАТЕМАТИЧЕСКОЕ ЯДРО
# ==========================================
class BlackScholes:
    @staticmethod
    def d1(S, K, T, r, sigma):
        with np.errstate(divide='ignore', invalid='ignore'):
            return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S, K, T, r, sigma):
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def price(S, K, T, r, sigma, type_='C'):
        if T <= 0 or sigma <= 0: return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        if type_ == 'C':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def delta(S, K, T, r, sigma, type_='C'):
        if T <= 0 or sigma <= 0: return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.cdf(d1) if type_ == 'C' else norm.cdf(d1) - 1

    @staticmethod
    def gamma(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def implied_volatility(price, S, K, T, r, type_='C'):
        intrinsic = max(0, S - K) if type_ == 'C' else max(0, K - S)
        if price <= intrinsic + 1e-6: return np.nan
        try:
            return brentq(lambda sigma: BlackScholes.price(S, K, T, r, sigma, type_) - price, 0.01, 5.0)
        except:
            return np.nan

# ==========================================
# 3. ЗАГРУЗЧИК (С УМНОЙ ФИЛЬТРАЦИЕЙ)
# ==========================================
class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.spot = None
        self.timestamp = None

    def run_pipeline(self):
        print(f"[1] Чтение файла: {self.filepath}...")
        try:
            df_raw = pd.read_parquet(self.filepath)
        except FileNotFoundError:
            raise FileNotFoundError("Файл не найден.")

        # 1. Фильтр по Активу
        df_asset = df_raw[df_raw['instrument_id'] == CONFIG['target_instrument_id']].copy()
        
        # 2. Синхронизация времени
        self.timestamp = df_asset['exchange_ts'].max()
        human_time = datetime.datetime.fromtimestamp(self.timestamp / 1e9)
        print(f"    -> Временной срез: {human_time}")
        
        df_snap = df_asset[df_asset['exchange_ts'] == self.timestamp].copy()

        # 3. Поиск Спота (Swap)
        swap_rows = df_snap[df_snap['instrument_type'] == CONFIG['TYPE_SWAP']]
        if not swap_rows.empty:
            self.spot = (swap_rows['best_bid_price'].iloc[0] + swap_rows['best_ask_price'].iloc[0]) / 2
        else:
            self.spot = 84000.0
            print("    -> [WARN] Спот не найден. Использую заглушку.")
        print(f"    -> Spot Price: ${self.spot:,.2f}")

        # 4. Выбираем только call опционы
        self.df = df_snap[df_snap['instrument_type'] == CONFIG['TYPE_CALL']].copy()
        self.df['type'] = 'C'

        # 5. Фильтр по Времени (> 2.2 дней)
        ns_in_year = 1e9 * 365 * 24 * 3600
        self.df['T'] = (self.df['maturity'] - self.timestamp) / ns_in_year
        self.df = self.df[self.df['T'] > CONFIG['min_time_to_expiry']]

        # 6. Цены USD
        mid_btc = (self.df['best_bid_price'] + self.df['best_ask_price']) / 2
        self.df['price_usd'] = mid_btc * self.spot
        
        print(f"    -> Опционов до фильтрации: {len(self.df)}")

        # === НОВЫЕ ФИЛЬТРЫ ЛИКВИДНОСТИ ===
        
        # Фильтр по Глубине (Объему в стакане)
        self.df['depth'] = self.df['bid_amount_total'] + self.df['ask_amount_total']
        self.df = self.df[self.df['depth'] > CONFIG['min_liquidity_depth']]
        
        # Фильтр по Спреду
        spread_btc = self.df['best_ask_price'] - self.df['best_bid_price']
        mid_btc_calc = (self.df['best_bid_price'] + self.df['best_ask_price']) / 2
        
        # Защита от деления на ноль
        self.df['spread_pct'] = np.where(mid_btc_calc > 0, spread_btc / mid_btc_calc, 1.0)
        self.df = self.df[self.df['spread_pct'] < CONFIG['max_spread_pct']]
        
        # В) Веса
        self.df['weight'] = 1.0 / (spread_btc + 1e-9)

        # 7. Расчет IV
        print("[2] Расчет IV (Call Only + Smart Filter)...")
        self.df['raw_iv'] = self.df.apply(
            lambda row: BlackScholes.implied_volatility(
                row['price_usd'], self.spot, row['strike'], row['T'], CONFIG['risk_free_rate'], 'C'
            ), axis=1
        )
        self.df = self.df.dropna(subset=['raw_iv'])
        
        # убираем явные баги > 300%
        self.df = self.df[(self.df['raw_iv'] > 0.01) & (self.df['raw_iv'] < 3.0)]

        print(f"    -> Опционов ПОСЛЕ фильтрации: {len(self.df)}")
        
        return self.df

# ==========================================
# 4. ДВИЖОК МОДЕЛИ
# ==========================================
class VolatilityEngine:
    def __init__(self, data_loader):
        self.loader = data_loader
        self.spot = data_loader.spot
        self.model = None
        self.F = None; self.T = None

    def fit(self, maturity_ns):
        subset = self.loader.df[self.loader.df['maturity'] == maturity_ns].copy()
        
        # Сортировка по страйку
        train_data = subset.sort_values('strike')
        
        if len(train_data) < 5: return False
        
        self.T = train_data['T'].iloc[0]
        self.F = self.spot
        
        train_data['log_mon'] = np.log(train_data['strike'] / self.F)
        train_data['total_var'] = (train_data['raw_iv'] ** 2) * self.T
        
        # Сплайн
        self.model = UnivariateSpline(
            x=train_data['log_mon'],
            y=train_data['total_var'],
            w=train_data['weight'],
            s=CONFIG['smoothing_factor'],
            k=3
        )
        return True

    def get_iv(self, strike):
        if not self.model: return 0.0
        d = np.log(strike / self.F)
        var = max(1e-6, self.model(d))
        return np.sqrt(var / self.T)

    def price_contract(self, strike, type_='C'):
        sigma = self.get_iv(strike)
        price = BlackScholes.price(self.spot, strike, self.T, CONFIG['risk_free_rate'], sigma, type_)
        delta = BlackScholes.delta(self.spot, strike, self.T, CONFIG['risk_free_rate'], sigma, type_)
        gamma = BlackScholes.gamma(self.spot, strike, self.T, CONFIG['risk_free_rate'], sigma)
        return price, delta, sigma, gamma

    def validate_and_visualize(self):
        print("[4] Визуализация...")
        
        # Расширенный диапазон отрисовки
        strikes = np.linspace(self.spot * 0.2, self.spot * 3.5, 300)
        
        model_ivs = []
        prices = []
        
        for k in strikes:
            p, _, iv, _ = self.price_contract(k, 'C')
            model_ivs.append(iv)
            prices.append(p)
            
        prices = np.array(prices)
        dk = strikes[1] - strikes[0]
        
        # PDF
        pdf = np.gradient(np.gradient(prices, dk), dk)
        
        # Арбитраж
        min_pdf = np.min(pdf)
        print(f"    -> Min PDF Value: {min_pdf:.6e}") 
        if min_pdf < -1e-6:
            print("  -> Найден АРБИТРАЖ. Модель нестабильна на краях.")
        else:
            print("  -> [OK] Модель корректна.")
            
        # Графики
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Берем данные для графика
        real = self.loader.df[self.loader.df['maturity'] == self.loader.df['maturity'].mode()[0]]
        
        ax1.scatter(real['strike'], real['raw_iv'], c='green', marker='^', alpha=0.7, label='Liquid Calls')
        ax1.plot(strikes, model_ivs, c='blue', linewidth=3, label='Spline Model')
        ax1.axvline(self.spot, c='black', linestyle=':', label='Spot')
        
        ax1.set_title(f'Volatility Surface (Filtered by Volume) T={self.T:.2f}')
        ax1.set_xlabel('Strike'); ax1.set_ylabel('IV')
        ax1.legend(); ax1.grid(True, alpha=0.3)
        
        # PDF
        ax2.plot(strikes, pdf, c='darkgreen', linewidth=2, label='Implied PDF')
        ax2.fill_between(strikes, pdf, color='green', alpha=0.1)
        ax2.axvline(self.spot, c='black', linestyle=':', label='Spot')
        
        atm_iv = self.get_iv(self.spot)
        norm_pdf = (1 / (strikes * atm_iv * np.sqrt(self.T) * np.sqrt(2 * np.pi))) * \
                   np.exp(- (np.log(strikes / self.spot) + 0.5 * atm_iv**2 * self.T)**2 / (2 * atm_iv**2 * self.T))
        ax2.plot(strikes, norm_pdf, c='gray', linestyle='--', label='Black-Scholes')
        
        ax2.set_title('Market Probability Distribution')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ==========================================
# 5. ЗАПУСК
# ==========================================
if __name__ == "__main__":
    try:
        loader = DataLoader(CONFIG['file_name'])
        loader.run_pipeline()
        
        
        engine = VolatilityEngine(loader)
        
        # Выбор самой популярной экспирации
        if not loader.df.empty:
            target_mat = loader.df['maturity'].mode()[0]
            dt_obj = datetime.datetime.fromtimestamp(target_mat / 1e9)
            print(f"[3] Калибровка на дату: {dt_obj.date()}")
            
            if engine.fit(target_mat):
                engine.validate_and_visualize()
                
                print("\n" + "="*75)
                print("="*75)
                print(f"{'Strike':<10} | {'Type':<4} | {'Model IV':<9} | {'Price ($)':<12} | {'Delta':<8} | {'Gamma':<10}")
                print("-" * 75)
                
                spot = loader.spot
                # Показываем 5 страйков вокруг цены
                test_strikes = np.linspace(spot * 0.9, spot * 1.1, 5)
                
                for k in test_strikes:
                    k = int(k)
                    p, d, iv, g = engine.price_contract(k, 'C')
                    print(f"{k:<10} | {'Call':<4} | {iv:.2%}   | ${p:<11.2f} | {d:<8.4f} | {g:.8f}")
                print("-" * 75)
            else:
                print("Ошибка: Мало данных после фильтрации.")
        else:
            print("Ошибка: Все данные были отфильтрованы как неликвидные.")
            
    except Exception as e:
        print(f"\nERROR: {e}")