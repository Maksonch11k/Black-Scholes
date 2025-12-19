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
# 1. КОНФИГУРАЦИЯ
# ==========================================
CONFIG = {
    'data_folder': '.', 
    'test_limit': None,
    
    'factors_to_test': [25.0],
    
    # Спецификация
    'target_instrument_id': 0,
    'TYPE_SWAP': 0,
    'TYPE_CALL': 2,
    
    # Фильтры
    'min_time_to_expiry': 0.006, 
    'min_liquidity_depth': 1.0,  
    'max_spread_pct': 0.10,      

    # Тест сколько не показываем модели, в данном случае скрываем 20%
    'test_size': 0.2
}

# ==========================================
# 2. МАТЕМАТИКА
# ==========================================
class BlackScholes:
    @staticmethod
    def price(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
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
# 3. МОДЕЛЬ
# ==========================================
class FixedFactorModel:
    def __init__(self, factor):
        self.factor = factor
        self.spline = None
        self.spot = None
        self.F = None; self.T = None
    
    def fit(self, df_train, spot):
        self.spot = spot
        self.F = spot 
        self.T = df_train['T'].iloc[0]
        
        df_train = df_train.sort_values('strike')
        df_train['log_mon'] = np.log(df_train['strike'] / self.F)
        df_train['total_var'] = (df_train['raw_iv'] ** 2) * self.T
        
        try:
            self.spline = UnivariateSpline(
                x=df_train['log_mon'],
                y=df_train['total_var'],
                w=df_train['weight'],
                s=self.factor,
                k=3
            )
            return True
        except:
            return False

    def predict_iv(self, strikes):
        if not self.spline: return np.zeros(len(strikes))
        d = np.log(strikes / self.F)
        var = np.maximum(1e-6, self.spline(d))
        return np.sqrt(var / self.T)
    
    def check_arbitrage(self):
        strikes = np.linspace(self.spot * 0.5, self.spot * 1.5, 100)
        ivs = self.predict_iv(strikes)
        prices = [BlackScholes.price(self.spot, k, self.T, 0, iv) for k, iv in zip(strikes, ivs)]
        dk = strikes[1] - strikes[0]
        pdf = np.gradient(np.gradient(prices, dk), dk)
        return np.min(pdf) < -1e-6

# ==========================================
# 4. ВАЛИДАТОР
# ==========================================
class ModelValidator:
    def __init__(self, folder):
        self.files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
        if CONFIG['test_limit']:
            self.files = self.files[:CONFIG['test_limit']]
        print(f"Файлов для проверки: {len(self.files)}")

    def evaluate_factor(self, factor):
        errors_iv = []
        arb_days = 0
        total_days = 0
        history = [] 
        
        print(f"\nЗапуск теста (Factor = {factor})...")
        print(f"{'Date':<12} | {'RMSE IV':<10} | {'Arb?':<5} | {'N Options'}")
        print("-" * 45)
        
        for filepath in self.files:
            try:
                df_raw = pd.read_parquet(filepath)
                df = df_raw[df_raw['instrument_id'] == CONFIG['target_instrument_id']].copy()
                if df.empty: continue
                
                timestamp = df['exchange_ts'].max()
                df = df[df['exchange_ts'] == timestamp].copy()
                
                swaps = df[df['instrument_type'] == CONFIG['TYPE_SWAP']]
                if swaps.empty: continue
                spot = (swaps['best_bid_price'].iloc[0] + swaps['best_ask_price'].iloc[0]) / 2
                
                # CALL опционы
                df = df[df['instrument_type'] == CONFIG['TYPE_CALL']].copy()
                
                ns_in_year = 1e9 * 365 * 24 * 3600
                df['T'] = (df['maturity'] - timestamp) / ns_in_year
                df = df[df['T'] > CONFIG['min_time_to_expiry']]
                
                if df.empty: continue
                target_mat = df['maturity'].mode()[0]
                df = df[df['maturity'] == target_mat].copy()
                
                mid = (df['best_bid_price'] + df['best_ask_price']) / 2
                df['price_usd'] = mid * spot
                
                # --- Фильтры на опционы ---
                df['depth'] = df['bid_amount_total'] + df['ask_amount_total']
                df = df[df['depth'] > CONFIG['min_liquidity_depth']]
                
                spread = df['best_ask_price'] - df['best_bid_price']
                df['spread_pct'] = spread / mid
                df = df[df['spread_pct'] < CONFIG['max_spread_pct']]
                
                df['weight'] = 1.0 / (spread + 1e-9)
                
                df['raw_iv'] = df.apply(lambda r: BlackScholes.implied_volatility(
                    r['price_usd'], spot, r['strike'], r['T'], 0), axis=1)
                df = df.dropna(subset=['raw_iv'])
                df = df[df['raw_iv'] < 3.0]
                
                if len(df) < 10: continue

                # OOS ТЕСТ
                mask = np.random.rand(len(df)) < CONFIG['test_size']
                train = df[~mask]
                test = df[mask].copy()
                
                if len(train) < 5 or len(test) < 1: continue
                
                model = FixedFactorModel(factor)
                if not model.fit(train, spot): continue
                
                pred_iv = model.predict_iv(test['strike'].values)
                act_iv = test['raw_iv'].values
                
                rmse = np.sqrt(np.mean((act_iv - pred_iv)**2))
                errors_iv.append(rmse)
                
                has_arb = model.check_arbitrage()
                if has_arb: arb_days += 1
                
                total_days += 1
                
                # Печать строки результата
                date_str = str(datetime.datetime.fromtimestamp(timestamp / 1e9).date())
                arb_str = "YES" if has_arb else ""
                print(f"{date_str:<12} | {rmse:.2%}     | {arb_str:<5} | {len(df)}")
                
                # Сохраняем историю
                history.append({
                    'date': datetime.datetime.fromtimestamp(timestamp / 1e9).date(),
                    'rmse': rmse,
                    'arb': has_arb
                })
                
            except:
                continue
        
        if total_days == 0: return None
        
        return {
            'factor': factor,
            'avg_rmse_iv': np.mean(errors_iv),
            'arb_rate': (arb_days / total_days) * 100,
            'history': pd.DataFrame(history)
        }

    def run(self):
        factor = CONFIG['factors_to_test'][0]
        res = self.evaluate_factor(factor)
        
        if res:
            print("\n" + "="*40)
            print("="*40)
            print(f"Средняя ошибка IV:     {res['avg_rmse_iv']:.2%}")
            print(f"Дней с Арбитражем:     {res['arb_rate']:.1f}%")
            
            # Графики
            df_hist = res['history']
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            ax1.plot(df_hist['date'], df_hist['rmse'], color='blue', label='IV RMSE')
            ax1.set_title(f'Динамика Ошибки во времени (Factor={factor})')
            ax1.set_ylabel('Ошибка IV')
            ax1.grid(True)
            
            ax2.bar(df_hist['date'], df_hist['arb'].astype(int), color='red', alpha=0.5, label='Arbitrage')
            ax2.set_title('Дни с обнаруженным арбитражем')
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['OK', 'ARB'])
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
        else:
            print("Не удалось провести тест.")

# ==========================================
# 5. ЗАПУСК
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)
    validator = ModelValidator(CONFIG['data_folder'])
    validator.run() 