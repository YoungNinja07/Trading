# Backtest XAUUSD (złoto) 2016–2026

Dane: świece 1-minutowe XAUUSD (HistData przez Hugging Face, `fokan/xauusd-2009-2026`), 2016-01 → 2026-01.

```
pip install pandas numpy numba pyarrow
# pobierz DAT_MT_XAUUSD_M1_*.csv do data/raw/
python3 backtest/run_all.py          # wszystkie interwały -> results/all_runs.csv
```

- Interwały: 1m, 5m, 15m, 30m, 1h, 4h, 1d
- Strategie: EMA cross, Donchian breakout, RSI mean reversion, Bollinger, MACD, Opening Range Breakout
- Wyjścia: SL 1–3 ATR, TP 0–4 ATR, long+short lub tylko long
- Koszt 0,35 USD/oz na transakcję, wejście na otwarciu następnej świecy, SL przed TP w tej samej świecy
- Wybór parametrów na 2016–2021, weryfikacja na 2022–2026 (dane niewidziane)

Najlepszy stabilny kandydat: **Donchian 55, 4h, tylko long, SL 2×ATR(14), TP 4×ATR(14)**.
