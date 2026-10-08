"""Schema-compatible synthetic RAN data, used ONLY for tests / smoke runs (never for reported results)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import RENAME

INV = {v: k for k, v in RENAME.items()}


def make_synthetic(n_sites: int = 12, days: int = 70, seed: int = 0, start: str = "2025-02-24") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=days * 96, freq="15min")
    T = len(ts)
    hod = ts.hour.to_numpy() + ts.minute.to_numpy() / 60
    dow = ts.dayofweek.to_numpy()
    profile = 0.15 + np.exp(-0.5 * ((hod - 12.5) / 3.2) ** 2) + 1.25 * np.exp(-0.5 * ((hod - 19.5) / 2.6) ** 2)
    profile /= profile.max()
    weekend = np.where(dow >= 5, 0.8, 1.0)
    frames = []
    for s in range(n_sites):
        e_bb = rng.integers(60, 120)
        for sec in (1, 2, 3):
            base = rng.lognormal(np.log(14), 0.6)
            noise = np.zeros(T)
            eps = rng.normal(0, 0.18, T)
            for i in range(1, T):
                noise[i] = 0.85 * noise[i - 1] + eps[i]
            rb = np.clip(base * profile * weekend * np.exp(noise), 0, 100)
            idle = rb < 0.6
            rb[idle] = 0.0
            users = rng.poisson(rb * 0.35).astype(float)
            vol = rb * rng.lognormal(np.log(18), 0.15, T) * (users > 0)
            df = pd.DataFrame({
                "site": f"Site {100 + s}", "sector": sec, "ts": ts,
                "energy_ru": 20 + 0.12 * rb + rng.normal(0, 1.5, T), "energy_bb": float(e_bb),
                "max_users_dl": users + rng.poisson(1.0, T), "max_users_ul": users + rng.poisson(2.0, T),
                "vol_dl": vol, "vol_ul": vol * 0.08, "max_rrc": users * 2 + rng.poisson(3.0, T),
                "rb": np.where(idle, np.nan, rb), "cqi1": np.where(users > 0, rng.normal(12, 1.4, T), 0.0),
                "cqi2": np.where(users > 0, rng.normal(12, 1.4, T), 0.0), "cqi3": np.where(users > 0, rng.normal(10, 1.4, T), 0.0),
                "cqi4": np.where(users > 0, rng.normal(9.5, 1.4, T), 0.0), "rrc": users * 1.6 + 0.5,
                "users_ul": users * 0.4, "users_dl": users, "mimo": 1.2 + 0.7 * rng.random(T),
            })
            for _ in range(rng.integers(1, 4)):          # outages: rows missing entirely
                a = rng.integers(0, T - 60)
                df = df.drop(df.index[a: a + rng.integers(4, 48)])
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    dup_exact = out.sample(frac=0.02, random_state=seed)
    dup_alt = out.sample(frac=0.02, random_state=seed + 1).copy()
    dup_alt[["vol_dl", "users_dl"]] *= 0.5
    out = pd.concat([out, dup_exact, dup_alt], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out = out.rename(columns=INV)
    out["Timestamp"] = out["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out
