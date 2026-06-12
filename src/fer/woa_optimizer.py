"""
src/fer/woa_optimizer.py
Whale Optimization Algorithm (WOA) untuk hyperparameter search FER.

WOA dipilih karena:
  - Ringan: konvergen dengan sedikit agent (8-10)
  - Tidak butuh GPU — setiap candidate dievaluasi dengan quick-train CPU
  - Cocok untuk continuous search space

Referensi:
  Mirjalili & Lewis (2016) — The Whale Optimization Algorithm
  Advances in Engineering Software, 95, 51-67.
"""
import math
import time
import numpy as np
from typing import Callable, Dict, List, Tuple, Optional


class WOAOptimizer:
    """
    Whale Optimization Algorithm untuk mencari hyperparameter terbaik.

    Args:
        bounds:     dict {param_name: (min_val, max_val)}
        n_agents:   jumlah agen (whale). Default 8 untuk CPU.
        max_iter:   iterasi maksimum. Default 12 untuk CPU.
        seed:       random seed
        minimize:   jika True, minimasi fitness (misal loss)
                    jika False, maksimasi fitness (misal F1-score)
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        n_agents: int = 8,
        max_iter: int = 12,
        seed: int = 42,
        minimize: bool = False,
    ):
        self.bounds   = bounds
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.minimize = minimize
        self.rng      = np.random.default_rng(seed)

        self.param_names = list(bounds.keys())
        self.dim         = len(self.param_names)
        self.lb = np.array([bounds[k][0] for k in self.param_names], dtype=float)
        self.ub = np.array([bounds[k][1] for k in self.param_names], dtype=float)

        # History
        self.best_params: Optional[Dict] = None
        self.best_fitness: float = float("inf") if minimize else float("-inf")
        self.history: List[Dict] = []

    def _init_population(self) -> np.ndarray:
        """Inisialisasi posisi agen secara acak dalam bounds."""
        pop = self.rng.uniform(0, 1, size=(self.n_agents, self.dim))
        return self.lb + pop * (self.ub - self.lb)

    def _clip(self, pos: np.ndarray) -> np.ndarray:
        return np.clip(pos, self.lb, self.ub)

    def _decode(self, pos: np.ndarray) -> Dict:
        """Konversi posisi kontinu ke dict hyperparameter."""
        return {name: float(pos[i]) for i, name in enumerate(self.param_names)}

    def _is_better(self, new_val: float, old_val: float) -> bool:
        if self.minimize:
            return new_val < old_val
        return new_val > old_val

    def optimize(
        self,
        fitness_fn: Callable[[Dict], float],
        verbose: bool = True,
        callback: Optional[Callable] = None,
    ) -> Dict:
        """
        Jalankan optimasi WOA.

        Args:
            fitness_fn: fungsi yang menerima dict hyperparameter
                        dan mengembalikan scalar fitness (F1 / accuracy)
            verbose:    print progress per iterasi
            callback:   opsional, dipanggil tiap iterasi dengan (iter, best_params, best_fitness)

        Returns:
            dict hyperparameter terbaik
        """
        X = self._init_population()

        # Evaluasi awal
        fitnesses = np.full(self.n_agents, float("-inf") if not self.minimize else float("inf"))
        t0_total  = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"🐋 WOA Hyperparameter Search")
            print(f"   Agents: {self.n_agents} | Max Iter: {self.max_iter}")
            print(f"   Search space ({self.dim} params):")
            for name, (lo, hi) in self.bounds.items():
                print(f"     {name}: [{lo}, {hi}]")
            print(f"{'='*60}\n")

        # Initial fitness evaluation
        for i in range(self.n_agents):
            params = self._decode(X[i])
            t0 = time.time()
            fitnesses[i] = fitness_fn(params)
            elapsed = time.time() - t0
            if verbose:
                print(f"  Init agent {i+1:2d}/{self.n_agents}: fitness={fitnesses[i]:.4f}  ({elapsed:.1f}s)  params={self._fmt(params)}")

            if self._is_better(fitnesses[i], self.best_fitness):
                self.best_fitness = fitnesses[i]
                self.best_params  = params.copy()

        # Best agent position
        best_X = X[np.argmax(fitnesses) if not self.minimize else np.argmin(fitnesses)].copy()

        # Main WOA loop
        for t in range(1, self.max_iter + 1):
            a  = 2.0 - 2.0 * (t / self.max_iter)   # linearly decreases from 2 to 0
            a2 = -1.0 - t / self.max_iter            # for spiral

            if verbose:
                print(f"\n--- Iter {t}/{self.max_iter}  (best so far: {self.best_fitness:.4f}) ---")

            for i in range(self.n_agents):
                r1 = self.rng.random()
                r2 = self.rng.random()
                A  = 2 * a * r1 - a
                C  = 2 * r2
                b  = 1.0   # spiral constant
                l  = self.rng.uniform(-1, 1)
                p  = self.rng.random()

                if p < 0.5:
                    if abs(A) < 1:
                        # Encircling prey — shrinking mechanism
                        D = abs(C * best_X - X[i])
                        X[i] = best_X - A * D
                    else:
                        # Random whale search
                        rand_idx = self.rng.integers(0, self.n_agents)
                        X_rand   = X[rand_idx]
                        D        = abs(C * X_rand - X[i])
                        X[i]     = X_rand - A * D
                else:
                    # Spiral update (bubble-net)
                    D_prime = abs(best_X - X[i])
                    X[i]    = D_prime * math.exp(b * l) * math.cos(2 * math.pi * l) + best_X

                X[i] = self._clip(X[i])

                # Evaluate
                params = self._decode(X[i])
                t0     = time.time()
                fit    = fitness_fn(params)
                elapsed = time.time() - t0
                fitnesses[i] = fit

                if verbose:
                    marker = "⭐" if self._is_better(fit, self.best_fitness) else "  "
                    print(f"  {marker} Agent {i+1:2d}: fitness={fit:.4f}  ({elapsed:.1f}s)  params={self._fmt(params)}")

                if self._is_better(fit, self.best_fitness):
                    self.best_fitness = fit
                    self.best_params  = params.copy()
                    best_X            = X[i].copy()

            self.history.append({
                "iter": t,
                "best_fitness": self.best_fitness,
                "best_params": self.best_params.copy() if self.best_params else None,
            })

            if callback:
                callback(t, self.best_params, self.best_fitness)

        total_time = time.time() - t0_total
        if verbose:
            print(f"\n{'='*60}")
            print(f"✅ WOA selesai dalam {total_time/60:.1f} menit")
            print(f"   Best fitness : {self.best_fitness:.4f}")
            print(f"   Best params  : {self._fmt(self.best_params)}")
            print(f"{'='*60}\n")

        return self.best_params

    @staticmethod
    def _fmt(params: Optional[Dict]) -> str:
        if params is None:
            return "N/A"
        parts = []
        for k, v in params.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.5f}")
            else:
                parts.append(f"{k}={v}")
        return "{" + ", ".join(parts) + "}"
