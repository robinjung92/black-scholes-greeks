import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import norm
import tkinter as tk
from tkinter import ttk, messagebox

class BlackScholes:
    """
    Classe pour calculer le prix des options et les Greeks selon le modèle Black-Scholes
    """
    
    def __init__(self, S, K, T, r, sigma):
        """
        Paramètres:
        S: Prix actuel de l'actif sous-jacent
        K: Prix d'exercice (strike)
        T: Temps jusqu'à l'expiration (en années)
        r: Taux sans risque
        sigma: Volatilité implicite
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
    
    def _d1(self):
        """Calcule d1 pour la formule Black-Scholes"""
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def _d2(self):
        """Calcule d2 pour la formule Black-Scholes"""
        return self._d1() - self.sigma * np.sqrt(self.T)
    
    def call_price(self):
        """Prix d'un call européen"""
        d1 = self._d1()
        d2 = self._d2()
        return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
    
    def put_price(self):
        """Prix d'un put européen"""
        d1 = self._d1()
        d2 = self._d2()
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
    
    def delta_call(self):
        """Delta d'un call"""
        return norm.cdf(self._d1())
    
    def delta_put(self):
        """Delta d'un put"""
        return norm.cdf(self._d1()) - 1
    
    def gamma(self):
        """Gamma (identique pour call et put)"""
        d1 = self._d1()
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self):
        """Vega (identique pour call et put)"""
        d1 = self._d1()
        return self.S * norm.pdf(d1) * np.sqrt(self.T)
    
    def theta_call(self):
        """Theta d'un call"""
        d1 = self._d1()
        d2 = self._d2()
        return (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) 
                - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
    
    def theta_put(self):
        """Theta d'un put"""
        d1 = self._d1()
        d2 = self._d2()
        return (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) 
                + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))
    
    def rho_call(self):
        """Rho d'un call"""
        d2 = self._d2()
        return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
    
    def rho_put(self):
        """Rho d'un put"""
        d2 = self._d2()
        return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)

class OptionPricingGUI:
    """Interface graphique pour le pricing d'options"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Black-Scholes Option Pricing Calculator")
        self.root.geometry("800x600")
        
        self.setup_gui()
        
    def setup_gui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Paramètres d'entrée
        input_frame = ttk.LabelFrame(main_frame, text="Paramètres d'entrée", padding="10")
        input_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Variables pour les entrées
        self.S_var = tk.DoubleVar(value=100)
        self.K_var = tk.DoubleVar(value=100)
        self.T_var = tk.DoubleVar(value=1.0)
        self.r_var = tk.DoubleVar(value=0.05)
        self.sigma_var = tk.DoubleVar(value=0.2)
        
        # Création des champs d'entrée
        ttk.Label(input_frame, text="Prix spot (S):").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(input_frame, textvariable=self.S_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(input_frame, text="Strike (K):").grid(row=0, column=2, sticky=tk.W, padx=5)
        ttk.Entry(input_frame, textvariable=self.K_var, width=10).grid(row=0, column=3, padx=5)
        
        ttk.Label(input_frame, text="Maturité (T):").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Entry(input_frame, textvariable=self.T_var, width=10).grid(row=1, column=1, padx=5)
        
        ttk.Label(input_frame, text="Taux sans risque (r):").grid(row=1, column=2, sticky=tk.W, padx=5)
        ttk.Entry(input_frame, textvariable=self.r_var, width=10).grid(row=1, column=3, padx=5)
        
        ttk.Label(input_frame, text="Volatilité (σ):").grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Entry(input_frame, textvariable=self.sigma_var, width=10).grid(row=2, column=1, padx=5)
        
        # Boutons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Calculer", command=self.calculate).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Graphique Payoff", command=self.show_payoff_graph).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Graphique Greeks", command=self.show_greeks_graph).pack(side=tk.LEFT, padx=5)
        
        # Résultats
        results_frame = ttk.LabelFrame(main_frame, text="Résultats", padding="10")
        results_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.results_text = tk.Text(results_frame, height=20, width=80)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
    def calculate(self):
        try:
            # Récupération des paramètres
            S = self.S_var.get()
            K = self.K_var.get()
            T = self.T_var.get()
            r = self.r_var.get()
            sigma = self.sigma_var.get()
            
            # Validation des paramètres
            if any(val <= 0 for val in [S, K, T, sigma]) or T <= 0:
                raise ValueError("Tous les paramètres doivent être positifs")
            
            # Création de l'objet Black-Scholes
            bs = BlackScholes(S, K, T, r, sigma)
            
            # Calculs
            call_price = bs.call_price()
            put_price = bs.put_price()
            
            # Greeks
            delta_call = bs.delta_call()
            delta_put = bs.delta_put()
            gamma = bs.gamma()
            vega = bs.vega()
            theta_call = bs.theta_call()
            theta_put = bs.theta_put()
            rho_call = bs.rho_call()
            rho_put = bs.rho_put()
            
            # Affichage des résultats
            results = f"""
=== PARAMÈTRES ===
Prix spot (S): {S:.2f}
Strike (K): {K:.2f}
Maturité (T): {T:.4f} années
Taux sans risque (r): {r:.4f} ({r*100:.2f}%)
Volatilité (σ): {sigma:.4f} ({sigma*100:.2f}%)

=== PRIX DES OPTIONS ===
Prix Call: {call_price:.4f}
Prix Put: {put_price:.4f}

=== GREEKS ===
Delta Call: {delta_call:.4f}
Delta Put: {delta_put:.4f}
Gamma: {gamma:.4f}
Vega: {vega:.4f}
Theta Call: {theta_call:.4f}
Theta Put: {theta_put:.4f}
Rho Call: {rho_call:.4f}
Rho Put: {rho_put:.4f}

=== ANALYSE ===
Moneyness: {"ITM" if S > K else "OTM" if S < K else "ATM"}
Parité Put-Call vérifiée: {abs(call_price - put_price - S + K * np.exp(-r * T)) < 1e-10}
            """
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, results)
            
        except ValueError as e:
            messagebox.showerror("Erreur", f"Erreur dans les paramètres: {e}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur de calcul: {e}")
    
    def show_payoff_graph(self):
        """Affiche le graphique de payoff interactif"""
        try:
            S = self.S_var.get()
            K = self.K_var.get()
            T = self.T_var.get()
            r = self.r_var.get()
            sigma = self.sigma_var.get()
            
            # Création de la figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            plt.subplots_adjust(bottom=0.25, hspace=0.3)
            
            # Range de prix
            S_range = np.linspace(0.5 * K, 1.5 * K, 100)
            
            def update_plot(val=None):
                ax1.clear()
                ax2.clear()
                
                # Payoffs à l'expiration
                call_payoff = np.maximum(S_range - K, 0)
                put_payoff = np.maximum(K - S_range, 0)
                
                # Prix actuels des options
                call_prices = []
                put_prices = []
                
                for s in S_range:
                    bs = BlackScholes(s, K, T, r, sigma)
                    call_prices.append(bs.call_price())
                    put_prices.append(bs.put_price())
                
                # Graphique 1: Payoffs à l'expiration vs Prix actuels
                ax1.plot(S_range, call_payoff, 'r--', label='Call Payoff (expiration)', linewidth=2)
                ax1.plot(S_range, put_payoff, 'b--', label='Put Payoff (expiration)', linewidth=2)
                ax1.plot(S_range, call_prices, 'r-', label='Prix Call actuel', linewidth=2)
                ax1.plot(S_range, put_prices, 'b-', label='Prix Put actuel', linewidth=2)
                ax1.axvline(S, color='green', linestyle=':', label=f'Prix spot actuel ({S})')
                ax1.axvline(K, color='black', linestyle=':', label=f'Strike ({K})')
                ax1.set_xlabel('Prix de l\'actif sous-jacent')
                ax1.set_ylabel('Valeur de l\'option')
                ax1.set_title('Payoff et Prix des Options')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Graphique 2: P&L à l'expiration
                bs_current = BlackScholes(S, K, T, r, sigma)
                call_current_price = bs_current.call_price()
                put_current_price = bs_current.put_price()
                
                call_pnl = call_payoff - call_current_price
                put_pnl = put_payoff - put_current_price
                
                ax2.plot(S_range, call_pnl, 'r-', label='P&L Call', linewidth=2)
                ax2.plot(S_range, put_pnl, 'b-', label='P&L Put', linewidth=2)
                ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
                ax2.axvline(S, color='green', linestyle=':', label=f'Prix spot actuel ({S})')
                ax2.axvline(K, color='black', linestyle=':', label=f'Strike ({K})')
                ax2.set_xlabel('Prix de l\'actif sous-jacent')
                ax2.set_ylabel('P&L à l\'expiration')
                ax2.set_title('Profit & Loss à l\'expiration')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.draw()
            
            # Sliders pour interactivité
            ax_S = plt.axes([0.2, 0.1, 0.3, 0.03])
            ax_sigma = plt.axes([0.2, 0.05, 0.3, 0.03])
            ax_T = plt.axes([0.6, 0.1, 0.3, 0.03])
            ax_r = plt.axes([0.6, 0.05, 0.3, 0.03])
            
            slider_S = Slider(ax_S, 'Prix Spot', 0.5*K, 1.5*K, valinit=S)
            slider_sigma = Slider(ax_sigma, 'Volatilité', 0.05, 1.0, valinit=sigma)
            slider_T = Slider(ax_T, 'Maturité', 0.01, 2.0, valinit=T)
            slider_r = Slider(ax_r, 'Taux', 0.0, 0.2, valinit=r)
            
            def update_params(val):
                nonlocal S, sigma, T, r
                S = slider_S.val
                sigma = slider_sigma.val
                T = slider_T.val
                r = slider_r.val
                update_plot()
            
            slider_S.on_changed(update_params)
            slider_sigma.on_changed(update_params)
            slider_T.on_changed(update_params)
            slider_r.on_changed(update_params)
            
            update_plot()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la création du graphique: {e}")
    
    def show_greeks_graph(self):
        """Affiche les graphiques des Greeks"""
        try:
            S = self.S_var.get()
            K = self.K_var.get()
            T = self.T_var.get()
            r = self.r_var.get()
            sigma = self.sigma_var.get()
            
            # Range de prix
            S_range = np.linspace(0.5 * K, 1.5 * K, 100)
            
            # Calcul des Greeks
            deltas_call = []
            deltas_put = []
            gammas = []
            vegas = []
            thetas_call = []
            thetas_put = []
            
            for s in S_range:
                bs = BlackScholes(s, K, T, r, sigma)
                deltas_call.append(bs.delta_call())
                deltas_put.append(bs.delta_put())
                gammas.append(bs.gamma())
                vegas.append(bs.vega())
                thetas_call.append(bs.theta_call())
                thetas_put.append(bs.theta_put())
            
            # Création des graphiques
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Delta
            ax1.plot(S_range, deltas_call, 'r-', label='Delta Call', linewidth=2)
            ax1.plot(S_range, deltas_put, 'b-', label='Delta Put', linewidth=2)
            ax1.axvline(S, color='green', linestyle=':', label=f'Prix spot actuel')
            ax1.axvline(K, color='black', linestyle=':', label=f'Strike')
            ax1.set_xlabel('Prix spot')
            ax1.set_ylabel('Delta')
            ax1.set_title('Delta vs Prix spot')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gamma
            ax2.plot(S_range, gammas, 'g-', linewidth=2)
            ax2.axvline(S, color='green', linestyle=':', label=f'Prix spot actuel')
            ax2.axvline(K, color='black', linestyle=':', label=f'Strike')
            ax2.set_xlabel('Prix spot')
            ax2.set_ylabel('Gamma')
            ax2.set_title('Gamma vs Prix spot')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Vega
            ax3.plot(S_range, vegas, 'm-', linewidth=2)
            ax3.axvline(S, color='green', linestyle=':', label=f'Prix spot actuel')
            ax3.axvline(K, color='black', linestyle=':', label=f'Strike')
            ax3.set_xlabel('Prix spot')
            ax3.set_ylabel('Vega')
            ax3.set_title('Vega vs Prix spot')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Theta
            ax4.plot(S_range, thetas_call, 'r-', label='Theta Call', linewidth=2)
            ax4.plot(S_range, thetas_put, 'b-', label='Theta Put', linewidth=2)
            ax4.axvline(S, color='green', linestyle=':', label=f'Prix spot actuel')
            ax4.axvline(K, color='black', linestyle=':', label=f'Strike')
            ax4.set_xlabel('Prix spot')
            ax4.set_ylabel('Theta')
            ax4.set_title('Theta vs Prix spot')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la création du graphique des Greeks: {e}")
    
    def run(self):
        self.root.mainloop()

def main():
    """Fonction principale"""
    app = OptionPricingGUI()
    app.run()

if __name__ == "__main__":
    main()