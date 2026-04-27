import numpy as np
import logging

logger = logging.getLogger(__name__)

class AvellanedaStoikov:
    """
    Avellaneda-Stoikov Market Making Model.
    Calculates the optimal Reservation Price and Spread dynamically to 
    manage inventory risk and maximize spread capture.
    """
    def __init__(self, gamma=0.1, k=1.5, terminal_time=1.0):
        # Risk aversion parameter. Higher = less tolerance for holding inventory
        self.gamma = gamma 
        # Order book depth / liquidity parameter
        self.k = k 
        # End of trading session (T). For crypto (24/7), usually set to 1.0 and t=0.
        self.terminal_time = terminal_time 

    def calculate_quotes(self, current_price: float, inventory_q: float, volatility: float, current_t: float = 0.0) -> dict:
        """
        Returns optimal Bid and Ask prices.
        
        :param current_price: Mid price of the asset (s)
        :param inventory_q: Current coin holdings (q). Positive if long, negative if short.
        :param volatility: Standard deviation of the asset (sigma)
        :param current_t: Current time (t), defaults to 0 in continuous markets
        """
        time_left = self.terminal_time - current_t
        
        # Calculate Reservation Price (r)
        # If you hold too much inventory (q > 0), the reservation price shifts DOWN to encourage selling.
        variance = volatility ** 2
        reservation_price = current_price - (inventory_q * self.gamma * variance * time_left)
        
        # Calculate Optimal Spread (delta)
        spread = (self.gamma * variance * time_left) + (2 / self.gamma) * np.log(1 + (self.gamma / self.k))
        
        optimal_bid = reservation_price - (spread / 2)
        optimal_ask = reservation_price + (spread / 2)
        
        return {
            "mid_price": current_price,
            "reservation_price": reservation_price,
            "optimal_spread": spread,
            "optimal_bid": optimal_bid,
            "optimal_ask": optimal_ask
        }