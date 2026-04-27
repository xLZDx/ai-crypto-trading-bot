import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Добавляем корень проекта в sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.engine.trade_tracker import TradeTracker
from src.engine.order_manager import OrderManager
from src.engine.agentic_llm import AgenticLLM


class TestTradeTracker(unittest.TestCase):
    def setUp(self):
        # Используем тестовый JSON файл
        self.test_file = 'test_trades.json'
        self.tracker = TradeTracker(filepath=self.test_file)
        self.tracker.trades = [] # Очищаем состояние

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_open_and_close_trade(self):
        trade = self.tracker.open_trade(
            symbol="BTC/USDT", amount_usdt=1000, current_price=50000, side="LONG"
        )
        self.assertEqual(trade["status"], "OPEN")
        self.assertEqual(trade["amount_coin"], 1000 / 50000)
        
        # Закрываем с прибылью (цена 55000)
        closed_trade = self.tracker.close_trade_by_id(trade["id"], real_sell_price=55000)
        self.assertEqual(closed_trade["status"], "CLOSED")
        self.assertGreater(closed_trade["pnl_usdt"], 0)

    def test_trailing_stop_long(self):
        # Открываем лонг на 50000, трейлинг 2%
        self.tracker.open_trade("BTC/USDT", 100, 50000, trailing_stop_percent=2.0, side="LONG")
        
        # Цена растет до 60000 -> Новый максимум, стоп передвигается на 58800 (60k - 2%)
        closed = self.tracker.update_trailing_stops("BTC/USDT", 60000)
        self.assertEqual(len(closed), 0) # Стоп не пробит
        
        # Цена падает до 59000 (все еще выше 58800)
        closed = self.tracker.update_trailing_stops("BTC/USDT", 59000)
        self.assertEqual(len(closed), 0)
        
        # Цена падает до 58000 (пробивает 58800)
        closed = self.tracker.update_trailing_stops("BTC/USDT", 58000)
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["status"], "OPEN") # Внутренне он еще OPEN, ждет исполнения ордера


class TestOrderManager(unittest.TestCase):
    @patch('src.engine.order_manager.ccxt.binance')
    def test_to_futures_symbol(self, mock_binance):
        # Статический метод не требует реального инстанса, но мокаем инициализацию ccxt для __init__
        manager = OrderManager()
        self.assertEqual(manager.to_futures_symbol("BTC/USDT"), "BTC/USDT:USDT")
        self.assertEqual(manager.to_futures_symbol("ETH/BUSD"), "ETH/BUSD:BUSD")
        
        with self.assertRaises(ValueError):
            manager.to_futures_symbol("BTCUSDT")


class TestAgenticLLM(unittest.TestCase):
    @patch('src.engine.agentic_llm.read_json')
    def test_evaluate_trade_approved(self, mock_read_json):
        mock_read_json.return_value = {}
        # Настраиваем мок-ответ от Gemini
        mock_response = MagicMock()
        mock_response.text = '{"decision": "APPROVED", "reason": "No severe risks detected."}'
        
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        agent = AgenticLLM()
        agent.is_active = True
        agent._client = mock_client

        # Mock _gntypes since google.genai might not be installed during tests
        with patch.dict('sys.modules', {'google.genai': MagicMock(), 'google.genai.types': MagicMock()}):
            decision, reason = agent.evaluate_trade("BTC/USDT", "BUY", "RSI is oversold", ["Market is calm"])
        
        self.assertEqual(decision, "APPROVED")
        self.assertEqual(reason, "No severe risks detected.")
        mock_client.models.generate_content.assert_called_once()


if __name__ == '__main__':
    unittest.main()