
import re
import numpy as np
import unittest.mock as mock

from tensortrade.instruments import *
from tensortrade.orders import Order, OrderStatus, OrderSpec, TradeSide, TradeType
from tensortrade.orders.criteria import Stop
from tensortrade.wallets import Wallet, Portfolio


@mock.patch('tensortrade.exchanges.ExchangePair')
@mock.patch('tensortrade.wallets.Portfolio')
def test_init(mock_portfolio_class, mock_exchange_pair):

    portfolio = mock_portfolio_class.return_value

    exchange_pair = mock_exchange_pair.return_value
    exchange_pair.pair = USD/BTC
    exchange_pair.exchange = "coinbase"

    order = Order(step=0,
                  exchange_pair=exchange_pair,
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  quantity=5000 * USD,
                  price=7000,
                  portfolio=portfolio)

    assert order
    assert order.id
    assert order.path_id
    assert order.step == 0
    assert order.quantity.instrument == USD
    assert order.filled_size == 0
    assert order.remaining_size == order.quantity
    assert isinstance(order.pair, TradingPair)
    assert order.pair.base == USD
    assert order.pair.quote == BTC


@mock.patch('tensortrade.wallets.Portfolio')
def test_properties(mock_portfolio_class):

    portfolio = mock_portfolio_class.return_value

    order = Order(step=0,
                  exchange_name="coinbase",
                  side=TradeSide.BUY,
                  trade_type=TradeType.LIMIT,
                  pair=USD/BTC,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    assert order
    assert order.step == 0
    assert order.base_instrument == USD
    assert order.quote_instrument == BTC
    assert order.size == 5000.00 * USD
    assert order.price == 7000.00
    assert order.trades == []
    assert order.is_buy
    assert not order.is_sell
    assert not order.is_market_order
    assert order.is_limit_order


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.wallets.Portfolio')
def test_is_executable_on(mock_portfolio_class, mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.name = "coinbase"
    portfolio = mock_portfolio_class.return_value

    # Market order
    order = Order(step=0,
                  exchange_name="coinbase",
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD/BTC,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    exchange.quote_price = mock.Mock(return_value=6800.00)
    assert order.is_executable_on(exchange)

    exchange.quote_price = mock.Mock(return_value=7200.00)
    assert order.is_executable_on(exchange)

    # Limit order
    order = Order(step=0,
                  exchange_name="coinbase",
                  side=TradeSide.BUY,
                  trade_type=TradeType.LIMIT,
                  pair=USD/BTC,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    exchange.quote_price = mock.Mock(return_value=6800.00)
    assert order.is_executable_on(exchange)

    exchange.quote_price = mock.Mock(return_value=7200.00)
    assert order.is_executable_on(exchange)

    # Stop Order
    order = Order(step=0,
                  exchange_name="coinbase",
                  side=TradeSide.SELL,
                  trade_type=TradeType.LIMIT,
                  pair=USD/BTC,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=7000.00,
                  criteria=Stop("down", 0.03))

    exchange.quote_price = mock.Mock(return_value=(1 - 0.031)*order.price)
    assert order.is_executable_on(exchange)

    exchange.quote_price = mock.Mock(return_value=(1 - 0.02) * order.price)
    assert not order.is_executable_on(exchange)


@mock.patch("tensortrade.wallets.Portfolio")
def test_is_complete(mock_portfolio_class):

    portfolio = mock_portfolio_class.return_value

    # Market order
    order = Order(step=0,
                  exchange_name="coinbase",
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    assert not order.is_complete()

    order.remaining_size = 0
    assert order.is_complete()


@mock.patch('tensortrade.orders.OrderSpec')
@mock.patch('tensortrade.wallets.Portfolio')
def test_add_order_spec(mock_portfolio_class, mock_order_spec_class):

    portfolio = mock_portfolio_class.return_value

    # Market order
    order = Order(step=0,
                  exchange_name="coinbase",
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    order_spec = mock_order_spec_class.return_value

    assert len(order._specs) == 0
    order.add_order_spec(order_spec)
    assert len(order._specs) == 1

    assert order_spec in order._specs


@mock.patch('tensortrade.orders.OrderListener')
@mock.patch('tensortrade.wallets.Portfolio')
def test_attach(mock_portfolio_class, mock_order_listener_class):

    portfolio = mock_portfolio_class.return_value
    order = Order(step=0,
                  exchange_name="coinbase",
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    listener = mock_order_listener_class.return_value

    assert len(order._listeners) == 0
    order.attach(listener)
    assert len(order._listeners) == 1
    assert listener in order._listeners


@mock.patch('tensortrade.orders.OrderListener')
@mock.patch('tensortrade.wallets.Portfolio')
def test_detach(mock_portfolio_class, mock_order_listener_class):
    portfolio = mock_portfolio_class.return_value
    order = Order(step=0,
                  exchange_name="coinbase",
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    listener = mock_order_listener_class.return_value
    order.attach(listener)
    assert len(order._listeners) == 1
    assert listener in order._listeners

    order.detach(listener)
    assert len(order._listeners) == 0
    assert listener not in order._listeners


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.orders.OrderListener')
def test_execute(mock_order_listener_class,
                 mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.id = "fake_id"

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(step=0,
                  exchange_name="coinbase",
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    listener = mock_order_listener_class.return_value
    listener.on_execute = mock.Mock(return_value=None)
    order.attach(listener)

    assert order.status == OrderStatus.PENDING
    order.execute()
    assert order.status == OrderStatus.OPEN

    wallet_usd = portfolio.get_wallet(exchange.id, USD)
    wallet_btc = portfolio.get_wallet(exchange.id, BTC)

    assert wallet_usd.balance == 4800 * USD
    assert wallet_usd.locked_balance == 5200 * USD
    assert order.path_id in wallet_usd.locked.keys()
    assert wallet_btc.balance == 0 * BTC

    listener.on_execute.assert_called_once_with(order)


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.orders.Trade')
@mock.patch('tensortrade.orders.OrderListener')
def test_fill(mock_order_listener_class,
              mock_trade_class,
              mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.id = "fake_exchange_id"

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(step=0,
                  exchange_name="coinbase",
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    listener = mock_order_listener_class.return_value
    listener.on_fill = mock.Mock(return_value=None)
    order.attach(listener)

    order.execute(exchange)

    trade = mock_trade_class.return_value
    trade.size = 3997.00
    trade.commission = 3.00 * USD

    assert order.status == OrderStatus.OPEN
    order.fill(exchange, trade)
    assert order.status == OrderStatus.PARTIALLY_FILLED

    assert order.remaining_size == 1200.00

    listener.on_fill.assert_called_once_with(order, exchange, trade)


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.orders.Trade')
@mock.patch('tensortrade.orders.OrderListener')
def test_complete_basic_order(mock_order_listener_class,
                              mock_trade_class,
                              mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.id = "fake_exchange_id"

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(step=0,
                  exchange_name="coinbase",
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    listener = mock_order_listener_class.return_value
    listener.on_complete = mock.Mock(return_value=None)
    order.attach(listener)

    order.execute(exchange)

    trade = mock_trade_class.return_value
    trade.size = 5217.00
    trade.commission = 3.00 * USD

    order.fill(exchange, trade)

    assert order.status == OrderStatus.PARTIALLY_FILLED
    next_order = order.complete(exchange)
    assert order.status == OrderStatus.FILLED

    listener.on_complete.assert_called_once_with(order, exchange)
    assert not next_order


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.orders.Trade')
def test_complete_complex_order(mock_trade_class,
                                mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.id = "fake_exchange_id"

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    side = TradeSide.BUY

    order = Order(step=0,
                  exchange_name="coinbase",
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    risk_criteria = Stop("down", 0.03) ^ Stop("up", 0.02)

    risk_management = OrderSpec(side=TradeSide.SELL if side == TradeSide.BUY else TradeSide.BUY,
                                trade_type=TradeType.MARKET,
                                pair=USD / BTC,
                                criteria=risk_criteria)

    order += risk_management

    order.execute(exchange)

    # Execute fake trade
    price = 7010.00
    scale = order.price / price
    commission = 3.00 * USD

    base_size = scale * order.size - commission.size

    trade = mock_trade_class.return_value
    trade.size = base_size
    trade.price = price
    trade.commission = commission

    base_wallet = portfolio.get_wallet(exchange.id, USD)
    quote_wallet = portfolio.get_wallet(exchange.id, BTC)

    base_size = trade.size + trade.commission.size
    quote_size = (order.price / trade.price) * (trade.size / trade.price)

    base_wallet -= Quantity(USD, size=base_size, path_id=order.path_id)
    quote_wallet += Quantity(BTC, size=quote_size, path_id=order.path_id)

    # Fill fake trade
    order.fill(exchange, trade)

    assert order.path_id in portfolio.get_wallet(exchange.id, USD).locked

    assert order.status == OrderStatus.PARTIALLY_FILLED
    next_order = order.complete(exchange)
    assert order.status == OrderStatus.FILLED

    assert next_order
    assert next_order.path_id == order.path_id
    assert next_order.size
    assert next_order.status == OrderStatus.PENDING
    assert next_order.side == TradeSide.SELL
    assert next_order.pair == USD/BTC


@mock.patch('tensortrade.exchanges.Exchange')
@mock.patch('tensortrade.orders.Trade')
@mock.patch('tensortrade.orders.OrderListener')
def test_cancel(mock_order_listener_class,
                mock_trade_class,
                mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.id = "fake_exchange_id"

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(step=0,
                  exchange_name="coinbase",
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    listener = mock_order_listener_class.return_value
    listener.on_cancel = mock.Mock(return_value=None)
    order.attach(listener)

    order.execute(exchange)

    # Execute fake trade
    price = 7010.00
    scale = order.price / price
    commission = 3.00 * USD

    trade = mock_trade_class.return_value
    trade.size = scale * order.size - commission.size
    trade.price = price
    trade.commission = commission

    base_wallet = portfolio.get_wallet(exchange.id, USD)
    quote_wallet = portfolio.get_wallet(exchange.id, BTC)

    base_size = trade.size + commission.size
    quote_size = (order.price / trade.price) * (trade.size / trade.price)

    base_wallet -= Quantity(USD, size=base_size, path_id=order.path_id)
    quote_wallet += Quantity(BTC, size=quote_size, path_id=order.path_id)

    order.fill(exchange, trade)

    assert order.status == OrderStatus.PARTIALLY_FILLED
    assert base_wallet.balance == 4800.00 * USD
    assert round(base_wallet.locked[order.path_id].size, 2) == 7.42
    assert quote_wallet.balance == 0 * BTC
    assert round(quote_wallet.locked[order.path_id].size, 8) == 0.73925519
    order.cancel()

    listener.on_cancel.assert_called_once_with(order)
    assert round(base_wallet.balance.size, 2) == 4807.42
    assert order.path_id not in base_wallet.locked
    assert round(quote_wallet.balance.size, 8) == 0.73925519
    assert order.path_id not in quote_wallet.locked


@mock.patch('tensortrade.exchanges.Exchange')
def test_release(mock_exchange_class):

    exchange = mock_exchange_class.return_value
    exchange.id = "fake_exchange_id"

    wallets = [Wallet(exchange, 10000 * USD), Wallet(exchange, 0 * BTC)]
    portfolio = Portfolio(USD, wallets)

    order = Order(step=0,
                  exchange_name="coinbase",
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    order.execute(exchange)

    wallet_usd = portfolio.get_wallet(exchange.id, USD)
    assert wallet_usd.balance == 4800 * USD
    assert wallet_usd.locked_balance == 5200 * USD
    assert order.path_id in wallet_usd.locked.keys()

    order.release()

    assert wallet_usd.balance == 10000 * USD
    assert wallet_usd.locked_balance == 0 * USD
    assert order.path_id not in wallet_usd.locked.keys()


@mock.patch('tensortrade.wallets.Portfolio')
def test_to_json(mock_portfolio_class):

    portfolio = mock_portfolio_class.return_value

    order = Order(step=0,
                  exchange_name="coinbase",
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    d = {
        "id": str(order.id),
        "step": int(order.step),
        "exchange_name": str(order.exchange_name),
        "status": str(order.status),
        "type": str(order.type),
        "side": str(order.side),
        "base_symbol": str(order.pair.base.symbol),
        "quote_symbol": str(order.pair.quote.symbol),
        "quantity": str(order.quantity),
        "size": float(order.size),
        "filled_size": float(order.filled_size),
        "price": float(order.price),
        "criteria": str(order.criteria),
        "path_id": str(order.path_id),
        "created_at": str(order.created_at)
    }

    assert order.to_json() == d


@mock.patch('tensortrade.orders.OrderSpec')
@mock.patch('tensortrade.wallets.Portfolio')
def test_iadd(mock_portfolio_class, mock_order_spec_class):

    portfolio = mock_portfolio_class.return_value

    # Market order
    order = Order(step=0,
                  exchange_name="coinbase",
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5000.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    order_spec = mock_order_spec_class.return_value

    assert len(order._specs) == 0
    order += order_spec
    assert len(order._specs) == 1

    assert order_spec in order._specs


@mock.patch('tensortrade.wallets.Portfolio')
def test_str(mock_portfolio_class):

    portfolio = mock_portfolio_class.return_value

    order = Order(step=0,
                  exchange_name="coinbase",
                  side=TradeSide.BUY,
                  trade_type=TradeType.MARKET,
                  pair=USD / BTC,
                  quantity=5200.00 * USD,
                  portfolio=portfolio,
                  price=7000.00)

    pattern = re.compile("<[A-Z][a-zA-Z]*:\\s(\\w+=.*,\\s)*(\\w+=.*)>")

    string = str(order)
    assert string

    assert string == pattern.fullmatch(string).string
