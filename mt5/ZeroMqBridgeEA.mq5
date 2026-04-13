#property strict
#property version   "1.00"
#property description "ZMQ subscriber + heartbeat EA with fail-safe flatten logic"

#include <Trade/Trade.mqh>

input string InpClientId = "mt5-xauusd-01";
input string InpSignalSubEndpoint = "tcp://127.0.0.1:5556";
input string InpSignalTopic = "trade.signal";
input string InpHeartbeatReqEndpoint = "tcp://127.0.0.1:5557";
input int InpHeartbeatIntervalMs = 1000;
input int InpHeartbeatTimeoutMs = 5000;
input int InpSocketTimeoutMs = 250;
input double InpDefaultLotSize = 0.01;
input int InpMaxSignalsPerTick = 10;
input long InpMagicNumber = 710001;
input int InpMaxSlippagePoints = 25;

// This EA expects a local wrapper DLL that exposes simplified ZMQ operations.
#import "ZmqMt5Bridge.dll"
int ZmqCreateSubscriber(string endpoint, string topic);
int ZmqCreateReq(string endpoint);
bool ZmqSubscriberRecvText(int socketHandle, string &message, int timeoutMs);
bool ZmqReqRoundtrip(int socketHandle, string request, string &response, int timeoutMs);
void ZmqClose(int socketHandle);
#import

CTrade trade;
int g_subSocket = -1;
int g_reqSocket = -1;
bool g_failsafeActive = false;
datetime g_lastHeartbeatOk = 0;
datetime g_lastHeartbeatSent = 0;

int OnInit()
{
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpMaxSlippagePoints);

   g_subSocket = ZmqCreateSubscriber(InpSignalSubEndpoint, InpSignalTopic);
   if(g_subSocket < 0)
   {
      Print("[ZMQ_EA] Failed to create subscriber socket");
      return(INIT_FAILED);
   }

   g_reqSocket = ZmqCreateReq(InpHeartbeatReqEndpoint);
   if(g_reqSocket < 0)
   {
      Print("[ZMQ_EA] Failed to create heartbeat REQ socket");
      ZmqClose(g_subSocket);
      g_subSocket = -1;
      return(INIT_FAILED);
   }

   g_lastHeartbeatOk = TimeCurrent();
   g_lastHeartbeatSent = 0;
   EventSetMillisecondTimer(200);
   Print("[ZMQ_EA] Initialized successfully");
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   if(g_subSocket >= 0)
   {
      ZmqClose(g_subSocket);
      g_subSocket = -1;
   }
   if(g_reqSocket >= 0)
   {
      ZmqClose(g_reqSocket);
      g_reqSocket = -1;
   }
   Print("[ZMQ_EA] Deinitialized. reason=", reason);
}

void OnTick()
{
   // Logic is timer-driven for deterministic heartbeat polling and signal draining.
}

void OnTimer()
{
   PumpHeartbeat();
   PumpSignals();
   EvaluateFailsafe();
}

void PumpHeartbeat()
{
   datetime now = TimeCurrent();
   long elapsedMs = (long)(now - g_lastHeartbeatSent) * 1000;
   if(elapsedMs < InpHeartbeatIntervalMs)
      return;

   g_lastHeartbeatSent = now;
   string request = BuildHeartbeatRequest();
   string response = "";

   bool ok = ZmqReqRoundtrip(g_reqSocket, request, response, InpSocketTimeoutMs);
   if(ok && StringFind(response, "\"ok\":true") >= 0)
   {
      g_lastHeartbeatOk = now;
      if(g_failsafeActive)
      {
         g_failsafeActive = false;
         Print("[ZMQ_EA] Heartbeat restored. Failsafe cleared.");
      }
   }
}

void PumpSignals()
{
   if(g_failsafeActive)
      return;

   int processed = 0;
   while(processed < InpMaxSignalsPerTick)
   {
      string message = "";
      bool got = ZmqSubscriberRecvText(g_subSocket, message, InpSocketTimeoutMs);
      if(!got || StringLen(message) == 0)
         break;

      processed++;
      HandleSignalEnvelope(message);
   }
}

void EvaluateFailsafe()
{
   datetime now = TimeCurrent();
   long silentMs = (long)(now - g_lastHeartbeatOk) * 1000;
   if(silentMs <= InpHeartbeatTimeoutMs)
      return;

   if(!g_failsafeActive)
   {
      g_failsafeActive = true;
      Print("[ZMQ_EA][FAILSAFE] Heartbeat timeout detected. Flattening + canceling pending orders.");
      FlattenAllPositions();
      CancelAllPendingOrders();
   }
}

void HandleSignalEnvelope(const string message)
{
   string symbol = ExtractJsonString(message, "symbol");
   string action = StringUpper(ExtractJsonString(message, "action"));

   if(symbol == "" || action == "")
   {
      Print("[ZMQ_EA] Ignored malformed signal envelope: ", message);
      return;
   }

   if(symbol != _Symbol)
      return;

   double lot = ExtractJsonNumber(message, "lot_size", InpDefaultLotSize);
   if(lot <= 0.0)
      lot = InpDefaultLotSize;

   ExecuteAction(action, lot);
}

void ExecuteAction(const string action, const double volume)
{
   bool success = false;

   if(action == "BUY")
      success = trade.Buy(volume, _Symbol, 0.0, 0.0, 0.0, "ZMQ_SIGNAL");
   else if(action == "SELL")
      success = trade.Sell(volume, _Symbol, 0.0, 0.0, 0.0, "ZMQ_SIGNAL");
   else if(action == "FLAT" || action == "CLOSE_ALL")
   {
      FlattenAllPositions();
      CancelAllPendingOrders();
      success = true;
   }

   if(!success)
   {
      Print("[ZMQ_EA] Trade action failed. action=", action, " retcode=", trade.ResultRetcode(), " desc=", trade.ResultRetcodeDescription());
      return;
   }

   Print("[ZMQ_EA] Executed action=", action, " volume=", DoubleToString(volume, 2));
}

void FlattenAllPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(!PositionSelectByTicket(ticket))
         continue;

      string symbol = PositionGetString(POSITION_SYMBOL);
      long magic = PositionGetInteger(POSITION_MAGIC);
      if(symbol != _Symbol)
         continue;
      if(InpMagicNumber != 0 && magic != InpMagicNumber)
         continue;

      if(!trade.PositionClose(ticket))
      {
         Print("[ZMQ_EA] Failed to close position ticket=", ticket, " retcode=", trade.ResultRetcode(), " desc=", trade.ResultRetcodeDescription());
      }
   }
}

void CancelAllPendingOrders()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0)
         continue;
      if(!OrderSelect(ticket))
         continue;

      string symbol = OrderGetString(ORDER_SYMBOL);
      long magic = OrderGetInteger(ORDER_MAGIC);
      ENUM_ORDER_TYPE type = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
      if(symbol != _Symbol)
         continue;
      if(InpMagicNumber != 0 && magic != InpMagicNumber)
         continue;
      if(type == ORDER_TYPE_BUY || type == ORDER_TYPE_SELL)
         continue;

      if(!trade.OrderDelete(ticket))
      {
         Print("[ZMQ_EA] Failed to delete pending ticket=", ticket, " retcode=", trade.ResultRetcode(), " desc=", trade.ResultRetcodeDescription());
      }
   }
}

string BuildHeartbeatRequest()
{
   string payload = "{";
   payload += "\"client_id\":\"" + JsonEscape(InpClientId) + "\",";
   payload += "\"symbol\":\"" + JsonEscape(_Symbol) + "\",";
   payload += "\"account\":" + IntegerToString((int)AccountInfoInteger(ACCOUNT_LOGIN)) + ",";
   payload += "\"ts\":" + IntegerToString((int)TimeCurrent());
   payload += "}";
   return payload;
}

string ExtractJsonString(const string src, const string key)
{
   string marker = "\"" + key + "\"";
   int keyPos = StringFind(src, marker);
   if(keyPos < 0)
      return "";

   int colonPos = StringFind(src, ":", keyPos + StringLen(marker));
   if(colonPos < 0)
      return "";

   int firstQuote = StringFind(src, "\"", colonPos + 1);
   if(firstQuote < 0)
      return "";

   int secondQuote = StringFind(src, "\"", firstQuote + 1);
   if(secondQuote < 0)
      return "";

   return StringSubstr(src, firstQuote + 1, secondQuote - firstQuote - 1);
}

double ExtractJsonNumber(const string src, const string key, const double fallback)
{
   string marker = "\"" + key + "\"";
   int keyPos = StringFind(src, marker);
   if(keyPos < 0)
      return fallback;

   int colonPos = StringFind(src, ":", keyPos + StringLen(marker));
   if(colonPos < 0)
      return fallback;

   int start = colonPos + 1;
   while(start < StringLen(src))
   {
      ushort ch = StringGetCharacter(src, start);
      if(ch != ' ' && ch != '\t')
         break;
      start++;
   }

   int end = start;
   while(end < StringLen(src))
   {
      ushort ch = StringGetCharacter(src, end);
      bool isNumChar = (ch >= '0' && ch <= '9') || ch == '.' || ch == '-' || ch == '+';
      if(!isNumChar)
         break;
      end++;
   }

   if(end <= start)
      return fallback;

   string raw = StringSubstr(src, start, end - start);
   return StringToDouble(raw);
}

string JsonEscape(const string value)
{
   string escaped = value;
   StringReplace(escaped, "\\", "\\\\");
   StringReplace(escaped, "\"", "\\\"");
   return escaped;
}
