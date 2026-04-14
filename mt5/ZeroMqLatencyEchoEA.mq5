// ZeroMqLatencyEchoEA.mq5 — MT5 REQ connects to Python REP server

#import "libzmq.dll"
   long zmq_ctx_new();
   int  zmq_ctx_term(long context);
   long zmq_socket(long context, int type);
   int  zmq_close(long socket);
   int  zmq_connect(long socket, string addr);
   int  zmq_send(long socket, uchar &data[], int len, int flags);
   int  zmq_recv(long socket, uchar &data[], int len, int flags);
#import

#define ZMQ_REQ 3

input int EchoPort = 5555;

long ctx_ptr = 0;
long req_ptr = 0;

int OnInit()
{
    ctx_ptr = zmq_ctx_new();
    if(ctx_ptr <= 0) {
        Print("[CRITICAL] Failed to create ZMQ context.");
        return(INIT_FAILED);
    }

    req_ptr = zmq_socket(ctx_ptr, ZMQ_REQ);
    if(req_ptr <= 0) {
        Print("[CRITICAL] Failed to create REQ socket.");
        return(INIT_FAILED);
    }

    string endpoint = "tcp://127.0.0.1:" + IntegerToString(EchoPort);
    int rc = zmq_connect(req_ptr, endpoint);

    if(rc != 0)
        Print("[ERROR] Connect failed to ", endpoint);
    else
        Print("[SYSTEM] Connected to Python echo server on ", endpoint);

    EventSetMillisecondTimer(1);
    return(INIT_SUCCEEDED);
}

void OnTimer()
{
    uchar ping[4];
    ping[0]='p'; ping[1]='i'; ping[2]='n'; ping[3]='g';
    zmq_send(req_ptr, ping, 4, 0);

    uchar buf[64];
    zmq_recv(req_ptr, buf, 64, 0);
}

void OnDeinit(const int reason)
{
    EventKillTimer();
    Print("[SYSTEM] Cleaning up...");
    if(req_ptr != 0) zmq_close(req_ptr);
    if(ctx_ptr != 0) zmq_ctx_term(ctx_ptr);
}