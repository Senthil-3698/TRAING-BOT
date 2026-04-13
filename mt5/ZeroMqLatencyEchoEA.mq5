// 1. DLL IMPORTS
#import "libzmq.dll"
   long zmq_ctx_new();
   int  zmq_ctx_term(long context);
   long zmq_socket(long context, int type);
   int  zmq_close(long socket);
   int  zmq_connect(long socket, string addr);
   int  zmq_send(long socket, uchar &data[], int len, int flags);
   int  zmq_recv(long socket, uchar &data[], int len, int flags);
   int  zmq_setsockopt(long socket, int option, string value, int len);
#import

// ZMQ Standards
#define ZMQ_SUB 2
#define ZMQ_REP 4
#define ZMQ_DONTWAIT 1
#define ZMQ_SUBSCRIBE 6

// Global Handles
long ctx_ptr = 0;
long sub_ptr = 0;
long rep_ptr = 0;

int OnInit() {
    Print("[SYSTEM] Finalizing Bridge Handshake...");
    
    ctx_ptr = zmq_ctx_new();
    if(ctx_ptr <= 0) {
        Print("[CRITICAL] DLL Error. Context could not be created.");
        return(INIT_FAILED);
    }
    
    sub_ptr = zmq_socket(ctx_ptr, ZMQ_SUB);
    rep_ptr = zmq_socket(ctx_ptr, ZMQ_REP);
    
    // CRITICAL: Subscribe to ALL messages (empty string)
    zmq_setsockopt(sub_ptr, ZMQ_SUBSCRIBE, "", 0);
    
    // Connect to ports
    int s1 = zmq_connect(sub_ptr, "tcp://127.0.0.1:5556");
    int s2 = zmq_connect(rep_ptr, "tcp://127.0.0.1:5557");
    
    if(s1 != 0 || s2 != 0) {
        Print("[ERROR] Port Connection Refused. Ensure Python script is running!");
        // We return success anyway so the EA stays on chart to retry
    } else {
        Print("[SYSTEM] SUCCESS: Sockets connected to 127.0.0.1");
    }
    
    EventSetMillisecondTimer(1);
    return(INIT_SUCCEEDED);
}

void OnTimer() {
    uchar buffer[1024];
    
    // AGGRESSIVE LOOP: Drain the buffer in case Python sends bursts
    while(true) {
        ArrayInitialize(buffer, 0);
        int bytes = zmq_recv(sub_ptr, buffer, 1024, ZMQ_DONTWAIT);
        
        if(bytes > 0) {
            // Signal detected! Send it back immediately.
            zmq_send(rep_ptr, buffer, bytes, ZMQ_DONTWAIT);
            Print("[DATA] Echoed " + IntegerToString(bytes) + " bytes back to Python.");
        } else {
            // No more data in the socket for this millisecond
            break;
        }
    }
}

void OnDeinit(const int reason) {
    EventKillTimer();
    Print("[SYSTEM] Cleaning up handles...");
    if(sub_ptr != 0) zmq_close(sub_ptr);
    if(rep_ptr != 0) zmq_close(rep_ptr);
    if(ctx_ptr != 0) zmq_ctx_term(ctx_ptr);
}