import can
try:
    bus = can.interface.Bus(bustype='vector', channel=1, app_name='test')
    print("Vector interface opened OK")
    bus.shutdown()
except Exception as e:
    print("Vector interface error:", e)