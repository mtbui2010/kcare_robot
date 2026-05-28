from robot_mqtt_bridge import MqttClient

client = MqttClient(bPrint=True, mqtt_ip="192.168.1.200")
try:
    # client.actionWork("move", "counter")
    # client.actionWork("lift", 0.6)
    ret = client.commWork("", timeout=180)
    print(ret)
finally:
    client.close()