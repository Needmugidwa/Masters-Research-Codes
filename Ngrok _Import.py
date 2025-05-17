
from pyngrok import ngrok
import flwr as fl

ngrok.set_auth_token("2wxleruomNiajmMWpmQTpgj98Qd_gd2FWgC65cFG1kiD7ps7")

public_url = ngrok.connect(addr="8081", proto="tcp")  # <<< TCP, not HTTP
print(f"Public URL: {public_url}")  # Will look like "tcp://0.tcp.ngrok.io:12345"

#fl.server.start_server(server_address="0.0.0.0:8081", config=fl.server.ServerConfig(num_rounds=2))
