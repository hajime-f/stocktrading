import os
from dotenv import load_dotenv

class Context:

    def __init__(self):

        # Reading environment variables from .env file
        load_dotenv()

        # Please set the API password in the .env file using the following format: APIPassword=XXXX
        try:
            self.api_password = os.getenv('APIPassword_production')
        except KeyError:
            print('API password not found in environment variables.')
            exit(1)
        except Exception as e:
            print(e)

        # Please set the order password in the .env file using the following format: OrderPassword=XXXX
        try:
            self.order_password = os.getenv('OrderPassword')
        except KeyError:
            print('Order password not found in environment variables.')
            exit(1)
        except Exception as e:
            print(e)

        # Please set the IP address in the .env file using the following format: IPAddress=192.168.0.3
        try:
            self.ip_address = os.getenv('IPAddress')
        except KeyError:
            print('IP address not found in environment variables.')
            exit(1)
        except Exception as e:
            print(e)

        # Please set the port number in the .env file using the following format: Port=:18080
        try:
            self.port = os.getenv('Port')
        except KeyError:
            print('Port number not found in environment variables.')
            exit(1)
        except Exception as e:
            print(e)

        # URL of kabu station server
        self.base_url = 'http://' + self.ip_address + self.port + '/kabusapi/'




