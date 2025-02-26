# from dotenv import load_dotenv   #for python-dotenv method
# load_dotenv()                    #for python-dotenv method

import os 

db_host = os.environ.get('DB_HOST')
db_name = os.environ.get('DB_NAME')
db_username = os.environ.get('DB_USERNAME')
db_password = os.environ.get('DB_PASSWORD')