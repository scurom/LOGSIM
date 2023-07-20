import os

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


class Database(object):
    """
    Database class to connect to MongoDB Atlas. Of type object, can be used as a singleton.

    Attributes:
        URI (str): URI connection string to MongoDB Atlas
        DATABASE (str): Database name
        local (bool): If True, connect to local MongoDB instance (no password, no string)
    """
    local = False
    if local:
        URI = "mongodb://localhost:27017"
    else:
        # get environment variable for URI connection string
        URI = os.environ.get('LOGSIM_URI')
        """:meta hide-value:"""
    DATABASE = None
    """:meta hide-value:"""

    @staticmethod
    def initialize():
        client = MongoClient(Database.URI, server_api=ServerApi('1'))
        Database.DATABASE = client['DB']

    @staticmethod
    def insert(collection, data):
        Database.DATABASE[collection].insert_one(data)

    @staticmethod
    def find(collection, query):
        return Database.DATABASE[collection].find(query)

    @staticmethod
    def find_one(collection, query):
        return Database.DATABASE[collection].find_one(query)
