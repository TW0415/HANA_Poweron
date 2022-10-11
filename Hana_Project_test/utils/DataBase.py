from pymongo import MongoClient 
from bson.json_util import dumps
import logging


class Database:

    
    def __init__(self, mongo_path):
        self.mongo_path = mongo_path
        self.client = MongoClient(mongo_path)
        self.db = self.client.pdf_data.Contract
        
    #
    def Search(self, classes, fund):
        
        cursor = self.db.find({'class' : {"$regex" : ".*" + classes + ".*"},
                         'fund' : {"$regex" : ".*" + fund + ".*"}})
        
        try:
            result = dumps(list(cursor), ensure_ascii=False)
            
        except Exception as ex:
            logging.error(ex)

        finally:
            
            return result
